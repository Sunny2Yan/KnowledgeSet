import dataclasses
import inspect
import warnings
from functools import wraps
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from accelerate.state import PartialState
from datasets import Dataset
from datasets.arrow_writer import SchemaInferenceError
from datasets.builder import DatasetGenerationError
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollator,
    DataCollatorForLanguageModeling,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)
from transformers.modeling_utils import unwrap_model
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalPrediction

from ..extras.dataset_formatting import get_formatting_func_from_dataset
from ..import_utils import is_peft_available
from .utils import (
    ConstantLengthDataset,
    DataCollatorForCompletionOnlyLM,
    neftune_post_forward_hook,
    peft_module_casting_to_bf16,
    trl_sanitze_kwargs_for_tagging,
)


if is_peft_available():
    from peft import PeftConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training


class SFTTrainer:
    def __init__(
        self,
        model_path,
        args: Optional[TrainingArguments] = None,
        data_collator: Optional[DataCollator] = None,  # type: ignore
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        peft_config: Optional["PeftConfig"] = None,
        dataset_text_field: Optional[str] = None,
        packing: Optional[bool] = False,
        formatting_func: Optional[Callable] = None,
        max_seq_length: Optional[int] = None,
        num_of_sequences: Optional[int] = 1024,
        chars_per_token: Optional[float] = 3.6,
        dataset_num_proc: Optional[int] = None,
        dataset_batch_size: int = 1000,
        neftune_noise_alpha: Optional[float] = None,
        dataset_kwargs: Optional[Dict] = None,
        eval_packing: Optional[bool] = None,
    ):
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.pad_token = pad_token if pad_token else tokenizer.eos_token
        tokenizer.padding_side = "right"
        max_seq_length = (max_seq_length if max_seq_length
                          else tokenizer.model_max_length)

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            load_in_8bit=False,
            device_map={"": Accelerator().local_process_index})
        model = prepare_model_for_kbit_training(model)
        if peft:
            lora_config = LoraConfig(
                r=16, lora_alpha=32, lora_dropout=0.05,
                bias="none", task_type="CAUSAL_LM", )
            model = get_peft_model(model, lora_config)

        if packing and data_collator is not None and isinstance(data_collator, DataCollatorForCompletionOnlyLM):
            raise ValueError(
                "You passed a `DataCollatorForCompletionOnlyLM` to the SFTTrainer. This is not compatible with the `packing` argument."
            )

        if is_peft_available() and peft_config is not None:
            if not isinstance(peft_config, PeftConfig):
                raise ValueError(
                    "If you want to use the PeftModel, you need to pass a PeftConfig object to the SFTTrainer."
                    f" and you passed a {type(peft_config)}."
                )

            if not isinstance(model, PeftModel):
                _support_gc_kwargs = hasattr(
                    args, "gradient_checkpointing_kwargs"
                ) and "gradient_checkpointing_kwargs" in list(
                    inspect.signature(prepare_model_for_kbit_training).parameters
                )
                gradient_checkpointing_kwargs = getattr(args, "gradient_checkpointing_kwargs", None) or {}
                if getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False):
                    prepare_model_kwargs = {
                        "use_gradient_checkpointing": getattr(args, "gradient_checkpointing", False)
                    }

                    if _support_gc_kwargs:
                        prepare_model_kwargs["gradient_checkpointing_kwargs"] = gradient_checkpointing_kwargs

                    model = prepare_model_for_kbit_training(model, **prepare_model_kwargs)

                    if args is not None:
                        args = dataclasses.replace(args, gradient_checkpointing=False)
                elif getattr(args, "gradient_checkpointing", False) and (
                    "use_reentrant" not in gradient_checkpointing_kwargs
                    or gradient_checkpointing_kwargs["use_reentrant"]
                ):
                    # For backward compatibility with older versions of transformers
                    if hasattr(model, "enable_input_require_grads"):
                        model.enable_input_require_grads()
                    else:

                        def make_inputs_require_grad(module, input, output):
                            output.requires_grad_(True)

                        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

                model = get_peft_model(model, peft_config)
                if args is not None and args.bf16 and getattr(model, "is_loaded_in_4bit", False):
                    peft_module_casting_to_bf16(model)


        self.dataset_num_proc = dataset_num_proc
        self.dataset_batch_size = dataset_batch_size

        self._trainer_supports_neftune = hasattr(args, "neftune_noise_alpha")

        if neftune_noise_alpha is not None and self._trainer_supports_neftune:
            args.neftune_noise_alpha = neftune_noise_alpha
            warnings.warn(
                "You passed a `neftune_noise_alpha` argument to the SFTTrainer, the value you passed will override the one in the `TrainingArguments`."
            )
            # self.neftune_noise_alpha is done at Trainer level
        elif not self._trainer_supports_neftune:
            self.neftune_noise_alpha = neftune_noise_alpha

        if formatting_func is None and dataset_text_field is None:
            # check if dataset has ChatML format or instruction format and is supported
            # if not stays #None
            formatting_func = get_formatting_func_from_dataset(train_dataset, tokenizer)

        if not packing:
            if dataset_text_field is None and formatting_func is None:
                raise ValueError(
                    "You passed `packing=False` to the SFTTrainer, but you didn't pass a `dataset_text_field` or `formatting_func` argument."
                )

            if data_collator is None:
                data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        # Pre-process the datasets only once per node. The remaining processes will use the cache.
        with PartialState().local_main_process_first():
            if dataset_kwargs is None:
                dataset_kwargs = {}
            if train_dataset is not None:
                train_dataset = self._prepare_dataset(
                    train_dataset,
                    tokenizer,
                    packing,
                    dataset_text_field,
                    max_seq_length,
                    formatting_func,
                    num_of_sequences,
                    chars_per_token,
                    remove_unused_columns=args.remove_unused_columns if args is not None else True,
                    **dataset_kwargs,
                )
            if eval_dataset is not None:
                _multiple = isinstance(eval_dataset, dict)
                _eval_datasets = eval_dataset if _multiple else {"singleton": eval_dataset}

                eval_packing = packing if eval_packing is None else eval_packing

                for _eval_dataset_name, _eval_dataset in _eval_datasets.items():
                    _eval_datasets[_eval_dataset_name] = self._prepare_dataset(
                        _eval_dataset,
                        tokenizer,
                        eval_packing,
                        dataset_text_field,
                        max_seq_length,
                        formatting_func,
                        num_of_sequences,
                        chars_per_token,
                        remove_unused_columns=args.remove_unused_columns if args is not None else True,
                        **dataset_kwargs,
                    )
                if not _multiple:
                    eval_dataset = _eval_datasets["singleton"]

    @wraps(Trainer.train)
    def train(self, *args, **kwargs):
        # Activate neftune right before training.
        if self.neftune_noise_alpha is not None and not self._trainer_supports_neftune:
            self.model = self._trl_activate_neftune(self.model)

        output = super().train(*args, **kwargs)

        # After training we make sure to retrieve back the original forward pass method
        # for the embedding layer by removing the forward post hook.
        if self.neftune_noise_alpha is not None and not self._trainer_supports_neftune:
            unwrapped_model = unwrap_model(self.model)
            if is_peft_available() and isinstance(unwrapped_model, PeftModel):
                embeddings = unwrapped_model.base_model.model.get_input_embeddings()
            else:
                embeddings = unwrapped_model.get_input_embeddings()

            self.neftune_hook_handle.remove()
            del embeddings.neftune_noise_alpha

        return output

    def _prepare_dataset(
        self,
        dataset,
        tokenizer,
        packing,
        dataset_text_field,
        max_seq_length,
        formatting_func,
        num_of_sequences,
        chars_per_token,
        remove_unused_columns=True,
        append_concat_token=True,
        add_special_tokens=True,
    ):
        if dataset is None:
            raise ValueError("The dataset should not be None")

        # check if torch dataset / dataloader and do nothing
        if isinstance(dataset, (torch.utils.data.IterableDataset, torch.utils.data.Dataset, ConstantLengthDataset)):
            return dataset

        if not packing:
            return self._prepare_non_packed_dataloader(
                tokenizer,
                dataset,
                dataset_text_field,
                max_seq_length,
                formatting_func,
                add_special_tokens,
                remove_unused_columns,
            )

        else:
            return self._prepare_packed_dataloader(
                tokenizer,
                dataset,
                dataset_text_field,
                max_seq_length,
                num_of_sequences,
                chars_per_token,
                formatting_func,
                append_concat_token,
                add_special_tokens,
            )

    def _prepare_non_packed_dataloader(
        self,
        tokenizer,
        dataset,
        dataset_text_field,
        max_seq_length,
        formatting_func=None,
        add_special_tokens=True,
        remove_unused_columns=True,
    ):
        use_formatting_func = formatting_func is not None and dataset_text_field is None
        self._dataset_sanity_checked = False

        # Inspired from: https://huggingface.co/learn/nlp-course/chapter7/6?fw=pt
        def tokenize(element):
            outputs = tokenizer(
                element[dataset_text_field] if not use_formatting_func else formatting_func(element),
                add_special_tokens=add_special_tokens,
                truncation=True,
                padding=False,
                max_length=max_seq_length,
                return_overflowing_tokens=False,
                return_length=False,
            )

            if use_formatting_func and not self._dataset_sanity_checked:
                if not isinstance(formatting_func(element), list):
                    raise ValueError(
                        "The `formatting_func` should return a list of processed strings since it can lead to silent bugs."
                    )
                else:
                    self._dataset_sanity_checked = True

            return {"input_ids": outputs["input_ids"], "attention_mask": outputs["attention_mask"]}

        signature_columns = ["input_ids", "labels", "attention_mask"]

        extra_columns = list(set(dataset.column_names) - set(signature_columns))

        if not remove_unused_columns and len(extra_columns) > 0:
            warnings.warn(
                "You passed `remove_unused_columns=False` on a non-packed dataset. This might create some issues with the default collator and yield to errors. If you want to "
                f"inspect dataset other columns (in this case {extra_columns}), you can subclass `DataCollatorForLanguageModeling` in case you used the default collator and create your own data collator in order to inspect the unused dataset columns."
            )

        tokenized_dataset = dataset.map(
            tokenize,
            batched=True,
            remove_columns=dataset.column_names if remove_unused_columns else None,
            num_proc=self.dataset_num_proc,
            batch_size=self.dataset_batch_size,
        )

        return tokenized_dataset

    def _prepare_packed_dataloader(
        self,
        tokenizer,
        dataset,
        dataset_text_field,
        max_seq_length,
        num_of_sequences,
        chars_per_token,
        formatting_func=None,
        append_concat_token=True,
        add_special_tokens=True,
    ):
        if dataset_text_field is not None or formatting_func is not None:
            if tokenizer is None:
                raise ValueError("You need to pass a tokenizer when using `dataset_text_field` with `SFTTrainer`.")

            constant_length_iterator = ConstantLengthDataset(
                tokenizer,
                dataset,
                dataset_text_field=dataset_text_field,
                formatting_func=formatting_func,
                seq_length=max_seq_length,
                infinite=False,
                num_of_sequences=num_of_sequences,
                chars_per_token=chars_per_token,
                eos_token_id=tokenizer.eos_token_id,
                append_concat_token=append_concat_token,
                add_special_tokens=add_special_tokens,
            )

            def data_generator(constant_length_iterator):
                yield from constant_length_iterator

            try:
                packed_dataset = Dataset.from_generator(
                    data_generator, gen_kwargs={"constant_length_iterator": constant_length_iterator}
                )
            except (DatasetGenerationError, SchemaInferenceError) as exc:
                raise ValueError(
                    "Error occurred while packing the dataset. "
                    "Make sure that your dataset has enough samples to at least yield one packed sequence."
                ) from exc
            return packed_dataset
        else:
            raise ValueError(
                "You need to pass a `dataset_text_field` or `formatting_func` argument to the SFTTrainer if you want to use the `ConstantLengthDataset`."
            )

    def _trl_activate_neftune(self, model):
        r"""
        Activates the neftune as presented in this code: https://github.com/neelsjain/NEFTune and paper: https://arxiv.org/abs/2310.05914
        Since in transformers Trainer we do have an `_activate_neftune` method, we need to rename this method to avoid conflicts.
        """
        unwrapped_model = unwrap_model(model)
        if is_peft_available() and isinstance(unwrapped_model, PeftModel):
            embeddings = unwrapped_model.base_model.model.get_input_embeddings()
        else:
            embeddings = unwrapped_model.get_input_embeddings()

        embeddings.neftune_noise_alpha = self.neftune_noise_alpha
        hook_handle = embeddings.register_forward_hook(neftune_post_forward_hook)
        self.neftune_hook_handle = hook_handle
        return model