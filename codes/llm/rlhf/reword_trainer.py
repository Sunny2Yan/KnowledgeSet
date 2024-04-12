# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

from datasets import Dataset
from dataclasses import dataclass, FrozenInstanceError, replace
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from transformers import (DataCollator, PreTrainedModel,
                          PreTrainedTokenizerBase, Trainer, TrainingArguments)
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_pt_utils import nested_detach
from transformers.trainer_utils import EvalPrediction
from peft import PeftModel, get_peft_model

from trl.trainer.utils import RewardDataCollatorWithPadding, compute_accuracy


@dataclass
class RewardConfig(TrainingArguments):
    max_length: Optional[int] = 512  # max_seq


class RewardTrainer(Trainer):
    r"""
    RewardTrainer 期望数据集具有特定的格式, 至少应该包含以下 4 个属性，如果不使用默认的data_collator `RewardDataCollatorWithPadding`. 属性名应命名如下：
    - `input_ids_chosen`: "question + good_answer"
    - `attention_mask_chosen`
    - `input_ids_rejected`: "question + bad_answer"
    - `attention_mask_rejected`

    Optionally, you can also pass a `margin` entry to the dataset. This entry should contain the margin used to modulate the
    loss of the reward model as outlined in https://ai.meta.com/research/publications/llama-2-open-foundation-and-fine-tuned-chat-models/.
    If you don't pass a margin, no margin will be used.
    """
    def __init__(
        self,
        model: Optional[Union[PreTrainedModel, nn.Module]] = None,
        args: Optional[RewardConfig] = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[
            torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
            None, None, ),
        preprocess_logits_for_metrics: Optional[
            Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        peft_config: Optional[Dict] = None,
    ):
        if peft_config is not None:
            if not isinstance(model, PeftModel):
                model = get_peft_model(model, peft_config)

        if compute_metrics is None:
            compute_metrics = compute_accuracy

        if data_collator is None:
            data_collator = RewardDataCollatorWithPadding(
                tokenizer, max_length=args.max_length)

            if args.remove_unused_columns:
                try:  # default=True, 但这里不移除
                    args.remove_unused_columns = False
                except FrozenInstanceError:
                    args = replace(args, remove_unused_columns=False)

            self.use_reward_data_collator = True
        else:
            self.use_reward_data_collator = False

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        rewards_chosen = model(
            input_ids=inputs["input_ids_chosen"],
            attention_mask=inputs["attention_mask_chosen"],
            return_dict=True, )["logits"]
        rewards_rejected = model(
            input_ids=inputs["input_ids_rejected"],
            attention_mask=inputs["attention_mask_rejected"],
            return_dict=True, )["logits"]

        # 计算loss (负对数Sigmoid loss), 可选是否调节边界
        if "margin" in inputs:
            loss = -nn.functional.logsigmoid(
                rewards_chosen - rewards_rejected - inputs["margin"]).mean()
        else:
            loss = -nn.functional.logsigmoid(
                rewards_chosen - rewards_rejected).mean()

        if return_outputs:
            return loss, {
                "rewards_chosen": rewards_chosen,
                "rewards_rejected": rewards_rejected, }

        return loss

    def prediction_step(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        with torch.no_grad():
            loss, logits_dict = self.compute_loss(model, inputs, return_outputs=True)

        if prediction_loss_only:
            return (loss, None, None)

        loss = loss.detach()
        logits = tuple(v for k, v in logits_dict.items() if k not in ignore_keys)
        logits = nested_detach(logits)
        # Stack accepted against rejected, mean over logits and softmax
        # to get preferences between accepted and rejected to sum to 1
        logits = torch.stack(logits).mean(dim=2).softmax(dim=0).T

        labels = torch.zeros(logits.shape[0])
        labels = self._prepare_inputs(labels)

        return loss, logits, labels