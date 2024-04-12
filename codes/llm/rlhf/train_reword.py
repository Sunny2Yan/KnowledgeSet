# -*- coding: utf-8 -*-
import os

import torch
import evaluate
import numpy as np
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    LlamaTokenizer,
    HfArgumentParser,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.utils import PaddingStrategy


@dataclass
class ScriptArguments:
    """Define and parse arguments."""
    local_rank: Optional[int] = field(default=-1, metadata={"help": "Used for multi-gpu"})
    resume_from_checkpoint: Optional[bool] = field(
        default=False,
        metadata={"help": "If you want to resume training where it left off."},
    )
    deepspeed: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to deepspeed config if using deepspeed. You may need this if the model that you want to train doesn't fit on a single GPU."
        },
    )
    per_device_train_batch_size: Optional[int] = field(default=4)
    per_device_eval_batch_size: Optional[int] = field(default=1)
    gradient_accumulation_steps: Optional[int] = field(default=1)
    learning_rate: Optional[float] = field(default=2e-5)
    weight_decay: Optional[int] = field(default=0.001)
    seed: Optional[int] = field(default=1103)
    max_length: Optional[int] = field(default=512)
    log_freq: Optional[int] = field(default=1)
    eval_freq: Optional[int] = field(default=500)
    save_freq: Optional[int] = field(default=500)
    save_total_limit: Optional[int] = field(default=3)
    lora_r: Optional[int] = field(default=8)
    lora_alpha: Optional[int] = field(default=32)
    lora_dropout: Optional[float] = field(default=0.1)
    model_name: Optional[str] = field(default="decapoda-research/llama-7b-hf")
    dataset_name: Optional[str] = field(
        default="./data/comparison_data_v2.json",
        metadata={"help": "The dataset name"},
    )
    bf16: Optional[bool] = field(default=True)
    num_train_epochs: Optional[int] = field(default=1)
    train_subset: Optional[int] = field(
        default=0,
        metadata={"help": "The size of the subset of the training data to use"},
    )
    eval_subset: Optional[int] = field(
        default=0,
        metadata={"help": "The size of the subset of the eval data to use"},
    )
    gradient_checkpointing: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables gradient checkpointing."},
    )
    optim: Optional[str] = field(
        default="adamw_hf",
        metadata={"help": "The optimizer to use."},
    )
    lr_scheduler_type: Optional[str] = field(
        default="linear",
        metadata={"help": "The lr scheduler"},
    )
    output_dir: Optional[str] = field(
        default="./checkpoints/training_reward_model/",
        metadata={"help": "n steps to save the model"})


@dataclass
class RewardDataCollatorWithPadding:
    """define a special data collator that batches the data in j vs k format."""
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        features_j = []
        features_k = []
        for feature in features:
            features_j.append(
                {
                    "input_ids": feature["input_ids_j"],
                    "attention_mask": feature["attention_mask_j"],
                }
            )
            features_k.append(
                {
                    "input_ids": feature["input_ids_k"],
                    "attention_mask": feature["attention_mask_k"],
                }
            )
        batch_j = self.tokenizer.pad(
            features_j,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch_k = self.tokenizer.pad(
            features_k,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch = {
            "input_ids_j": batch_j["input_ids"],
            "attention_mask_j": batch_j["attention_mask"],
            "input_ids_k": batch_k["input_ids"],
            "attention_mask_k": batch_k["attention_mask"],
            "return_loss": True,
        }
        return batch


class RewardTrainer(Trainer):
    """ Define how to compute the reward loss.
    We use the InstructGPT pairwise logloss: https://arxiv.org/abs/2203.02155
    """
    def compute_loss(self, model, inputs, return_outputs=False):
        rewards_j = model(input_ids=inputs["input_ids_j"],
                          attention_mask=inputs["attention_mask_j"])[0]
        rewards_k = model(input_ids=inputs["input_ids_k"],
                          attention_mask=inputs["attention_mask_k"])[0]
        loss = -nn.functional.logsigmoid(rewards_j - rewards_k).mean()
        if return_outputs:
            return loss, {"rewards_j": rewards_j, "rewards_k": rewards_k}
        return loss


class LlamaRewardTrainer:
    DEFAULT_PAD_TOKEN = "[PAD]"
    DEFAULT_EOS_TOKEN = "</s>"
    DEFAULT_BOS_TOKEN = "</s>"
    DEFAULT_UNK_TOKEN = "</s>"

    def __init__(self):
        args, self.output_name = self.get_args()
        self.seed = args.seed
        set_seed(self.seed)
        self.model_name = args.model_name

        peft_config, training_args = self.get_config(args, self.output_name)

        train_dataset, eval_dataset = self.load_datasets(
            args.dataset_name, args.seed, args.train_subset, args.eval_subset)

        self.tokenizer, self.model = self.load_model(peft_config)
        self.model.print_trainable_parameters()
        self.model.config.use_cache = args.gradient_checkpointing

        self.train(train_dataset, eval_dataset, args.max_length, training_args,
                   args.resume_from_checkpoint, self.output_name)

    @staticmethod
    def get_args():
        parser = HfArgumentParser(ScriptArguments)
        script_args = parser.parse_args_into_dataclasses()[0]

        model_name_split = script_args.model_name.split("/")[-1]
        output_name = (
            f"{model_name_split}_peft_gpt-4-llm_rm_{script_args.train_subset}_{script_args.learning_rate}")

        return script_args, output_name

    @staticmethod
    def get_config(script_args, output_name):
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=script_args.lora_r,
            lora_alpha=script_args.lora_alpha,
            lora_dropout=script_args.lora_dropout,
        )

        # 定义训练参数。如果使用deepspeed，则需要在加载模型之前完成。
        training_args = TrainingArguments(
            output_dir=os.path.join(script_args.output_dir, output_name),
            learning_rate=script_args.learning_rate,
            per_device_train_batch_size=script_args.per_device_train_batch_size,
            per_device_eval_batch_size=script_args.per_device_eval_batch_size,
            num_train_epochs=script_args.num_train_epochs,
            weight_decay=script_args.weight_decay,
            evaluation_strategy="steps",
            eval_steps=script_args.eval_freq,
            save_strategy="steps",
            save_steps=script_args.save_freq,
            save_total_limit=script_args.save_total_limit,
            gradient_accumulation_steps=script_args.gradient_accumulation_steps,
            gradient_checkpointing=script_args.gradient_checkpointing,
            deepspeed=script_args.deepspeed,
            local_rank=script_args.local_rank,
            remove_unused_columns=False,
            label_names=[],
            bf16=script_args.bf16,
            logging_strategy="steps",
            logging_steps=script_args.log_freq,
            optim=script_args.optim,
            lr_scheduler_type=script_args.lr_scheduler_type,
        )

        return peft_config, training_args

    def load_model(self, peft_config=None):
        # 1. load tokenizer
        if "decapoda" in self.model_name.lower():
            tokenizer = LlamaTokenizer.from_pretrained(self.model_name)
            # required for llama
            tokenizer.add_special_tokens({
                "eos_token": self.DEFAULT_EOS_TOKEN,
                "bos_token": self.DEFAULT_BOS_TOKEN,
                "unk_token": self.DEFAULT_UNK_TOKEN,
                "pad_token": self.DEFAULT_PAD_TOKEN,
                })
        else:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if getattr(tokenizer, "pad_token", None) is None:
                tokenizer.pad_token = tokenizer.eos_token

        # 2. load model
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=1, torch_dtype=torch.bfloat16
        )
        model = get_peft_model(model, peft_config)

        return tokenizer, model

    @staticmethod
    def load_datasets(dataset_name, seed, train_subset, eval_subset):
        data_path = dataset_name
        if data_path.endswith(".json") or data_path.endswith(".jsonl"):
            dataset = load_dataset("json", data_files=data_path, split="train")
        else:
            dataset = load_dataset(data_path, split="train")
        dataset = dataset.train_test_split(test_size=0.1, seed=seed)

        train_data, eval_data = dataset["train"], dataset["test"]
        if train_subset > 0:
            train_data = train_data.select(range(train_subset))
        if eval_subset > 0:
            eval_data = eval_data.select(range(eval_subset))

        return train_data, eval_data

    def train(self, train_dataset, eval_dataset, max_length,
              training_args, resume_from_checkpoint, output_dir):

        num_proc = 24  # Can adjust to be higher if you have more processors.
        original_columns = train_dataset.column_names
        # 预处理数据集并过滤掉长度大于 max_length 的 QAs
        train_dataset = train_dataset.map(
            self.preprocess_function, batched=True, num_proc=num_proc,
            remove_columns=original_columns
        )
        train_dataset = train_dataset.filter(
            lambda x: len(x["input_ids_j"]) <= max_length and len(
                x["input_ids_k"]) <= max_length)
        eval_dataset = eval_dataset.map(
            self.preprocess_function, batched=True, num_proc=num_proc,
            remove_columns=original_columns
        )
        eval_dataset = eval_dataset.filter(
            lambda x: len(x["input_ids_j"]) <= max_length and len(
                x["input_ids_k"]) <= max_length)

        # Train the model
        trainer = RewardTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=self.compute_metrics,
            data_collator=RewardDataCollatorWithPadding(
                tokenizer=self.tokenizer, max_length=max_length),
        )

        trainer.train(resume_from_checkpoint)

        print("Saving last checkpoint of the model")
        self.model.save_pretrained(output_dir + "peft_last_checkpoint")

    @staticmethod
    def compute_metrics(eval_pred):
        # Define the metric that we'll use for validation.
        accuracy = evaluate.load("accuracy")

        predictions, _ = eval_pred
        # Here, predictions is rewards_j and rewards_k.
        # We want to see how much of the time rewards_j > rewards_k.
        predictions = np.argmax(predictions, axis=0)
        labels = np.zeros(predictions.shape)
        return accuracy.compute(predictions=predictions, references=labels)

    def preprocess_function(self, examples):
        # Turn the dataset into pairs of post + summaries,
        # text_j 是首选的 question + answer，text_k 是其他的. Then tokenize the dataset.
        new_examples = {
            "input_ids_j": [],
            "attention_mask_j": [],
            "input_ids_k": [],
            "attention_mask_k": [],
        }
        for question, response_j, response_k in zip(
                examples["user_input"], examples["completion_a"],
                examples["completion_b"]):
            tokenized_j = self.tokenizer(question + response_j, truncation=True)
            tokenized_k = self.tokenizer(question + response_k, truncation=True)

            new_examples["input_ids_j"].append(tokenized_j["input_ids"])
            new_examples["attention_mask_j"].append(
                tokenized_j["attention_mask"])
            new_examples["input_ids_k"].append(tokenized_k["input_ids"])
            new_examples["attention_mask_k"].append(
                tokenized_k["attention_mask"])

        return new_examples