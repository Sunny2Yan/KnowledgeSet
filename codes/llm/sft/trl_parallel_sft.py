import torch
import argparse

from inspect import signature
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from trl import SFTConfig, SFTTrainer


class CMSFT:
    def __init__(self, args):
        self.args = args

        self.tokenizer, self.model = self.load_model()

    def load_model(self):
        tokenizer = AutoTokenizer.from_pretrained(
            self.args.model_path,
            trust_remote_code=self.args.trust_remote_code)
        tokenizer.padding_side = 'right'
        tokenizer.chat_template = "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"

        model = AutoModelForCausalLM.from_pretrained(
            self.args.model_path,
            trust_remote_code=self.args.trust_remote_code,
            torch_dtype=torch.bfloat16, )

        return tokenizer, model

    def process_func(self, example):
        output_texts = []
        for i in range(len(example['conversations'])):
            question = example["conversations"][i][0]["value"]
            answer = example["conversations"][i][1]["value"]
            messages = [{"role": "user", "content": question},
                        {"role": "assistant", "content": answer}]
            formatted_input = self.tokenizer.apply_chat_template(messages, tokenize=False)
            output_texts.append(formatted_input)

        return output_texts

    def sft_train(self):
        dataset = load_dataset("json", data_files=self.args.data_path, split="train")

        train_config_keys = signature(SFTConfig).parameters
        config_args = {}
        for key, param in train_config_keys.items():
            value = getattr(self.args, key, param.default)
            config_args[key] = value

        config = SFTConfig(**config_args)
        print(config)

        trainer = SFTTrainer(
            train_dataset=dataset,
            model=self.model,
            tokenizer=self.tokenizer,
            args=config,
            formatting_func=self.process_func,
        )

        trainer.train()


def parse_args():
    parser = argparse.ArgumentParser(description="Training the CM 3B model.")

    parser.add_argument("--data_path", type=str, required=True, help="Path to the training data file.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the pre-trained model directory.")
    parser.add_argument("--deepspeed", type=str, help="Path to DeepSpeed configuration file.")
    parser.add_argument("--output_dir", type=str, default="/workspace/cm-sft/ckpt/", help="Directory to save model checkpoints.")

    parser.add_argument("--seed", type=int, default=1234, help="Random seed for reproducibility.")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training.")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Maximum sequence length for the model.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8,
                        help="Batch size per device during training.")
    parser.add_argument("--train_micro_batch_size_per_gpu", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2,
                        help="Number of gradient accumulation steps.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate for the optimizer.")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for the optimizer.")
    parser.add_argument("--log_level", type=str, default="info",
                        choices=["debug", "info", "warning", "error", "critical"],
                        help="Set the logging level. Options are 'debug', 'info', 'warning', 'error', 'critical'. Default is 'info'.")
    parser.add_argument("--logging_strategy", type=str, default="steps", choices=["steps", "epoch"],
                        help="Set the strategy for logging. Options are 'steps' or 'epoch'. Default is 'steps'.")
    parser.add_argument("--logging_steps", type=int, default=1000,
                        help="Number of steps between logging training metrics.")
    parser.add_argument("--save_strategy", type=str, default="epoch", choices=["steps", "epoch"],
                        help="Set the strategy for saving checkpoints. Options are 'steps' or 'epoch'. Default is 'epoch'.")
    parser.add_argument("--save_steps", type=int, default=1000,
                        help="Number of steps between saving model checkpoints.")
    parser.add_argument("--warmup_steps", type=int, default=1000,
                        help="Number of warmup steps for learning rate scheduler.")
    parser.add_argument("--save_only_model", action="store_true",
                        help="If set, only the model is saved, excluding the optimizer and scheduler. This option is a flag and does not require a value. If specified, only the model's weights will be saved, otherwise, the full training state will be saved.")
    parser.add_argument("--save_safetensors", action="store_true",
                        help="If set, saves the model in the safetensors format. This option is a flag and does not require a value. If specified, the model will be saved in the safetensors format, otherwise the default format will be used.")

    parser.add_argument("--bf16", action="store_true", help="Weight type.")
    parser.add_argument("--packing", type=str, required=False, help="Packing strategy for training")
    parser.add_argument('--trust_remote_code', action="store_true", help='Whether to trust remote code.')

    parser.add_argument("--report_to", type=str, default="wandb", help="Reporting tool for logging (e.g., 'wandb').")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    cm_sft = CMSFT(args)
    cm_sft.sft_train()