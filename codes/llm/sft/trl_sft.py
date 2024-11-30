import torch

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer


class CMSFT:
    def __init__(self, data_path, model_path):
        self.data_path = data_path
        self.model_path = model_path

        self.tokenizer, self.model = self.load_model(self.model_path)

    @staticmethod
    def load_model(model_path):
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, )
        tokenizer.padding_side = 'right'

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto", )

        return tokenizer, model

    def process_func(self, example):
        """修改example[""]列下的所有数据"""
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
        dataset = load_dataset("json", data_files=self.data_path, split="train")

        args = SFTConfig(
            output_dir="",
            max_seq_length=2048,
            per_device_train_batch_size=8,
            gradient_accumulation_steps=2,
            logging_steps=1000,
            num_train_epochs=3,
            gradient_checkpointing=True,
            save_strategy="epoch",
            learning_rate=1e-5,
            weight_decay=1e-4,
            lr_scheduler_type="cosine",
            warmup_steps=1000,
            seed=1234,
            save_on_each_node=True,
            report_to="wandb",
        )

        trainer = SFTTrainer(
            train_dataset=dataset,
            tokenizer=self.tokenizer,
            model=self.model,
            args=args,
            formatting_func=self.process_func,
        )
        trainer.train()


if __name__ == '__main__':
    data_path = ''
    model_path = ''

    cm_sft = CMSFT(data_path, model_path)
    cm_sft.sft_train()