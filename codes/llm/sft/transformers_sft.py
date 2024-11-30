import torch

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq


class CMSFT:
    def __init__(self, data_path, model_path):
        self.data_path = data_path
        self.model_path = model_path

        self.tokenizer, self.model = self.load_model(self.model_path)

    @staticmethod
    def load_model(model_path):
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
        tokenizer.padding_side = 'right'
        tokenizer.chat_template = "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto", )

        return tokenizer, model

    def process_func(self, example):
        """example表示一条数据"""
        question = example["conversations"][0]["value"]
        answer = example["conversations"][1]["value"]
        messages = [{"role": "user", "content": question},
                    {"role": "assistant", "content": answer}]

        formatted_input = self.tokenizer.apply_chat_template(messages, tokenize=False)
        tokenized_output = self.tokenizer(
            formatted_input,
            truncation=True,
            max_length=2048,
            return_tensors="pt",
        )

        return {
            "input_ids": tokenized_output.input_ids.squeeze(0),
            "attention_mask": tokenized_output.attention_mask.squeeze(0),
            "labels": tokenized_output.input_ids.squeeze(0),
        }

    def sft_train(self):
        dataset = load_dataset("json", data_files=self.data_path, split="train")
        tokenized_data = dataset.map(self.process_func, remove_columns=dataset.column_names)

        args = TrainingArguments(
            output_dir="",
            per_device_train_batch_size=4,
            gradient_accumulation_steps=2,
            bf16=True,
            logging_steps=1000,
            num_train_epochs=3,
            gradient_checkpointing=True,
            save_strategy="epoch",
            save_only_model=True,
            learning_rate=1e-5,
            weight_decay=1e-4,
            lr_scheduler_type="cosine",
            warmup_steps=1000,
            seed=1234,
            save_on_each_node=True,
            report_to="wandb",
        )
        print(args)

        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=tokenized_data,
            data_collator=DataCollatorForSeq2Seq(tokenizer=self.tokenizer, padding=True),
        )
        trainer.train()
        trainer.save_model()

    def inference(self, messages):
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.model.device)
        input_length = inputs.shape[1]
        # self.tokenizer.eos_token_id = 159251
        outputs = self.model.generate(input_ids=inputs, max_new_tokens=512)
        outputs = outputs[:, input_length:]
        response = self.tokenizer.batch_decode(outputs, add_special_tokens=True)  # kip_special_tokens=True)

        return response


if __name__ == '__main__':
    data_path = ''
    model_path = ''

    cm_sft = CMSFT(data_path, model_path)
    cm_sft.sft_train()

    # messages = [{"role": "user", "content": "你是谁？"}]
    # print(cm_sft.inference(messages))
