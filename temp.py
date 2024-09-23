# 使用 DataCollatorForCompletionOnlyLM SFTTrainer 示例

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM


dataset = load_dataset("lucasmccabe-lmi/CodeAlpaca-20k", split="train")

model = AutoModelForCausalLM.from_pretrained(
    "instructlab/merlinite-7b-lab",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2")
tokenizer = AutoTokenizer.from_pretrained("instructlab/merlinite-7b-lab")
tokenizer.pad_token = tokenizer.eos_token

def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['instruction'])):
        text = f"### Question: {example['instruction'][i]}\n ### Answer: {example['output'][i]}"
        output_texts.append(text)
    return output_texts

response_template = " ### Answer:"
response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)[2:]
collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer, padding_free=True)

trainer = SFTTrainer(
    model,
    train_dataset=dataset,
    args=SFTConfig(
        output_dir="./tmp",
        gradient_checkpointing=True,
        per_device_train_batch_size=8
    ),
    formatting_func=formatting_prompts_func,
    data_collator=collator,
)

trainer.train()
