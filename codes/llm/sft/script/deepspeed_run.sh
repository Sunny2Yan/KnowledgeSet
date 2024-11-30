deepspeed --num_gpus 4 ../trl_sft.py \
--data_path /data/datasets/test.jsonl \
--model_path /data/models/qwen-3b \
--deepspeed ../configs/deepspeed_config.json \
--output_dir ../ckpt/v2/ \
--trust_remote_code \
--seed 1234 \
--num_train_epochs 3 \
--per_device_train_batch_size 4 \
--gradient_accumulation_steps 1 \
--gradient_checkpointing \
--log_level info \
--logging_strategy steps \
--logging_steps 500 \
--save_strategy epoch \
--save_only_model \
--save_safetensors \
--bf16 \
--learning_rate 1e-5 \
--lr_scheduler_type cosine \
--weight_decay 1e-4 \
--warmup_steps 1000 \
