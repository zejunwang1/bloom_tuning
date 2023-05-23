deepspeed --include localhost:0 train.py \
--model_name_or_path /path/to/bloom/ \
--data_path data/train.jsonl \
--max_input_length 200 \
--max_output_length 768 \
--output_dir output \
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 16 \
--learning_rate 3e-5 \
--num_train_epochs 2 \
--lr_scheduler_type "cosine" \
--warmup_steps 2000 \
--logging_steps 10 \
--save_strategy "steps" \
--save_steps 200 \
--save_total_limit 1 \
--deepspeed deepspeed.json \
--fp16 False

