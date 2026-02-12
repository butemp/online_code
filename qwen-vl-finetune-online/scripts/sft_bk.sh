#!/bin/bash

# Distributed training configuration
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NNODES=${WORLD_SIZE:-1}
NPROC_PER_NODE=1
# DeepSpeed configuration
deepspeed=ICLR/code/online_code/qwen-vl-finetune/scripts/zero3.json

# Model configuration
# llm=/data1/qjw/ckpt/Qwen/Qwen2.5-VL-3B-Instruct  # Using HuggingFace model ID
llm=ckpt/Qwen2.5-VL-3B

# Training hyperparameters
lr=2e-5
batch_size=2
grad_accum_steps=2

# Training entry point
entry_file=ICLR/code/online_code/qwen-vl-finetune/qwenvl/train/train_qwen.py

# Dataset configuration (replace with public dataset names)
# 一阶段数据集
#datasets=offline_cap%50,vript_stage_one,shot2story_stage_one

# 二阶段数据集
datasets=shot2story_qa
#,didemo_qa
#shot2story_qa 
# Output configuration
run_name="94_pretrain_3b"
output_dir=ICLR/code/ckpt/94_pretrain_3b

# Training arguments
args="
    --deepspeed ${deepspeed} \
    --model_name_or_path "${llm}" \
    --dataset_use ${datasets} \
    --data_flatten True \
    --tune_mm_vision False \
    --tune_mm_mlp True \
    --tune_mm_llm True \
    --bf16 \
    --output_dir ${output_dir} \
    --num_train_epochs 1 \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size $((batch_size*2)) \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --max_pixels 50176 \
    --min_pixels 784 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 3000 \
    --save_total_limit 4 \
    --learning_rate ${lr} \
    --weight_decay 0 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 8 \
    --lora_train True \
    --model_max_length 10240 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --train_stage 1 \
    --run_name ${run_name}"


torchrun  \
         --nproc_per_node=${NPROC_PER_NODE} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         ${entry_file} ${args}