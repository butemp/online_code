# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import logging
import pathlib
import torch
import transformers
import json
from typing import Dict
import shutil
import sys
from pathlib import Path
import torch.nn as nn
from typing import Dict, List, Optional, Sequence, Union, Any
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

import qwenvl.train.trainer
# import qwenvl.train.WeightedTokenTrainer
from peft import LoraConfig, get_peft_model, TaskType

from trainer import replace_qwen2_vl_attention_class

from transformers.training_args import OptimizerNames

from accelerate.utils import DistributedType

from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
)
from qwenvl.data.data_qwen import make_supervised_data_module

from qwenvl.train.argument import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
)
from transformers import AutoTokenizer, AutoProcessor, Qwen2VLImageProcessor, Trainer

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def set_model(model_args, model):
    if model_args.tune_mm_vision:
        for n, p in model.visual.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_mlp:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_llm:
        for n, p in model.model.named_parameters():
            p.requires_grad = True
        model.lm_head.requires_grad = True
    else:
        for n, p in model.model.named_parameters():
            p.requires_grad = False
        model.lm_head.requires_grad = False


# 以下是你提供的代码片段的修正版本
class WeightedTokenTrainer(Trainer):
    def __init__(self, *args, response_token_id=151668, silent_token_id=151667, response_weight=1.0, silent_weight=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.response_token_id = response_token_id
        self.silent_token_id = silent_token_id
        self.response_weight = response_weight
        self.silent_weight = silent_weight

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        重写compute_loss方法，对特定token应用不同的权重，并确保损失值量级稳定。
        """
        # 前向传播
        outputs = model(**inputs)
        logits = outputs.logits

        # 获取标签
        labels = inputs.get("labels")
        if labels is not None:
            # 上移logits和标签以对齐
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # 展平
            shift_logits = shift_logits.view(-1, self.model.config.vocab_size)
            shift_labels = shift_labels.view(-1).to(shift_logits.device)
            
            # 创建一个布尔掩码，用于标识有效标签（非 -100）
            valid_labels_mask = (shift_labels != -100)

            # 计算原始损失（无权重），用于日志记录
            original_loss_fct = nn.CrossEntropyLoss()
            original_loss = original_loss_fct(shift_logits, shift_labels)

            # 使用 reduction='none' 获取所有 token 的损失
            weighted_loss_fct = nn.CrossEntropyLoss(reduction='none')
            per_token_losses = weighted_loss_fct(shift_logits, shift_labels)

            # 创建权重张量，并设置无效标签的权重为0
            weights = torch.ones_like(per_token_losses)
            response_mask = (shift_labels == self.response_token_id)
            weights = torch.where(response_mask, self.response_weight, weights)
            silent_mask = (shift_labels == self.silent_token_id)
            weights = torch.where(silent_mask, self.silent_weight, weights)
            # 确保无效标签（-100）的权重为0，这一步很重要
            weights[~valid_labels_mask] = 0.0

            # 获取有效token总数作为分母
            num_valid_tokens = valid_labels_mask.sum().item()

            # 计算加权平均损失，分母为有效token总数
            if num_valid_tokens > 0:
                weighted_loss = (per_token_losses * weights).sum() / num_valid_tokens
            else:
                # 如果没有有效token，损失为0，避免除以零
                weighted_loss = torch.tensor(0.0, device=shift_logits.device)

            # 将损失值存储在outputs中，以便在日志中使用
            outputs.original_loss = original_loss
            outputs.weighted_loss = weighted_loss
            outputs.loss = weighted_loss
        else:
            # 如果没有标签，使用模型的默认损失
            weighted_loss = outputs.loss
            original_loss = outputs.loss

        return (weighted_loss, outputs) if return_outputs else weighted_loss

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        """
        重写log方法，添加原始loss到日志中
        """
        if hasattr(self, '_current_original_loss'):
            logs['original_loss'] = self._current_original_loss

        super().log(logs, start_time)

    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], num_items_in_batch=None
    ) -> torch.Tensor:
        """
        重写training_step方法以捕获原始loss值。
        修改函数签名以接受 num_items_in_batch 参数。
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            # 传递 num_items_in_batch 参数
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch)

        if hasattr(outputs, 'original_loss'):
            # 保存原始loss值用于日志记录
            self._current_original_loss = outputs.original_loss.item()

        if self.args.n_gpu > 1:
            loss = loss.mean()

        if self.use_apex:
            from apex import amp
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            # 传递 kwargs 参数以支持 LOMO 和 DeepSpeed
            kwargs = {}
            if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
                kwargs["learning_rate"] = self._get_learning_rate()
            if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs["scale_wrt_gas"] = False
            
            # 使用 accelerator.backward()
            self.accelerator.backward(loss, **kwargs)
            
        # 注意: 原始的 Trainer 代码在 `self.accelerator.backward(loss, **kwargs)` 之前已经对 loss 进行了归一化
        # `loss = loss / self.args.gradient_accumulation_steps`。
        # 你的代码中移除了这部分。如果你希望保留原始行为，需要考虑是否重新引入。
        # 在 `accelerator.backward` 之后返回 loss.detach() 即可。
        torch.cuda.empty_cache()
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        return loss.detach()

def train(attn_implementation="flash_attention_2"):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)
    
    # 默认使用qwen2.5-vl即可
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        attn_implementation=attn_implementation,
        torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
    )
    data_args.image_processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
    ).image_processor
    data_args.model_type = "qwen2.5vl"

    if data_args.data_flatten:
        replace_qwen2_vl_attention_class()
    model.config.use_cache = False

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    set_model(model_args, model)

    if torch.distributed.get_rank() == 0:
        model.visual.print_trainable_parameters()
        # model.model.print_trainable_parameters()

    
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    
    if training_args.lora_train:
        # 设置 LoRA 配置
        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            modules_to_save=["embed_tokens", "lm_head"]
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    
    # 使用自定义的WeightedTokenTrainer
    # trainer = WeightedTokenTrainer(
    #     model=model, 
    #     processing_class=tokenizer, 
    #     args=training_args, 
    #     response_token_id=151668,  # <|response|> token id
    #     silent_token_id=151667,    # <|silent|> token id
    #     response_weight=10.0,       # 给response token更高的权重
    #     silent_weight=0.8,         # 保持silent token的默认权重
    #     **data_module
    # )    
    
    
    trainer = Trainer(
        model=model, processing_class=tokenizer, args=training_args, **data_module
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        logging.info("checkpoint found, resume training")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    
    if training_args.lora_train:
        # 保存 LoRA 适配器
        trainer.model.save_pretrained(training_args.output_dir)
        source_path = os.path.join(model_args.model_name_or_path, "chat_template.json")
        template_path = os.path.join(training_args.output_dir, "chat_template.json")
        shutil.copy2(source_path, template_path)
    else:
        model.config.use_cache = True
        
        trainer.save_state()
        data_args.image_processor.save_pretrained(training_args.output_dir)

        source_path = os.path.join(model_args.model_name_or_path, "chat_template.json")
        template_path = os.path.join(training_args.output_dir, "chat_template.json")
        shutil.copy2(source_path, template_path)

        model.config.use_cache = True

        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
