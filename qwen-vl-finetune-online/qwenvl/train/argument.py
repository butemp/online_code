import os
import transformers
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List
from datetime import datetime

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen2.5-VL-3B-Instruct")
    tune_mm_llm: bool = field(default=False)
    tune_mm_mlp: bool = field(default=False)
    tune_mm_vision: bool = field(default=False)

@dataclass
class DataArguments:
    dataset_use: str = field(default="")
    video_max_frames: Optional[int] = field(default=8)
    video_min_frames: Optional[int] = field(default=4)
    data_flatten: bool = field(default=False)
    base_interval: int = field(default=2)
    max_pixels: int = field(default=28 * 28 * 576)
    min_pixels: int = field(default=28 * 28 * 16)
    video_max_frame_pixels: int = field(default=32 * 28 * 28)
    video_min_frame_pixels: int = field(default=4 * 28 * 28)
    add_tags: Optional[bool] = field(default=False) # 是否为每个question和prompt添加tag标签<question-id></question-id>和<response-id></response-id>


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    mm_projector_lr: Optional[float] = None
    vision_tower_lr: Optional[float] = None
    lora_train: Optional[bool] = field(default=False)
    train_stage: Optional[int] = field(default=2) # 训练阶段，1表示第一阶段是预训练，2表示第二阶段是微调
    report_to: Optional[str] = "tensorboard"
    base_log_dir: Optional[str] = 'ICLR/code/ckpt/log_dir' # 用绝对目录吧
    run_name: Optional[str] = "default_run"
    run_time: Optional[str] = "97_fix_bug"
    logging_dir: Optional[str] = os.path.join(base_log_dir, f"{run_name}={run_time}")
    
    
