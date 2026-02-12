import os
import copy
import json
import random
import logging
import re
import time
import math
import itertools
import ast
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List, Tuple
from io import BytesIO
import base64
from collections.abc import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from decord import VideoReader
from torchcodec.decoders import VideoDecoder
import transformers

from . import data_list
from .rope2d import get_rope_index_25, get_rope_index_2

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 151655
VIDEO_TOKEN_INDEX = 151656
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def read_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def preprocess_qwen_2_visual(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    grid_thw_image: List = [],
    grid_thw_video: List = [],
) -> Dict:
    roles = {"human": "user", "gpt": "assistant"}
    system_message = "You are a helpful assistant."

    tokenizer = copy.deepcopy(tokenizer)
    chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    tokenizer.chat_template = chat_template

    visual_replicate_index_image = 0
    visual_replicate_index_video = 0
    input_ids, targets = [], []

    for i, source in enumerate(sources):
        try:
            if roles[source[0]["from"]] != roles["human"]:
                source = source[1:]
        except:
            print(sources)

        input_id, target = [], []

        input_id += tokenizer.apply_chat_template(
            [{"role": "system", "content": system_message}]
        )
        target += [IGNORE_INDEX] * len(input_id)

        for conv in source:
            try:
                role = conv["role"]
                content = conv["content"]
            except:
                role = conv["from"]
                content = conv["value"]

            role = roles.get(role, role)
            if role == "user":
                if "<image>" in content:
                    parts = content.split("<image>")
                    new_parts = []
                    for i in range(len(parts) - 1):
                        new_parts.append(parts[i])
                        replacement = (
                            "<|vision_start|>"
                            + f"<|image_pad|>"
                            * grid_thw_image[visual_replicate_index_image]
                            + "<|vision_end|>"
                        )
                        new_parts.append(replacement)
                        visual_replicate_index_image += 1
                    new_parts.append(parts[-1])
                    content = "".join(new_parts)

                if "<video>" in content:
                    parts = content.split("<video>")
                    new_parts = []
                    for i in range(len(parts) - 1):
                        new_parts.append(parts[i])
                        replacement = (
                            "<|vision_start|>"
                            + f"<|video_pad|>"
                            * grid_thw_video[visual_replicate_index_video]
                            + "<|vision_end|>"
                        )
                        new_parts.append(replacement)
                        visual_replicate_index_video += 1
                    new_parts.append(parts[-1])
                    content = "".join(new_parts)

            conv = [{"role": role, "content": content}]
            encode_id = tokenizer.apply_chat_template(conv)
            input_id += encode_id
            if role in ["user", "system"]:
                target += [IGNORE_INDEX] * len(encode_id)
            else:
                target_mask = encode_id.copy()
                target_mask[:3] = [IGNORE_INDEX] * 3
                target += target_mask

        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        input_ids.append(input_id)
        targets.append(target)

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, data_args):
        super(LazySupervisedDataset, self).__init__()

        dataset = data_args.dataset_use.split(",")
        dataset_list = data_list(dataset)
        rank0_print(f"Loading datasets: {dataset_list}")
        self.video_max_total_pixels = getattr(
            data_args, "video_max_total_pixels", 1664 * 28 * 28
        )
        self.video_min_total_pixels = getattr(
            data_args, "video_min_total_pixels", 256 * 28 * 28
        )
        self.model_type = data_args.model_type
        if data_args.model_type == "qwen2.5vl":
            self.get_rope_index = get_rope_index_25
        else:
            self.get_rope_index = get_rope_index_2
            
        self.use_pruning = True
        self.pruning_threshold = 0.1
        self.pruning_min_tokens = 1

        list_data_dict = []

        for data in dataset_list:
            file_format = data["annotation_path"].split(".")[-1]
            if file_format == "jsonl":
                annotations = read_jsonl(data["annotation_path"])
            else:
                annotations = json.load(open(data["annotation_path"], "r"))
            sampling_rate = data.get("sampling_rate", 1.0)
            if sampling_rate < 1.0:
                annotations = random.sample(
                    annotations, int(len(annotations) * sampling_rate)
                )
                print(f"sampling {len(annotations)} examples from dataset {data}")
            else:
                rank0_print(f"dataset name: {data}")
            for ann in annotations:
                ann["data_path"] = data["data_path"]
            list_data_dict += annotations

        rank0_print(f"Total training samples: {len(list_data_dict)}")

        random.shuffle(list_data_dict)  

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.data_args.image_processor.max_pixels = data_args.max_pixels
        self.data_args.image_processor.min_pixels = data_args.min_pixels
        self.data_args.image_processor.size["longest_edge"] = data_args.max_pixels
        self.data_args.image_processor.size["shortest_edge"] = data_args.min_pixels

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if "image" in sample else 0
            length_list.append(
                sum(len(conv["value"].split()) for conv in sample["conversations"])
                + img_tokens
            )
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(
                len(conv["value"].split()) for conv in sample["conversations"]
            )
            cur_len = (
                cur_len if ("image" in sample) or ("video" in sample) else -cur_len
            )
            length_list.append(cur_len)
        return length_list

    @property
    def pre_calculated_length(self):
        if "num_tokens" in self.list_data_dict[0]:
            length_list = [sample["num_tokens"] for sample in self.list_data_dict]
            return np.array(length_list)
        else:
            print("No pre-calculated length available.")
            return np.array([1] * len(self.list_data_dict))

    def process_image_unified(self, image_file):
        processor = copy.deepcopy(self.data_args.image_processor)
        image = Image.open(image_file).convert("RGB")

        visual_processed = processor.preprocess(image, return_tensors="pt")
        image_tensor = visual_processed["pixel_values"]
        if isinstance(image_tensor, List):
            image_tensor = image_tensor[0]
        grid_thw = visual_processed["image_grid_thw"][0]
        return image_tensor, grid_thw

    def process_video(self, video_file):
        decord_video = None
        decord_attempts = 0
        max_decord_attempts = 3
        while decord_attempts < max_decord_attempts:
            try:
                decord_video = self.video_decord(video_file)
                if decord_video:
                    return decord_video
            except Exception as e:
                print(f"Decord attempt {decord_attempts + 1} failed: {e}")
                decord_attempts += 1

        torchcodec_video = None
        try:
            torchcodec_video = self.video_torchcodec(video_file)
            return torchcodec_video
        except Exception as e:
            print(f"torchcodec attempt failed: {e}")

    def video_decord(self, video_file):
        if not os.path.exists(video_file):
            print(f"File not exist: {video_file}")
        vr = VideoReader(video_file, num_threads=4)
        total_frames = len(vr)
        avg_fps = vr.get_avg_fps()
        video_length = total_frames / avg_fps
        interval = getattr(self.data_args, "base_interval", 4)

        num_frames_to_sample = round(video_length / interval)
        video_min_frames = getattr(self.data_args, "video_min_frames", 4)
        video_max_frames = getattr(self.data_args, "video_max_frames", 8)

        target_frames = min(
            max(num_frames_to_sample, video_min_frames), video_max_frames
        )
        frame_idx = np.linspace(0, total_frames - 1, target_frames, dtype=int)
        frame_idx = np.unique(frame_idx)
        video = vr.get_batch(frame_idx).asnumpy()
        return self.process_video_frames(video, frame_idx, video_length)

    def video_torchcodec(self, video_file):
        device = "cpu"  # or e.g. "cuda"
        decoder = VideoDecoder(video_file, device=device)
        total_frames = decoder.metadata.num_frames
        avg_fps = decoder.metadata.average_fps
        video_length = total_frames / avg_fps
        interval = getattr(self.data_args, "base_interval", 4)

        num_frames_to_sample = round(video_length / interval)
        video_min_frames = getattr(self.data_args, "video_min_frames", 4)
        video_max_frames = getattr(self.data_args, "video_max_frames", 8)

        target_frames = min(
            max(num_frames_to_sample, video_min_frames), video_max_frames
        )
        frame_idx = np.linspace(0, total_frames - 1, target_frames, dtype=int)
        frame_idx = np.unique(frame_idx)
        frame_batch = decoder.get_frames_at(indices=frame_idx.tolist())
        video = frame_batch.data.cpu().numpy()
        return self.process_video_frames(video, frame_idx, video_length)

    def process_video_frames(self, video, frame_idx, video_length):
        fps = len(frame_idx) / video_length
        processor = copy.deepcopy(self.data_args.image_processor)
        processor.max_pixels = self.data_args.video_max_frame_pixels
        processor.min_pixels = self.data_args.video_min_frame_pixels
        processor.size["longest_edge"] = processor.max_pixels
        processor.size["shortest_edge"] = processor.min_pixels
        video_processed = processor.preprocess(
            images=None, videos=video, return_tensors="pt"
        )
        video_tensor = video_processed["pixel_values_videos"]
        grid_thw = video_processed["video_grid_thw"][0]
        second_per_grid_ts = [
            self.data_args.image_processor.temporal_patch_size / fps
        ] * len(grid_thw)
        return video_tensor, grid_thw, second_per_grid_ts

    def _get_compression_mask(
        self,
        pixel_values: torch.FloatTensor,
        batched_num_patches: torch.LongTensor,
        grid_sizes: torch.LongTensor,
        merge_sizes: torch.LongTensor,
        modals: List[str],
        threshold: float = 0.1,
        min_tokens: int = 1,
    ) -> torch.BoolTensor:
        if pixel_values.dim() == 3 and pixel_values.size(0) == 1:
            print(f"Reshaping pixel_values from {pixel_values.shape} to ({pixel_values.size(1)}, {pixel_values.size(2)})")
            pixel_values = pixel_values.squeeze(0) 

        if pixel_values.dim() != 3 and pixel_values.dim() != 2:
            print(f"Warning: Expected 2D or 3D tensor for pixel_values, got shape {pixel_values.shape}. Reshaping if possible.")
            if pixel_values.dim() == 2:
                if pixel_values.size(1) > 0:
                    hidden_dim = pixel_values.size(1)
                else:
                    print("Error: Cannot reshape 2D tensor with empty second dimension.")
                    return torch.ones(pixel_values.size(0), dtype=torch.bool, device=pixel_values.device)
            else:
                print("Error: Cannot handle tensor with dimension other than 2 or 3.")
                return torch.ones(pixel_values.size(0), dtype=torch.bool, device=pixel_values.device)
        
        if pixel_values.dim() == 3:
            batch_size, seq_len, hidden_dim = pixel_values.shape
        else: 
            batch_size = pixel_values.size(0) 
            seq_len = 1
            hidden_dim = pixel_values.size(1)  
            
        device = pixel_values.device

        full_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
        if len(batched_num_patches) == 0 or len(grid_sizes) == 0 or len(merge_sizes) == 0 or len(modals) == 0:
            print("Warning: Empty inputs to _get_compression_mask. Returning default mask.")
            return full_mask

        min_length = min(len(batched_num_patches), len(grid_sizes), len(merge_sizes), len(modals))
        if min_length < max(len(batched_num_patches), len(grid_sizes), len(merge_sizes), len(modals)):
            print(f"Warning: Mismatched input lengths in _get_compression_mask. Using first {min_length} elements.")
            batched_num_patches = batched_num_patches[:min_length]
            grid_sizes = grid_sizes[:min_length]
            merge_sizes = merge_sizes[:min_length]
            modals = modals[:min_length]

        start_idx = 0
        masks = []
        
        for num_patches, grid_size, merge_size, modal in zip(
            batched_num_patches, grid_sizes, merge_sizes, modals
        ):
            end_idx = start_idx + num_patches
            
            if len(grid_size) == 3:
                t, h, w = grid_size
            elif len(grid_size) == 2:
                h, w = grid_size
                t = 1
            else:
                print(f"Warning: Unexpected grid_size shape: {grid_size.shape}. Using default values.")
                t, h, w = 1, grid_size[0] if len(grid_size) > 0 else 16, grid_size[-1] if len(grid_size) > 0 else 16
            
            if modal == "image" or (modal == "video" and t == 1):
                masks.append(torch.ones(num_patches, dtype=torch.bool, device=device))
            
            elif modal == "video" and num_patches > 0:
                if start_idx < pixel_values.size(0) and end_idx <= pixel_values.size(0):
                    video_tokens = pixel_values[start_idx:end_idx]

                    if video_tokens.dim() == 3 and video_tokens.size(0) == 1:
                        video_tokens = video_tokens.squeeze(0)  
                    elif video_tokens.dim() == 1:
                        video_tokens = video_tokens.unsqueeze(0)
                else:
                    print(f"Warning: Index out of bounds. start_idx={start_idx}, end_idx={end_idx}, size={pixel_values.size(0)}")
                    masks.append(torch.ones(num_patches, dtype=torch.bool, device=device))
                    start_idx = end_idx
                    continue
                try:
                    if merge_size <= 0:
                        raise ValueError(f"Invalid merge_size: {merge_size}")
                        
                    if h % merge_size != 0 or w % merge_size != 0:
                        print(f"Warning: Grid dimensions ({h},{w}) not divisible by merge_size {merge_size}")
                        h_patches = max(1, h // merge_size)
                        w_patches = max(1, w // merge_size)
                        patches_per_frame = h_patches * w_patches
                    else:
                        patches_per_frame = (h // merge_size) * (w // merge_size)
                    if video_tokens.size(0) < t * patches_per_frame:
                        print(f"Warning: Not enough tokens for reshape. Expected {t * patches_per_frame}, got {video_tokens.size(0)}")
                        t = max(1, video_tokens.size(0) // patches_per_frame)
                    token_limit = min(t * patches_per_frame, video_tokens.size(0))
                    if video_tokens.dim() == 2: 
                        frames = video_tokens[:token_limit].reshape(t, patches_per_frame, video_tokens.size(1))
                    elif video_tokens.dim() == 3: 
                        frames = video_tokens[0, :token_limit].reshape(t, patches_per_frame, video_tokens.size(2))
                    else:
                        raise ValueError(f"Unexpected video_tokens dimension: {video_tokens.dim()}, shape: {video_tokens.shape}")
                except Exception as e:
                    print(f"Warning: Error in reshaping video tokens: {e}. Using default mask.")
                    masks.append(torch.ones(num_patches, dtype=torch.bool, device=device))
                    start_idx = end_idx
                    continue
                pixel_diff = frames[1:] - frames[:-1]
                pixel_diff = torch.abs(pixel_diff).mean(dim=-1) * 255
                first_frame_diff = torch.full((1, patches_per_frame), threshold + 1, 
                                            dtype=pixel_diff.dtype, device=device)
                pixel_diff = torch.cat([first_frame_diff, pixel_diff], dim=0)

                frame_mask = pixel_diff > threshold

                for frame_idx in range(t):
                    if frame_mask[frame_idx].sum() < min_tokens:
                        frame_mask[frame_idx, :min_tokens] = True
                video_mask = frame_mask.reshape(-1)
                if len(video_mask) < num_patches:
                    padding = torch.ones(num_patches - len(video_mask), dtype=torch.bool, device=device)
                    video_mask = torch.cat([video_mask, padding], dim=0)
                elif len(video_mask) > num_patches:
                    video_mask = video_mask[:num_patches]
                
                masks.append(video_mask)
            else:
                masks.append(torch.ones(num_patches, dtype=torch.bool, device=device))

            start_idx = end_idx

        if masks:
            return torch.cat(masks, dim=0)
        else:
            return full_mask

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        num_base_retries = 3
        num_final_retries = 30

        # try the current sample first
        for attempt_idx in range(num_base_retries):
            try:
                sample = self._get_item(i)
                return sample
            except Exception as e:
                # sleep 1s in case it is a cloud disk issue
                print(f"[Try #{attempt_idx}] Failed to fetch sample {i}. Exception:", e)
                time.sleep(1)

        # try other samples, in case it is file corruption issue
        for attempt_idx in range(num_base_retries):
            try:
                next_index = min(i + 1, len(self.list_data_dict) - 1)
                # sample_idx = random.choice(range(len(self)))
                sample = self._get_item(next_index)
                return sample
            except Exception as e:
                # no need to sleep
                print(
                    f"[Try other #{attempt_idx}] Failed to fetch sample {next_index}. Exception:",
                    e,
                )
                pass

        try:
            sample = self._get_item(i)
            return sample
        except Exception as e:
            raise e

    def _get_item(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME

        # define some variables
        grid_thw_merged = None
        video_grid_thw_merged = None
        grid_thw = None
        video_grid_thw = None
        second_per_grid_ts = None

        if "image" in sources[0]:
            image_folder = self.list_data_dict[i]["data_path"]
            image_file = self.list_data_dict[i]["image"]
            if isinstance(image_file, List):
                if len(image_file) > 1:
                    image_file = [
                        os.path.join(image_folder, file) for file in image_file
                    ]
                    results = [self.process_image_unified(file) for file in image_file]
                    image, grid_thw = zip(*results)
                else:
                    image_file = image_file[0]
                    image_file = os.path.join(image_folder, image_file)
                    image, grid_thw = self.process_image_unified(image_file)
                    image = [image]
            else:
                image_file = os.path.join(image_folder, image_file)
                image, grid_thw = self.process_image_unified(image_file)
                image = [image]
            grid_thw_merged = copy.deepcopy(grid_thw)
            if not isinstance(grid_thw, Sequence):
                grid_thw_merged = [grid_thw_merged]
                grid_thw = [grid_thw]
            grid_thw_merged = [
                merged_thw.prod() // self.data_args.image_processor.merge_size**2
                for merged_thw in grid_thw_merged
            ]
        if "video" in sources[0]:
            video_file = self.list_data_dict[i]["video"]
            video_folder = self.list_data_dict[i]["data_path"]
            if isinstance(video_file, List):
                if len(video_file) > 1:
                    video_file = [
                        os.path.join(video_folder, file) for file in video_file
                    ]
                    results = [self.process_video(file) for file in video_file]
                    video, video_grid_thw, second_per_grid_ts = zip(*results)
                else:
                    video_file = video_file[0]
                    video_file = os.path.join(video_folder, video_file)
                    video, video_grid_thw, second_per_grid_ts = self.process_video(
                        video_file
                    )
                    video = [video]
            else:
                video_file = os.path.join(video_folder, video_file)
                video, video_grid_thw, second_per_grid_ts = self.process_video(
                    video_file
                )
                video = [video]
            video_grid_thw_merged = copy.deepcopy(video_grid_thw)
            if not isinstance(video_grid_thw, Sequence):
                video_grid_thw_merged = [video_grid_thw_merged]
                video_grid_thw = [video_grid_thw]
            video_grid_thw_merged = [
                merged_thw.prod() // self.data_args.image_processor.merge_size**2
                for merged_thw in video_grid_thw_merged
            ]
        chat_sources = copy.deepcopy([e["conversations"] for e in sources])
        data_dict = preprocess_qwen_2_visual(
            chat_sources,
            self.tokenizer,
            grid_thw_image=grid_thw_merged if grid_thw_merged else None,
            grid_thw_video=video_grid_thw_merged if video_grid_thw_merged else None,
        )
        position_ids, _ = self.get_rope_index(
            self.data_args.image_processor.merge_size,
            data_dict["input_ids"],
            image_grid_thw=torch.stack(grid_thw, dim=0) if grid_thw else None,
            video_grid_thw=(
                torch.stack(video_grid_thw, dim=0) if video_grid_thw else None
            ),
            second_per_grid_ts=second_per_grid_ts if second_per_grid_ts else None,
        )
        if "image" not in sources[0] and "video" not in sources[0]:
            grid_thw_merged = None
            sources = copy.deepcopy([e["conversations"] for e in sources])
            data_dict = preprocess_qwen_2_visual(
                sources, self.tokenizer, grid_thw=grid_thw_merged
            )
            position_ids = (
                torch.arange(0, data_dict["input_ids"].size(1))
                .view(1, -1)
                .unsqueeze(0)
                .expand(3, -1, -1)
            )

        data_dict["position_ids"] = position_ids
        data_dict["attention_mask"] = [data_dict["input_ids"][0].size(0)]

        if "image" in self.list_data_dict[i]:
            data_dict["pixel_values"] = torch.cat(image, dim=0)
            data_dict["image_grid_thw"] = torch.cat(
                [thw.unsqueeze(0) for thw in grid_thw], dim=0
            )
        # video exist in the data
        elif "video" in self.list_data_dict[i]:
            video_tensor = torch.cat(video, dim=0)
            video_grid_thw_tensor = torch.cat(
                [thw.unsqueeze(0) for thw in video_grid_thw], dim=0
            )
            
            if self.use_pruning and "video" in sources[0]:
                try:
                    if not video or not all(v.size(0) > 0 for v in video):
                        raise ValueError("Empty video tensors detected")
                        
                    batched_num_patches = torch.tensor([v.size(0) for v in video], device=video[0].device)

                    merge_size = getattr(self.data_args.image_processor, 'merge_size', 16)
                    if not isinstance(merge_size, int) or merge_size <= 0:
                        print(f"Warning: Invalid merge_size: {merge_size}. Using default value 16.")
                        merge_size = 16
                        
                    merge_sizes = torch.tensor([merge_size] * len(video), device=video[0].device)
                    modals = ["video"] * len(video)
                    
                    if video_tensor.dim() == 3 and video_tensor.size(0) == 1:
                        print(f"Reshaping video_tensor from {video_tensor.shape} to ({video_tensor.size(1)}, {video_tensor.size(2)})")
                        video_tensor_reshaped = video_tensor.squeeze(0)
                    else:
                        video_tensor_reshaped = video_tensor
                    
                    compression_mask = self._get_compression_mask(
                        video_tensor_reshaped,  
                        batched_num_patches,
                        video_grid_thw_tensor,
                        merge_sizes,
                        modals,
                        threshold=self.pruning_threshold,
                        min_tokens=self.pruning_min_tokens
                    )
                    
                    if compression_mask.shape[0] == video_tensor_reshaped.shape[0]:
                        selected_indices = torch.nonzero(compression_mask).squeeze(1)
                        if selected_indices.dim() == 0:
                            selected_indices = selected_indices.unsqueeze(0)
                        
                        if len(selected_indices) > 0:
                            if video_tensor.dim() == 3 and video_tensor.size(0) == 1:
                                print(f"Indexing 3D tensor with shape {video_tensor.shape} using indices with max {selected_indices.max()}")
                                data_dict["pixel_values_videos"] = video_tensor[:, selected_indices, :]
                            else:
                                data_dict["pixel_values_videos"] = video_tensor_reshaped[selected_indices]
                        else:
                            # Keep at least one token if none are selected
                            if video_tensor.dim() == 3 and video_tensor.size(0) == 1:
                                data_dict["pixel_values_videos"] = video_tensor[:, :1, :]
                            else:
                                data_dict["pixel_values_videos"] = video_tensor_reshaped[:1]

                            if compression_mask.size(0) > 0:
                                compression_mask[0] = True
                            else:
                                compression_mask = torch.ones(1, dtype=torch.bool, device=video_tensor.device)
                        

                        data_dict["original_video_shape"] = torch.tensor(video_tensor.shape)
                        data_dict["compression_mask"] = compression_mask
                    else:
                        print(f"Warning: Compression mask shape {compression_mask.shape} doesn't match video tensor shape {video_tensor_reshaped.shape}. Using all tokens.")
                        if video_tensor.dim() == 3 and video_tensor.size(0) == 1:
                            data_dict["pixel_values_videos"] = video_tensor
                        else:
                            data_dict["pixel_values_videos"] = video_tensor_reshaped
                except Exception as e:
                    print(f"Warning: Error in pruning: {e}. Using all tokens.")
                    if video_tensor.dim() == 3 and video_tensor.size(0) == 1:
                        data_dict["pixel_values_videos"] = video_tensor
                    else:
                        data_dict["pixel_values_videos"] = video_tensor_reshaped if 'video_tensor_reshaped' in locals() else video_tensor
            else:
                data_dict["pixel_values_videos"] = video_tensor
            
            data_dict["video_grid_thw"] = video_grid_thw_tensor

        return data_dict


def pad_and_cat(tensor_list):
    max_length = max(tensor.shape[2] for tensor in tensor_list)

    padded_tensors = []
    for tensor in tensor_list:
        pad_length = max_length - tensor.shape[2]
        padded_tensor = torch.nn.functional.pad(tensor, (0, pad_length), "constant", 1)
        padded_tensors.append(padded_tensor)

    stacked_tensor = torch.cat(padded_tensors, dim=1)

    return stacked_tensor


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    use_pruning: bool = True

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, position_ids = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "position_ids")
        )
        input_ids = [ids.squeeze(0) for ids in input_ids]
        labels = [ids.squeeze(0) for ids in labels]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        position_ids = pad_and_cat(position_ids)
        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        labels = labels[:, : self.tokenizer.model_max_length]
        position_ids = position_ids[:, : self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
        images = list(
            instance["pixel_values"]
            for instance in instances
            if "pixel_values" in instance
        )
        videos = list(
            instance["pixel_values_videos"]
            for instance in instances
            if "pixel_values_videos" in instance
        )
        if len(images) != 0:
            concat_images = torch.cat([image for image in images], dim=0)
            grid_thw = [
                instance["image_grid_thw"]
                for instance in instances
                if "image_grid_thw" in instance
            ]
            grid_thw = torch.cat(grid_thw, dim=0)
        else:
            concat_images = None
            grid_thw = None

        if len(videos) != 0:
            concat_videos = torch.cat([video for video in videos], dim=0)
            video_grid_thw = [
                instance["video_grid_thw"]
                for instance in instances
                if "video_grid_thw" in instance
            ]
            video_grid_thw = torch.cat(video_grid_thw, dim=0)

            compression_masks = []
            video_indices = []
            current_idx = 0
            
            for instance in instances:
                if "compression_mask" in instance and self.use_pruning:
                    mask = instance["compression_mask"]
                    compression_masks.append(mask)
                    mask_size = mask.size(0)
                    video_indices.append((current_idx, current_idx + mask_size))
                    current_idx += mask_size
                elif "pixel_values_videos" in instance:
                    video_size = instance["pixel_values_videos"].size(0)
                    video_indices.append((current_idx, current_idx + video_size))
                    current_idx += video_size
            
            if compression_masks and self.use_pruning:
                batch["compression_masks"] = compression_masks
                batch["video_indices"] = video_indices
        else:
            concat_videos = None
            video_grid_thw = None

        batch["pixel_values"] = concat_images
        batch["image_grid_thw"] = grid_thw
        batch["pixel_values_videos"] = concat_videos
        batch["video_grid_thw"] = video_grid_thw
        batch["position_ids"] = position_ids
        return batch


@dataclass
class FlattenedDataCollatorForSupervisedDataset(DataCollatorForSupervisedDataset):
    """Collate examples into packed sequence with multi-modal support."""

    tokenizer: transformers.PreTrainedTokenizer
    use_pruning: bool = True

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, position_ids, attention_mask = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "position_ids", "attention_mask")
        )
        attention_masks = []
        for instance in instances:
            if "attention_mask" in instance:
                if isinstance(instance["attention_mask"], list):
                    attention_masks.extend(instance["attention_mask"])
                elif isinstance(instance["attention_mask"], (torch.Tensor, int)):
                    attention_masks.append(instance["attention_mask"])
                else:
                    print(f"Warning: Unexpected attention_mask type: {type(instance['attention_mask'])}")
        
        if not attention_masks:
            attention_masks = [0]

        attention_mask_ints = []
        for mask in attention_masks:
            if isinstance(mask, torch.Tensor):
                attention_mask_ints.append(mask.item())
            else:
                attention_mask_ints.append(int(mask))
                
        seq_lens = torch.tensor([0] + attention_mask_ints, dtype=torch.int32)
        cumsum_seq_lens = torch.cumsum(seq_lens, dim=0, dtype=torch.int32)
        input_ids = torch.cat(input_ids, dim=1)
        labels = torch.cat(labels, dim=1)
        position_ids = torch.cat(position_ids, dim=2)

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=cumsum_seq_lens,
            position_ids=position_ids,
        )
        images = list(
            instance["pixel_values"]
            for instance in instances
            if "pixel_values" in instance
        )
        videos = list(
            instance["pixel_values_videos"]
            for instance in instances
            if "pixel_values_videos" in instance
        )
        if len(images) != 0:
            concat_images = torch.cat([image for image in images], dim=0)
            grid_thw = [
                instance["image_grid_thw"]
                for instance in instances
                if "image_grid_thw" in instance
            ]
            grid_thw = torch.cat(grid_thw, dim=0)
        else:
            concat_images = None
            grid_thw = None

        if len(videos) != 0:
            concat_videos = torch.cat([video for video in videos], dim=0)
            video_grid_thw = [
                instance["video_grid_thw"]
                for instance in instances
                if "video_grid_thw" in instance
            ]
            video_grid_thw = torch.cat(video_grid_thw, dim=0)

            compression_masks = []
            video_indices = []
            current_idx = 0
            
            for instance in instances:
                if "compression_mask" in instance and self.use_pruning:
                    mask = instance["compression_mask"]
                    compression_masks.append(mask)
                    mask_size = mask.size(0)
                    video_indices.append((current_idx, current_idx + mask_size))
                    current_idx += mask_size
                elif "pixel_values_videos" in instance:
                    video_size = instance["pixel_values_videos"].size(0)
                    video_indices.append((current_idx, current_idx + video_size))
                    current_idx += video_size
            
            if compression_masks and self.use_pruning:
                batch["compression_masks"] = compression_masks
                batch["video_indices"] = video_indices
        else:
            concat_videos = None
            video_grid_thw = None

        batch["pixel_values"] = concat_images
        batch["image_grid_thw"] = grid_thw
        batch["pixel_values_videos"] = concat_videos
        batch["video_grid_thw"] = video_grid_thw

        return batch


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer, data_args=data_args)
    use_pruning = True
    
    if data_args.data_flatten:
        data_collator = FlattenedDataCollatorForSupervisedDataset(tokenizer=tokenizer, use_pruning=use_pruning)
        return dict(
            train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
        )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer, use_pruning=use_pruning)
    return dict(
        train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
    )


if __name__ == "__main__":
    pass
