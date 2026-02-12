"""主要是将离线的QA数据处理为最终的训练格式，最终的训练格式包括实时回复、离线回复两大类"""
import json
import os
import glob
from concurrent.futures import ProcessPoolExecutor
import random
import re

FRAME_DIR = {
    "shot2story": "/groups/g900403/home/share/qjw/ICLR/data/shot2story-videos/frames",
    "vript": "/groups/g900403/home/share/qjw/ICLR/data/Vript/vript_long_frames",
    "virpt": "/groups/g900403/home/share/qjw/ICLR/data/Vript/vript_long_frames",
    "charades": "/groups/g900403/home/share/qjw/ICLR/data/charades/frames",
    "youcook2": "/groups/g900403/home/share/qjw/ICLR/data/youcook2/frames",
    "didemo": "/groups/g900403/home/share/qjw/ICLR/data/didemo/frames",
    "ego4d": "/groups/g900403/home/share/qjw/ICLR/data/ego/ego/v2/full_scale"
}

FRAME_FPS = {
    "shot2story": 2,
    "vript": 1,
    "charades": 1,
    "youcook2": 1,
    "coin": 1,
    "didemo": 1,
    "0_30_s_academic_v0_1":1,
    "ego4d": 1,
    'default': 1
}

offline_prompt = "\nPlease answer this question after the video stream has finished playing."

realtime_prompt = "\nPlease answer this question immediately based on the video stream just now."

def random_value(original_value):
    # 生成一个0到1之间的随机数
    if random.random() < 0.5:
        return original_value
    else:
        return 0

def trans_offline_to_online_for_stage_one(data_args, data_dict) -> dict: 
    conversation = []
    image = []
    video_id = data_dict['id']
    video_frame_dir = data_dict['frame_path']
    old_frame_fps = FRAME_FPS.get(data_dict['data_source'], 1)
    
    frame_list = sorted(
        glob.glob(os.path.join(video_frame_dir, "frame_*.jpg")),
        key=lambda x: int(os.path.basename(x).split("_")[1].split(".")[0])
    )
    
    while len(frame_list) > 80:
        frame_list = frame_list[::2]  # 每隔一帧取一帧
        old_frame_fps = old_frame_fps * 2
        

    # 只允许一条额外指令
    extra_instruction = None
    for conv in data_dict.get("conversations", []):
        if conv["from"] == "human":
            if conv["value"].startswith("<image>"):
                extra_instruction = conv["value"].replace("<image>", "").strip()
                break
            else:
                extra_instruction = conv["value"].strip()
                break
    
    if_real_time = random.random() < 0.5  # 50%概率选择实时回复
    if not if_real_time:
        # 等到<|end_of_streaming|>之后再回复用户的问题
        insert_idx = random.randint(0, len(frame_list) - 1)
        insert_idx = random_value(insert_idx)
        
        # 为这个问题随机挑选一个q_id
        if random.random() < 0.5:
            count_idx = 0
        else:
            count_idx = random.randint(0, 5)
            
        for idx, frame in enumerate(frame_list):
            image.append(frame)
            if idx == insert_idx:
                if data_args.add_tags == True:
                    conversation.append({
                        "from": "human", 
                        "value": f"<question-{count_idx}>" + extra_instruction + offline_prompt + f"</question-{count_idx}>",
                        "label": 0
                    })
                else:
                    conversation.append({
                        "from": "human", 
                        "value": extra_instruction + offline_prompt,
                        "label": 0
                    })
                    # 离线数据要求模型在视频流结束后回答问题
                
                # 如果不是real-time的问题，问题之后也要加上<|silent|>
                conversation.append({"from": "gpt", "value": "<|silent|>", "label": 1})
                
                # 问题一般默认插入在<image>之前
                conversation.append({"from": "human", "value": "<image>", "label": 0})
            else:
                conversation.append({"from": "human", "value": "<image>", "label": 0})
            conversation.append({"from": "gpt", "value": "<|silent|>", "label": 1})

        # 视频结束 加上end_of_streaming,以及最后的回答
        conversation.append({"from": "human", "value": "<|end_of_streaming|>","label": 0})
        if data_args.add_tags == True:
            conversation.append({
                "from": "gpt",
                "value": f"<|response|>\n<answer-{count_idx}>{data_dict['conversations'][-1]['value']}</answer-{count_idx}>",
                "label": 1
            })
        else:
            conversation.append({
                "from": "gpt",
                "value": f"<|response|>\n{data_dict['conversations'][-1]['value']}",
                "label": 1
            })
    else:
        # 实时回复,视频播放完之后(不用加end_of_streaming)用户提出问题，模型立即回答
        for idx, frame in enumerate(frame_list):
            image.append(frame)
            conversation.append({"from": "human", "value": "<image>", "label": 0})
            conversation.append({"from": "gpt", "value": "<|silent|>", "label": 1})
        
        if random.random() < 0.5:
            count_idx = 0
        else:
            count_idx = random.randint(0, 5)
            
        if data_args.add_tags == True:
            conversation.append({
                "from": "human", 
                "value": f"<question-{count_idx}>" + extra_instruction + realtime_prompt + f"</question-{count_idx}>",
                "label": 0
            })
            conversation.append({
                "from": "gpt",
                "value": f"<|response|>\n<answer-{count_idx}>{data_dict['conversations'][-1]['value']}</answer-{count_idx}>",
                "label": 1
            })
        else:
            conversation.append({
                "from": "human", 
                "value": extra_instruction + realtime_prompt,
                "label": 0
            })
            conversation.append({
                "from": "gpt",
                "value": f"<|response|>\n{data_dict['conversations'][-1]['value']}",
                "label": 1
            })
        
    if not image:
        raise ValueError(f"视频 {video_id} 没有帧，丢弃")
    if not conversation:
        raise ValueError(f"视频 {video_id} 没有对话，丢弃")
    
    return {
        "video_id": video_id,
        "image": image,
        "conversations": conversation
    }
    
def trans_offline_to_online_for_stage_two(data_args, data_dict) -> dict: 
    return trans_offline_to_online_for_stage_one(data_args, data_dict)
    
if __name__ == "__main__":
    msin()