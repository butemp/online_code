import os
import json
import random

from tqdm import tqdm

prompt_for_stage_one = """Please describe the content of the streaming video at the appropriate time during the video streaming."""

FRAME_DIR = {
    "shot2story": "ICLR/data/shot2story-videos/frames",
    "vript": "ICLR/data/Vript/vript_long_frames",
    "virpt": "ICLR/data/Vript/vript_long_frames",
    "charades": "ICLR/data/charades/frames",
    "youcook2": "ICLR/data/youcook2/frames",
    "didemo": "ICLR/data/didemo/frames",
    "ego4d": "ICLR/data/ego/ego/v2/full_scale"
}

FRAME_FPS = {
    "shot2story": 2,
    "vript": 1,
    "charades": 1,
    "youcook2": 1,
    "coin": 1,
    "didemo": 1,
    "0_30_s_academic_v0_1":1,
    'default': 1
}

from collections import defaultdict

def group_by_video(data_list):
    video2items = defaultdict(list)
    for d in data_list:
        video2items[d["video_id"]].append(d)
    return video2items

import random
import os
import math
import glob

def build_qa_list_stage2_for_single_turn(data_dict):
    """最新的处理单轮的函数"""
    # 最终返回类似[{"id":0, "timestamp":q_time, "text":data_dict['question'],"qa":"query"}]
    qa_id = 0
    if data_dict['source'] == "ego4d":
        # 对于这种数据 直接让回答的时间戳全部+1（除了最后一个）
        new_answer = [{"id":0, } for ans in answer]



def build_qa_list_stage2(data_dict):
    qa_list = []
    
    for q in data_dict.get("question", []):
        qa_list.append({
            "id": 0,
            "timestamp": float(q["time"]),
            "qa": "query",
            "text": f"<question_{q['count']}>{q['text']}</question_{q['count']}>"
        })

    for a in data_dict.get("answer", []):
        qa_list.append({
            "id": 0,
            "timestamp": float(a["start"]),
            "qa": "answer",
            "text": f"<|response|>\n<answer_{a['count']}>{a['text']}</answer_{a['count']}>"
        })

    qa_list = sorted(
        qa_list,
        key=lambda x: (x["timestamp"], 0 if x["qa"] == "query" else 1)
    )
    return qa_list

def build_qa_list_stage2_without_tag(data_dict):
    qa_list = []
    
    for q in data_dict.get("question", []):
        qa_list.append({
            "id": 0,
            "timestamp": float(q["time"]),
            "qa": "query",
            "text": f"<question_{q['count']}>{q['text']}</question_{q['count']}>"
        })

    for a in data_dict.get("answer", []):
        qa_list.append({
            "id": 0,
            "timestamp": float(a["start"]),
            "qa": "answer",
            "text": f"<|response|>\n<answer_{a['count']}>{a['text']}</answer_{a['count']}>"
        })

    qa_list = sorted(
        qa_list,
        key=lambda x: (x["timestamp"], 0 if x["qa"] == "query" else 1)
    )
    
    return qa_list


def process_frames_and_timestamps(existing_frames_dir, existing_fps, target_fps, qa_list):
    """
    根据给定的参数，执行帧重采样，并为每个时间戳找到新列表中最接近的帧。

    Args:
        existing_frames_dir (str): 已经提取出的帧目录路径。
        existing_fps (int or float): 提取现有帧时的帧率。
        target_fps (int or float): 期望得到的新的帧列表的帧率。
        timestamps_list (list): 包含时间戳（秒）的列表。

    Returns:
        tuple: 包含两个元素的元组：
               - new_frames_list (list): 新的帧列表，包含文件路径。
               - timestamp_to_frame_map (dict): 字典，键为时间戳，值为对应的帧路径。
    """
    
    # 预处理一下
    if ".mp4" in existing_frames_dir:
        existing_frames_dir = existing_frames_dir.replace(".mp4", "")
    if ".avi" in existing_frames_dir:
        existing_frames_dir = existing_frames_dir.replace(".avi", "")
    if ".mov" in existing_frames_dir:
        existing_frames_dir = existing_frames_dir.replace(".mov", "")
    if ".mkv" in existing_frames_dir:
        existing_frames_dir = existing_frames_dir.replace(".mkv", "")
    
    if not os.path.exists(existing_frames_dir):
        print(f"错误: 现有帧目录 '{existing_frames_dir}' 不存在。")
        return None, None, None

    # 第一步：根据现有帧和目标帧率，生成新的帧列表
    
    # 获取并排序所有现有帧文件
    existing_frame_files = sorted(
        glob.glob(os.path.join(existing_frames_dir, 'frame_*.jpg')),
        key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0])
    )
    
    if not existing_frame_files:
        print(f"警告: 目录 '{existing_frames_dir}' 中没有找到任何帧文件。")
        return [], {}
        
    if target_fps <= 0:
        print("警告: 目标帧率必须大于0。")
        return [], {}

    # ========== 优化点：判断帧率是否相同 ==========
    if existing_fps == target_fps:
        # print("检测到目标帧率与现有帧率相同，跳过重采样步骤。直接使用现有帧列表。")
        new_frames_list = existing_frame_files
    else:
        # 重采样逻辑
        total_existing_frames = len(existing_frame_files)
        video_duration = total_existing_frames / existing_fps
        total_new_frames = int(video_duration * target_fps) + 1 # +1 确保覆盖到最后一秒
        
        new_frames_list = []
        
        for new_frame_index in range(total_new_frames):
            # 计算新帧在视频中的时间点
            time_s = new_frame_index / target_fps
            
            # 将该时间点转换为现有帧列表中的最接近索引
            closest_existing_index = round(time_s * existing_fps)
            
            # 确保索引在有效范围内
            final_existing_index = max(0, min(closest_existing_index, total_existing_frames - 1))
            
            # 从现有列表中获取对应的帧路径并添加到新列表中
            frame_path = existing_frame_files[final_existing_index]
            new_frames_list.append(frame_path)
            
    # 第二步：为每个时间戳找到新列表中最接近的帧
    
    timestamp_to_frame_map = {}
    frame_to_timestamp_map = {}
    if not new_frames_list:
        # print("警告: 新的帧列表为空，无法进行时间戳映射。")
        return new_frames_list, timestamp_to_frame_map, frame_to_timestamp_map

    for qaidx, qa in enumerate(qa_list):
        # 将时间戳转换为新列表中的最接近索引
        closest_new_index = round(qa['timestamp'] * target_fps)
        
        # 确保索引在有效范围内
        final_new_index = max(0, min(closest_new_index, len(new_frames_list) - 1))
        
        # 从新列表中获取帧路径并存储到字典中
        frame_path = new_frames_list[final_new_index]
        timestamp_to_frame_map[qaidx] = frame_path
        if frame_path not in frame_to_timestamp_map:
            frame_to_timestamp_map[frame_path] = []
        frame_to_timestamp_map[frame_path].append(qa)

    return new_frames_list, timestamp_to_frame_map, frame_to_timestamp_map


def pre_precess_raw_data(data_args, data_dict):
    
    """预处理原始数据，将其转为chat格式"""
    
    # import debugpy
    # import os

    # # 获取当前进程的排名，默认为0
    # rank = int(os.environ.get("RANK", 0))

    # # 只让排名为0的进程启动调试监听
    # if rank == 0:
    #     debug_port = 7000  # 指定一个端口，例如6000
    #     debugpy.listen(("0.0.0.0", debug_port))
    #     print(f"  Waiting for debugger attach on port {debug_port}...")
    #     debugpy.wait_for_client()
    #     debugpy.breakpoint()
    
    
    conversation = []
    image = []

    
    if data_dict.get("data_type", "") == "offline":
        video_id = data_dict['id']
        video_frame_dir = data_dict['frame_path']
        old_frame_fps = FRAME_FPS.get(data_dict['data_source'], 1)
        
        frame_list = sorted(
            glob.glob(os.path.join(video_frame_dir, "frame_*.jpg")),
            key=lambda x: int(os.path.basename(x).split("_")[1].split(".")[0])
        )
        
        # 只允许一条额外指令
        extra_instruction = None
        for conv in data_dict.get("conversations", []):
            if conv["from"] == "human" and conv["value"].startswith("<image>"):
                extra_instruction = conv["value"].replace("<image>", "").strip()
                break

        old_frame_fps = int(old_frame_fps)
        

        def random_value(original_value):
            # 生成一个0到1之间的随机数
            if random.random() < 0.5:
                return original_value
            else:
                return 0
        # def random_value(original_value):
        #     rand = random.random()
        #     if rand < 0.3:
        #         return 0
        #     elif rand < 0.7:
        #         return original_value
        #     else:
        #         return len(frame_list) - 1
        
        insert_idx = random.randint(0, len(frame_list) - 1) if extra_instruction else -1
        
        # insert_idx有30%的概率等于0 40%的概率等于原始值 30%的概率等于最后一帧(等于最后一帧的话，就可以看作real-time的任务)
        insert_idx = random_value(insert_idx)
        
        for idx, frame in enumerate(frame_list[::old_frame_fps]):
            image.append(frame)
            if idx == insert_idx:
                conversation.append({
                    "from": "human", 
                    "value": extra_instruction + "\nPlease answer this question after the video stream has finished playing."
                })
                # 离线数据要求模型在视频流结束后回答问题
                
                # 如果不是real-time的问题，问题之后也要加上<|silent|>
                conversation.append({"from": "gpt", "value": "<|silent|>", "label": 0})
                
                # 问题一般默认插入在<image>之前
                conversation.append({"from": "human", "value": "<image>"})
            else:
                conversation.append({"from": "human", "value": "<image>"})
            conversation.append({"from": "gpt", "value": "<|silent|>", "label": 0})

        # 视频结束 加上end_of_streaming,以及最后的回答
        conversation.append({"from": "human", "value": "<|end_of_streaming|>"})
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
    else: 
        video_id = data_dict['video_id']
        old_frame_fps = FRAME_FPS.get(data_dict['source'], 1)
        video_frame_dir = os.path.join(FRAME_DIR[data_dict['source']], video_id)
        
        if data_dict.get('train_stage', 2) == 2:
            
            if data_dict['is_multi_turn'] == False:
                # 考虑单轮的情况
                pass
            # qa_list = build_qa_list_stage2_for_single_turn(data_dict)
        else:           
            if "captions" in data_dict.keys(): # 说明是一阶段预训练，一阶段的prompt是固定的
                query_list = [{"id":0, "timestamp":data_dict['captions'][0]['start'], "text":prompt_for_stage_one,"qa":"query"}]
                answer_list = [{"id":0,"timestamp":cap['end'],"text":cap['text'],"qa":"answer"} for cap in data_dict['captions']]
            else:
                # TODO:现在默认都是用end作为问题回答的时间戳
                if data_dict['source'] == "virpt" or data_dict['source'] == "youcook2":
                    q_time = float(data_dict['answer'][0]['start'])
                else:
                    q_time = 0.0
                
                query_list = [{"id":0, "timestamp":q_time, "text":data_dict['question'],"qa":"query"}]
                answer_list = [{"id":0,"timestamp":float(cap['end']),"text":cap['text'],"qa":"answer"} for cap in data_dict['answer']]
            qa_list = query_list + answer_list
            # timestamp=9999表示要放在<|end_of_streaming|>之后
            # answer_list.append({"id":0,"timestamp":9999,"text":data_dict['summary'],"qa":"answer"})
    
        # 循环遍历QA信息,将question和answer合并排序
        qa_list = sorted(qa_list, key=lambda x: x['timestamp'])
        sample_frame_fps = 1 # HARDCODE 先按照1fps采样
        while True:
            new_frames_list, qa_idx2frame, frame2_qa = process_frames_and_timestamps(video_frame_dir,old_frame_fps,sample_frame_fps,qa_list) 
            # 找到第一个帧和第二个帧的路径，然后找到其对应的索引
            start_frame_idx = new_frames_list.index(qa_idx2frame[0])
            end_frame_idx = new_frames_list.index(qa_idx2frame[len(qa_list)-1])
            
            # FIXME: 这里先简单处理一下，后续可以考虑更复杂的采样方式
            if end_frame_idx - start_frame_idx + 1 <= 80:
                break
            else:
                sample_frame_fps = max(sample_frame_fps / 2, 0.3)
        
        # 组装conversation和image
        for i in range(start_frame_idx, end_frame_idx + 1):
            frame = new_frames_list[i]
            image.append(frame)

            conversation.append({"from": "human", "value": "<image>"})

            qas = frame2_qa.get(frame, None)
            if qas is not None:
                if len(qas) == 1:
                    if qa['qa'] == "query":
                        conversation.append({"from": "human", "value": qa['text']})
                        conversation.append({"from": "gpt", "value": "<|silent|>", "label": 0})
                        conversation.append({"from": "human", "value": "<image>"})

                    elif qa['qa'] == "answer":
                        conversation.append({"from": "human", "value": "<image>"})
                        conversation.append({"from": "gpt", "value": f"<|response|>\n{qa['text']}", "label": 1})
                else: # 最多一张图片对应时间戳只有一个q+一个a
                    if qas[0]['qa'] == "query":
                        q_text = qas[0]['text']
                        a_text = qas[1]['text']
                    else:
                        q_text = qas[1]['text']
                        a_text = qas[0]['text']
                    conversation.append({"from": "human", "value": qa['text']})
                    conversation.append({"from": "gpt", "value": "<|silent|>", "label": 0})
                    conversation.append({"from": "human", "value": "<image>"})
                    conversation.append({"from": "gpt", "value": f"<|response|>\n{a_text}", "label": 1})
                    
            else:
                conversation.append({"from": "human", "value": "<image>"})
                conversation.append({"from": "gpt", "value": "<|silent|>", "label": 0})
        
        # 如果是一阶段训练数据，还需要加上最后的summary
        if data_dict.get('train_stage', 1) == 1:
            conversation.append({
                "from": "human",
                "value": "<|end_of_streaming|>"
            })
            if data_args.add_tags == True:
                conversation.append({
                    "from": "gpt",
                    "value": f"<|response|>\n<question_0>{data_dict['summary']}</question_0>"
                })
            else:
                conversation.append({
                    "from": "gpt",
                    "value": "<|response|>\n" + data_dict['summary']
                })   

        return {
            "video_id": video_id,
            "image": image,
            "conversations": conversation
        }

class data_arg:
    add_tags = True

if __name__ == "__main__":
    pass
    # shot2story_file = "ICLR/train_file/shot2story_for_stage_1_37k.json"
    # vript_file = "ICLR/train_file/final_vript_chunked_data_33k_30_120_filterd.json"
    # didemo_file = "ICLR/train_file/shot2story_qa_type1_all.json"
    test_file = "ICLR/train_file/final_train/charades_train.json"
    # test_file_filterd = "ICLR/train_file/final_train/ego4d_qa_type2_1_filterd.json"
    
    data = json.load(open(test_file, 'r'))
    correct_num = 0
    total_num = 0
    # 检测一下数据
    # data_args = {"add_tags": True}
    
    filtered_data = []
    
    data_args = data_arg()
    for item in tqdm(data):
        # item['image'] = item['video']
        total_num += 1
        try:
            out = pre_precess_raw_data(data_args, item)
        except Exception as e:
            print(f"视频 {item['id']} 处理失败，跳过。错误信息: {e}")
            continue
        
        correct_num += 1
        
        # 检查image的长度和conversation中的image数量是否一致
        image_count = 0
        for conv in out['conversations']:
            if conv['from'] == 'human' and conv['value'] == '<image>':
                image_count += 1

        assert image_count == len(out['image']), f"视频 {out['video_id']} 中的 image 数量不匹配: {image_count} != {len(out['image'])}"
        filtered_data.append(item)
        
    print(f"总共处理了 {total_num} 个视频，其中 {correct_num} 个视频处理成功。")
    print(f"成功率: {correct_num / total_num * 100:.2f}%")
    
    with open(test_file_filterd, 'w') as f:
        json.dump(filtered_data, f, indent=4)
    print(f"处理后的数据已保存到 {test_file_filterd}")