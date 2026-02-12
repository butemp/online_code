import os
import re
import json
import glob
import math
import random
import pprint
import sys

sys.path.append("ICLR/code/online_code/qwen-vl-finetune/qwenvl/data")

from tqdm import tqdm
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from utils import trans_offline_to_online_for_stage_one, trans_offline_to_online_for_stage_two


prompt_for_stage_one = """Please describe the content of the streaming video at the appropriate time during the video streaming."""

FRAME_DIR = {
    "shot2story": "ICLR/data/shot2story-videos/frames",
    "vript": "ICLR/data/Vript/vript_long_frames",
    "virpt": "ICLR/data/Vript/vript_long_frames",
    "charades": "ICLR/data/charades/frames",
    "youcook2": "ICLR/data/youcook2/frames",
    "didemo": "ICLR/data/didemo/frames",
    "ego4d": "ICLR/data/ego/ego/v2/full_scale",
    "hivau-ucf": "ICLR/data/hivau-ucf/frames/train",
    "hivau-xd": "ICLR/data/hivau-xd/frames/train"
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

def get_target_fps_frame_list(all_frame_paths, original_fps, target_fps):
    """
    从高密度帧列表中，精确降采样到目标帧率（支持整数和小数 fps）。

    Args:
        all_frame_paths (list): 原始的、高密度的帧文件路径列表。
        original_fps (float): 原始帧率。
        target_fps (float): 目标帧率。

    Returns:
        list: 一个代表 target_fps 时间流的帧文件路径列表。
    """
    if not all_frame_paths:
        return []

    all_frame_paths.sort()

    if target_fps > original_fps:
        raise ValueError("目标帧率 (target_fps) 不能高于原始帧率 (original_fps)。")

    # 视频时长（秒）
    duration = len(all_frame_paths) / original_fps

    # 目标帧总数 = 时长 * 目标fps
    target_count = int(math.floor(duration * target_fps + 1e-6))

    result = []
    for i in range(target_count):
        # 当前目标帧对应的时间点（秒）
        t = i / target_fps
        # 找到最接近该时间点的原始帧索引
        idx = int(round(t * original_fps))
        idx = min(idx, len(all_frame_paths) - 1)
        result.append(all_frame_paths[idx])

    return result

def map_qa_to_frames_at_target_fps_for_multurn(qa_item, all_frame_paths, original_fps=1.0, target_fps=1.0):
    # 1. 从QA数据中提取必要信息
    video_start_time = qa_item.get("video_start_time")
    video_end_time = qa_item.get("video_end_time")
    
    if video_start_time is None or video_end_time is None:
        print(f"错误: QA数据缺少 'video_start_time' 或 'video_end_time' 字段。")
        return {}

    # 2. 将原始帧列表降采样到我们指定的目标帧率
    all_frame_paths = [os.path.join(all_frame_paths, img) for img in os.listdir(all_frame_paths)]
    # print(f"原始帧列表有 {len(all_frame_paths)} 帧 (采样率: {original_fps}fps)。正在降采样至 {target_fps}fps...")
    target_fps_paths = get_target_fps_frame_list(all_frame_paths, original_fps, target_fps)
    
    if not target_fps_paths:
        print("错误：未能获取到目标帧率的基准帧列表。")
        return {}
    
    # 3. 根据时间段，从目标帧率列表中切出我们关心的片段
    # 列表的索引 * (1/target_fps) = 时间戳
    start_index = math.floor(video_start_time * target_fps)
    end_index = math.ceil(video_end_time * target_fps)
    
    relevant_target_frames = target_fps_paths[start_index : end_index + 1]

    # 初始化输出字典
    frame_to_qa_map = {path: [] for path in relevant_target_frames}
    
    # 4. 收集所有需要映射的Q和A事件
    events_to_map = qa_item['qa_list']

    # 5. 遍历并映射每一个事件到目标帧率的时间轴上
    for event in events_to_map:
        event_time = event['verified_timestamp']
        
        # 核心规则：向上取整来找到目标帧的序号
        # 例如：t=5.2s, target_fps=2 -> ceil(5.2*2) = ceil(10.4) = 11。即映射到第11个2fps的帧上
        if event_time <= 0.0:
             target_frame_index = 0
        else:
            target_frame_index = int(math.ceil(event_time * target_fps))
        
        # 边界处理
        max_frame_index = len(target_fps_paths) - 1
        target_frame_index = min(target_frame_index, max_frame_index)
        
        # 在目标帧率列表中找到对应的帧路径
        target_frame_path = target_fps_paths[target_frame_index]
        
        if target_frame_path in frame_to_qa_map:
            frame_to_qa_map[target_frame_path].append(event)

    return relevant_target_frames, frame_to_qa_map



def map_qa_to_frames_at_target_fps(qa_item, all_frame_paths, original_fps=1.0, target_fps=1.0):
    # 1. 从QA数据中提取必要信息
    video_start_time = qa_item.get("video_start_time")
    video_end_time = qa_item.get("video_end_time")
    
    if video_start_time is None or video_end_time is None:
        print(f"错误: QA数据缺少 'video_start_time' 或 'video_end_time' 字段。")
        return {}

    # 2. 将原始帧列表降采样到我们指定的目标帧率
    all_frame_paths = [os.path.join(all_frame_paths, img) for img in os.listdir(all_frame_paths)]
    # print(f"原始帧列表有 {len(all_frame_paths)} 帧 (采样率: {original_fps}fps)。正在降采样至 {target_fps}fps...")
    target_fps_paths = get_target_fps_frame_list(all_frame_paths, original_fps, target_fps)
    
    if not target_fps_paths:
        print("错误：未能获取到目标帧率的基准帧列表。")
        return {}
    
    # 3. 根据时间段，从目标帧率列表中切出我们关心的片段
    # 列表的索引 * (1/target_fps) = 时间戳
    start_index = math.floor(video_start_time * target_fps)
    end_index = math.ceil(video_end_time * target_fps)
    
    relevant_target_frames = target_fps_paths[start_index : end_index + 1]

    # 初始化输出字典
    frame_to_qa_map = {path: [] for path in relevant_target_frames}
    
    # 4. 收集所有需要映射的Q和A事件
    events_to_map = []
    
    # 50%的概率为0 50%的概率从[1,5]中选一个 
    if random.random() < 0.5:
        count_idx = 0
    else:
        count_idx = random.randint(1, 5)
    
    if 'question_time' in qa_item:
        events_to_map.append({
            "type": "question", "timestamp": qa_item['question_time'], "text": qa_item['question'], "count": count_idx
        })
    
    if 'answer' in qa_item and qa_item['answer']:
        for answer_item in qa_item['answer']:
            a_time = answer_item['verified_timestamp']
            events_to_map.append({
                "type": "answer", "timestamp": a_time, "text": answer_item['text'], "count": count_idx
            })

    # 5. 遍历并映射每一个事件到目标帧率的时间轴上
    for event in events_to_map:
        event_time = event['timestamp']
        
        # 核心规则：向上取整来找到目标帧的序号
        # 例如：t=5.2s, target_fps=2 -> ceil(5.2*2) = ceil(10.4) = 11。即映射到第11个2fps的帧上
        if event_time <= 0.0:
             target_frame_index = 0
        else:
            target_frame_index = int(math.ceil(event_time * target_fps))
        
        # 边界处理
        max_frame_index = len(target_fps_paths) - 1
        target_frame_index = min(target_frame_index, max_frame_index)
        
        # 在目标帧率列表中找到对应的帧路径
        target_frame_path = target_fps_paths[target_frame_index]
        
        if target_frame_path in frame_to_qa_map:
            frame_to_qa_map[target_frame_path].append(event)

    # 6. 对每个帧上的事件列表，按照时间戳进行排序
    for frame_path in frame_to_qa_map:
        frame_to_qa_map[frame_path].sort(key=lambda x: x['timestamp'])
        
    return relevant_target_frames, frame_to_qa_map


def process_frames_and_timestamps(existing_frames_dir, existing_fps, target_fps, qa_list):
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
            
        # print(f"已根据现有 {existing_fps}fps 的 {total_existing_frames} 帧，重采样得到 {len(new_frames_list)} 帧，帧率为 {target_fps}fps。")
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
    conversation = []
    image = []
    
    # 一开始就处理下qa_list这个字段，确保
    
    if data_dict.get("data_type", "") == "offline":
        if data_dict.get('train_stage', 1) == 1:
            return trans_offline_to_online_for_stage_one(data_args, data_dict)
        else:
            return trans_offline_to_online_for_stage_two(data_args, data_dict)
    else: 
        # 处理online训练数据
        video_id = data_dict['video_id']
        old_frame_fps = FRAME_FPS.get(data_dict['source'], 1)
        video_frame_dir = os.path.join(FRAME_DIR[data_dict['source']], video_id)
        
        stage_one_count_idx = 0
        
        sample_frame_fps = 1 # HARDCODE 先按照1fps采样
        while True:
            if data_dict.get('is_multi_turn', False) == False:
                relevant_target_frames, qa_frame_mapping = map_qa_to_frames_at_target_fps(
                    qa_item=data_dict,
                    all_frame_paths=video_frame_dir,
                    original_fps=old_frame_fps,
                    target_fps=sample_frame_fps # <--- 传入目标采样率
                )
            else: # 处理多轮对话
                relevant_target_frames, qa_frame_mapping = map_qa_to_frames_at_target_fps_for_multurn(
                    qa_item=data_dict,
                    all_frame_paths=video_frame_dir,
                    original_fps=old_frame_fps,
                    target_fps=sample_frame_fps # <--- 传入目标采样率
                )
            
            # FIXME: 这里先简单处理一下，后续可以考虑更复杂的采样方式
            if len(relevant_target_frames) <= 80:
                break
            else:
                if sample_frame_fps == 0.2:
                    return None
                sample_frame_fps = max(sample_frame_fps - 0.2, 0.2)
        
        # 处理只有一个Question的数据
        if data_dict.get('is_multi_turn', False) == False:
            for i in range(len(relevant_target_frames)):
                frame_path = relevant_target_frames[i]
                if not os.path.exists(frame_path):
                    return None
                
                if qa_frame_mapping.get(frame_path, []) != []:
                    # 当前帧有对应的qa
                    if len(qa_frame_mapping[frame_path]) != 1:
                        # 既有Q又有A，或者多个Q/A，目前不支持，直接丢弃
                        # 这在stage one的online数据中不应该出现;以及stage two中的单轮的online数据中也不应该出现
                        return None
                    qa = qa_frame_mapping[frame_path][0]
                    if qa['type'] == "question":
                        if data_args.add_tags == True:
                            stage_one_count_idx = qa['count']
                            conversation.append({
                                "from": "human",
                                "value": f"<question-{qa['count']}>{qa['text']}</question-{qa['count']}>",
                                "label": 0
                            })
                        else:
                            conversation.append({
                                "from": "human",
                                "value": f"{qa['text']}",
                                "label": 0
                            })
                        conversation.append({
                            "from": "gpt",
                            "value": "<|silent|>",
                            "label": 1
                        })    
                        conversation.append({
                            "from": "human",
                            "value": "<image>",
                            "label": 0
                        })
                        conversation.append({
                            "from": "gpt",
                            "value": "<|silent|>",
                            "label": 1
                        })
                        
                    else:
                        # answer
                        conversation.append({
                            "from": "human",
                            "value": "<image>",
                            "label": 0
                        })
                                        
                        if data_args.add_tags == True:
                            conversation.append({
                                "from": "gpt",
                                "value": f"<|response|>\n<answer-{qa['count']}>{qa['text']}</answer-{qa['count']}>",
                                "label": 1
                            })
                        else:
                            conversation.append({
                                "from": "gpt",
                                "value": "<|response|>\n" + qa['text'],
                                "label": 1
                            })
                else:
                    # 当前帧没有对应的qa时候
                    conversation.append({
                        "from": "human",
                        "value": "<image>",
                        "label": 0
                    })
                    conversation.append({
                        "from": "gpt",
                        "value": "<|silent|>",
                        "label": 1
                    })

                image.append(frame_path)
            
            # 插入end_of_streaming
            conversation.append({
                "from": "human",
                "value": "<|end_of_streaming|>",
                "label": 0
            })
            if data_dict.get("train_stage","") == 1:
                conversation.append({
                    "from": "gpt",
                    "value": f"<|response|>\n<answer-{stage_one_count_idx}>{data_dict['summary']}</answer-{stage_one_count_idx}>",
                    "label": 1
                })
            else:
                conversation.append({
                    "from": "gpt",
                    "value": "<|silent|>",
                    "label": 1
                })
            
        else: # 处理多轮对话
            for i in range(len(relevant_target_frames)):
                frame_path = relevant_target_frames[i]
                if not os.path.exists(frame_path):
                    return None
                
                if qa_frame_mapping.get(frame_path, []) != []:
                    # 当前帧有对应的qa
                    if len(qa_frame_mapping[frame_path]) != 1:
                        return None
                    for qa in qa_frame_mapping[frame_path]:
                        if qa['type'] == "query":
                            if data_args.add_tags == True:
                                stage_one_count_idx = qa['count']
                                conversation.append({
                                    "from": "human",
                                    "value": f"<question-{qa['count']}>{qa['text']}</question-{qa['count']}>",
                                    "label": 0
                                })
                            else:
                                conversation.append({
                                    "from": "human",
                                    "value": f"{qa['text']}",
                                    "label": 0
                                })
                            conversation.append({
                                "from": "gpt",
                                "value": "<|silent|>",
                                "label": 1
                            })
                            
                            conversation.append({
                                "from": "human",
                                "value": "<image>",
                                "label": 0
                            })
                            conversation.append({
                                "from": "gpt",
                                "value": "<|silent|>",
                                "label": 1
                            })
                            
                        elif qa['type'] == "answer":
                            conversation.append({
                                "from": "human",
                                "value": "<image>",
                                "label": 0
                            })
                                            
                            if data_args.add_tags == True:
                                conversation.append({
                                    "from": "gpt",
                                    "value": f"<|response|>\n<answer-{qa['count']}>{qa['text']}</answer-{qa['count']}>",
                                    "label": 1
                                })
                            else:
                                conversation.append({
                                    "from": "gpt",
                                    "value": "<|response|>\n" + qa['text'],
                                    "label": 1
                                })
                    
                else:
                    # 当前帧没有对应的qa时候
                    conversation.append({
                        "from": "human",
                        "value": "<image>",
                        "label": 0
                    })
                    conversation.append({
                        "from": "gpt",
                        "value": "<|silent|>",
                        "label": 1
                    })

                image.append(frame_path)

        return {
            "video_id": video_id,
            "image": image,
            "conversations": conversation
        }

class data_arg:
    add_tags = True
    
    
def process_item(item):
    """子进程调用的处理函数"""
    try:
        out = pre_precess_raw_data(data_args, item)
        if out is None:
            return None, f"警告⚠️ 输出为None!!"  # 跳过
        # 检查 image 数量
        image_count = 0
        for conv in out['conversations']:
            if conv['from'] == 'human' and conv['value'] == '<image>':
                image_count += 1
        
        if image_count == 0:
            return None, f"警告: 视频 {out['video_id']} 中没有 image，跳过。"
        if len(out['image']) == 0:
            return None, f"警告: 视频 {out['video_id']} 中 image 列表为空，跳过。"
        
        if image_count != len(out['image']):
            return None, f"警告: 视频 {out['video_id']} 中的 image 数量不匹配: {image_count} != {len(out['image'])}"
        return item, None  # 成功
    except Exception as e:
        return None, f"视频 {item.get('id', '未知')} 处理失败，跳过。错误信息: {e}"


if __name__ == "__main__":
    # test_file = "ICLR/train_file/final_train/stage_one/vript_stage_one.json"
    # test_file = "ICLR/train_file/data918/hivau_type1_refine.json"
    # test_file = "ICLR/train_file/final_train/multiturn/vript_type3_3_filted.json"
    # test_file = "ICLR/train_file/final_train/stage_two/shot2story/shot2story_qa_type2_filtered_refine_train_filted_fix.json"
    
    test_file = "ICLR/train_file/final_train/stage_one/shot2story_stage_one_filted_v921.json"
    # test_file = "ICLR/train_file/final_train/multiturn/ok/multi_vript_type3_3_train_ok.json"
    # test_file = "ICLR/train_file/final_train/multiturn/ok/shot2story_type3_2_train_ok.json"
    
    
    data = json.load(open(test_file, 'r'))
    random.shuffle(data)
    data_args = data_arg()
    
    total_num = len(data)
    correct_num = 0
    filtered_data = []
    
    DEBUG = False  # DEBUG模式不开多进程，方便调试
    # 开多进程
    
    if DEBUG:
        for item in tqdm(data, total=total_num):
            result, err = process_item(item)
            if err:
                print(err)
                continue
            if result:
                filtered_data.append(result)
                correct_num += 1
        print(f"总共处理了 {total_num} 个视频，其中 {correct_num} 个视频处理成功。")
        print(f"成功率: {correct_num / total_num * 100:.2f}%")
        print(f"处理后的数据量: {len(filtered_data)}")
    else:
        with ProcessPoolExecutor(max_workers=32) as executor:  # 可以改 max_workers 数量
            futures = [executor.submit(process_item, item) for item in data]

            for future in tqdm(as_completed(futures), total=total_num):
                result, err = future.result()
                if err:
                    print(err)
                    continue
                if result:
                    filtered_data.append(result)
                    correct_num += 1

        print(f"总共处理了 {total_num} 个视频，其中 {correct_num} 个视频处理成功。")
        print(f"成功率: {correct_num / total_num * 100:.2f}%")
        print(f"处理后的数据量: {len(filtered_data)}")

        if total_num == correct_num:
            print("所有视频均处理成功，无需保存。")
        else:
            if DEBUG:
                pass
            else:
                # 保存结果
                out_file = test_file.replace(".json", "_v921.json")
                with open(out_file, 'w') as f:
                    json.dump(filtered_data, f, indent=4, ensure_ascii=False)
                print(f"处理后的数据已保存到 {out_file}")