import os
import json
import random

prompt_list = """As a narrator, please describe what’s happening in the live video, but only output when there is a significant change in content.
Whenever you see a new video segment, if there's clear progress in the story, continue the narration from the previous part; skip output if nothing has changed.
Keep tracking the progress of the live video, and only provide narration when there’s a noticeable change in scene, event, or character, continuing from earlier context.
You are watching a livestream—describe the story as it unfolds, but only speak when something important or new happens.
Watch the video stream as a storyteller, and only output a coherent narration when a plot twist or key event occurs.
As the livestream progresses, accumulate your understanding, and summarize only when there is a substantial shift in content.
Adopt a commentator’s perspective while watching; continue narrating the livestream, but remain silent when things are static—speak only when major changes occur.
Build a continuous narration of the livestream, but update your output only when there's a scene change, character shift, or unexpected event.
You are following a livestream; narrate in chronological order, but only speak when there is a significant shift in the video content.
Whenever a new scene, event, or topic appears in the video, continue the narration from before; otherwise, don’t output anything.
Narrate the livestream as if it were a documentary, and only provide updates when meaningful developments occur.
Track and report the key progress in the livestream, and only begin a new narration when the content clearly differs from the last segment.
You are telling the story of a livestream—only continue your narration when the content transitions or major events unfold.
Act as a professional commentator for the video stream, maintaining a continuous narrative, but update only when there's a clear change in content.
Observe how the video unfolds, and provide narration only when new events happen, environments shift, or the plot develops.
While watching the livestream, gradually build a narrative, but only add new content when key frames or significant events appear.
When new and important information appears in the video, continue narrating from earlier context; otherwise, do not output.
You are analyzing a live video stream—begin narration only when you detect a scene switch or a core event.
Turn the livestream into a coherent story, unfolding in time; continue the narration only when changes in the content occur.
Treat the video stream like a news report—only output when a new phase begins or a notable event happens."""

prompt_list = prompt_list.split("\n")

def process_online_data(data_dict) -> dict:
    """处理online格式的数据 转为传统的chat格式"""
    # 1. 找到所有的图片
    video_frame_dir = os.path.join("/data/qjw/workdirs/data/youcook2/video_frames_1fps", data_dict["video_id"])
    start = data_dict['data'][0]['start']
    end = data_dict['data'][-1]['end']
    
    # 2. 获取图片列表
    image_list = [os.path.join(video_frame_dir, f"frame_{(idx+1):04d}.jpg") for idx in range(start, end + 1)]
    data_dict['image'] = image_list
    
    # 3. 组装成传统的chat格式
    conversation = []
    idx = random.randint(0,19)
    first_meesage = {
        "from": "human",
        "value": prompt_list[idx], # 随机选择一条prompt
    }
    conversation.append(first_meesage)
    
    for i, item in enumerate(data_dict['data']):
        start = item['start']
        end = item['end']
        text = item['summary']
        
        if i != 0:
            conversation.append({
                "from": "human",
                "value": "<image>"
            })
            conversation.append({
                "from": "gpt",
                "value": f"<|response|>\n{data_dict['data'][i-1]['summary']}",
                "label": 1
            })
            # conversation.append({
            #     "from": "gpt",
            #     "value": data_dict['data'][i-1]['summary']
            # })
            end = item['end'] - 1
        
        for idx in range(start, end + 1):
            conversation.append({
                "from": "human",
                "value": "<image>"
            })
            conversation.append({
                "from": "gpt",
                "value": "<|slient|>",
                "label": 0
            })
        if i == len(data_dict['data']) - 1:
            # 处理最后一个summary
            conversation.append({
                "from": "human",
                "value": "<|end_of_streaming|>",
                "label": 1
            })
            conversation.append({
                "from": "gpt",
                "value": text
            })
    return {
        "video_id": data_dict["video_id"],
        "image" : image_list,
        "conversations": conversation
    }
        
if __name__ == "__main__":
    # 测试代码
    file = "/data/qjw/workdirs/Qwen2.5-VL-main/qwen-vl-finetune/demo/online_data.json"
    with open(file, 'r') as f:
        data = json.load(f)
    processed_data = process_online_data(data[0])
    with open("test.json", 'w') as f:
        json.dump(processed_data, f, indent=4, ensure_ascii=False)
    # print(prompt_list)
    # for _ in range(20):
    #     idx = random.randint(0,19)
    #     print(idx)
    # print(prompt_list[idx])