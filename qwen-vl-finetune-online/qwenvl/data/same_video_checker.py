import json
from collections import defaultdict

file_path = "ICLR/train_file/shot2story_qa_type1_all_filtered.json"

with open(file_path, "r") as f:
    data = json.load(f)

video_count = defaultdict(int)
for item in data:
    video_count[item["video_id"]] += 1

duplicate_videos = {vid: count for vid, count in video_count.items() if count > 1}

print("总共有多少个 video_id:", len(video_count))
print("有多少个 video_id 出现过多次:", len(duplicate_videos))
print("重复的例子（前20个）:", list(duplicate_videos.items())[:100])
