import json
import argparse
from pprint import pprint

from data_online_copy import pre_precess_raw_data, data_arg

def test_one_video(json_file, index=0, video_id=None):
    # 读取数据
    with open(json_file, "r") as f:
        data = json.load(f)

    item = data[index]


    args = data_arg()
    out = pre_precess_raw_data(args, item)

    print(f"===== 测试视频 {out['video_id']} =====")
    print(f"帧数: {len(out['image'])}")
    print(f"对话轮数: {len(out['conversations'])}")
    print("=== 前几条对话示例 ===")
    pprint(out['conversations'][:100])  

    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=str, required=True, help="输入的 JSON 文件路径")
    parser.add_argument("--index", type=int, default=0, help="测试第几个视频 (默认第0个)")
    parser.add_argument("--video_id", type=str, default=None, help="按 video_id 测试")

    args = parser.parse_args()

    test_one_video(args.json, args.index, args.video_id)
