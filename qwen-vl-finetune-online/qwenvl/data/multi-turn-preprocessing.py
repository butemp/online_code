import json
import random
from collections import defaultdict

def choose_question_time(answers, 
                         p_head=0.5, 
                         p_before_first=0.3, 
                         p_middle=0.2):
    video_start = 0.0
    answers = sorted(answers, key=lambda x: float(x["start"]))
    first_ans_start = float(answers[0]["start"])
    last_ans_start  = float(answers[-1]["start"])

    mode = random.random()
    if mode < p_head:
        q_time = video_start
    elif mode < p_head + p_before_first:
        q_time = random.uniform(video_start, max(video_start, first_ans_start - 1e-3))
    elif mode < p_head + p_before_first + p_middle and len(answers) > 1:
        valid_intervals = []
        for i in range(len(answers) - 1):
            seg_start = float(answers[i]["end"])
            seg_end   = float(answers[i + 1]["start"])
            if seg_end <= last_ans_start:
                valid_intervals.append((seg_start, seg_end))
        if valid_intervals:
            seg_start, seg_end = random.choice(valid_intervals)
            q_time = random.uniform(seg_start, seg_end - 1e-3)
        else:
            q_time = random.uniform(video_start, max(video_start, first_ans_start - 1e-3))
    else:
        q_time = random.uniform(video_start, last_ans_start)

    return round(q_time, 2)


def merge_json_with_random_time(input_file, output_file):
    with open(input_file, 'r') as f:
        data = json.load(f)

    video_dict = defaultdict(list)
    for item in data:
        video_dict[item["video_id"]].append(item)

    merged_data = []
    for video_id, items in video_dict.items():
        base = {
            "source": items[0].get("source", ""),
            "id": items[0].get("id", 0),
            "video_id": video_id,
            "data_type": items[0].get("data_type", "online"),
            "train_stage": items[0].get("train_stage", 2),
            "length": items[0].get("length", ""),
            "question_category": items[0].get("question_category", "")
        }

        merged_questions = []
        merged_answers = []
        q_count = 0

        for item in items:
            answers = item.get("answer", [])
            if not answers:
                continue
            q_time = choose_question_time(answers)

            merged_questions.append({
                "time": q_time,
                "count": q_count,
                "text": item["question"]
            })

            for ans in answers:
                merged_answers.append({
                    "start": ans["start"],
                    "end": ans["end"],
                    "count": q_count,
                    "text": ans["text"]
                })

            q_count += 1

        base["question"] = merged_questions
        base["answer"] = merged_answers
        merged_data.append(base)

    with open(output_file, 'w') as f:
        json.dump(merged_data, f, indent=4)

    print(f"处理完成，保存到 {output_file}")


if __name__ == "__main__":
    input_file = "ICLR/train_file/all_youcook2_qa_type2_filtered.json"         
    output_file = "ICLR/train_file/shot2story_qa_type2_all_filtered_multi.json"     
    merge_json_with_random_time(input_file, output_file)
