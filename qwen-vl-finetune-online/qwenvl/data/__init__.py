import re

# Define placeholders for dataset paths
CAMBRIAN_737K = {
    "annotation_path": "PATH_TO_CAMBRIAN_737K_ANNOTATION",
    "data_path": "",
}

MP_DOC = {
    "annotation_path": "PATH_TO_MP_DOC_ANNOTATION",
    "data_path": "PATH_TO_MP_DOC_DATA",
}

CLEVR_MC = {
    "annotation_path": "PATH_TO_CLEVR_MC_ANNOTATION",
    "data_path": "PATH_TO_CLEVR_MC_DATA",
}

VIDEOCHATGPT = {
    "annotation_path": "PATH_TO_VIDEOCHATGPT_ANNOTATION",
    "data_path": "PATH_TO_VIDEOCHATGPT_DATA",
}

DEMO = {
    "annotation_path": "/data/qjw/workdirs/Qwen2.5-VL-main/qwen-vl-finetune/demo/video.json",
    "data_path": "/data/qjw/workdirs/Qwen2.5-VL-main/qwen-vl-finetune/demo/videos",
}

YOUCOOK2 = {
    "annotation_path": "/data1/qjw/data/youcook2/youcook2_data.json",
    "data_path": "",
}

COIN = {
    "annotation_path": "/data1/qjw/ICLR/data/coin_data.json",
    "data_path": "",
}

COIN_STAGE2 = {
    "annotation_path": "/data1/qjw/ICLR/code_for_data/train_data/coin_stage2.json",
    "data_path": "",
}

DEBUG_DATA = {
    "annotation_path": "/data1/qjw/ICLR/code_for_data/train_data/debug_data.json",
    "data_path": "",
}



VRIPT_STAGE_ONE = {
    "annotation_path": "ICLR/train_file/final_train/stage_one/vript_stage_one_filted_v921.json",
    "data_path": "",
}

SHOT2STORY_STAGE_ONE = {
    "annotation_path": "ICLR/train_file/final_train/stage_one/shot2story_stage_one_filted_v921.json",
    "data_path": "",
}

# 二阶段QA数据

SHOT2STORY_QA_TYPE1 = {
    "annotation_path": "ICLR/train_file/final_train/stage_two/shot2story/shot2story_qa_type1_train_v3_filted_ok_v921.json",   
    "data_path": "",
}

SHOT2STORY_QA_TYPE2 = {
    "annotation_path": "ICLR/train_file/final_train/stage_two/shot2story/shot2story_qa_type2_filtered_refine_train_filted_fix_ok_v921.json",
    "data_path": "",
}


VRIPT_QA_TYPE1 = {
    "annotation_path": "ICLR/train_file/final_train/stage_two/vript/vript_type1_train_v3_filted_ok_v920_v921.json",
    "data_path": "",
}

VRIPT_QA_TYPE2 = {
    "annotation_path": "ICLR/train_file/final_train/stage_two/vript/vript_type2_train_v2_filted_ok_v920_v921.json",
    "data_path": "",
}

CHARADES_QA = {
    "annotation_path": "ICLR/train_file/final_train/stage_two/charades/charades_train_v3_filted_ok.json",
    "data_path": "",
}

DIDEMO_QA = {
    "annotation_path":"ICLR/train_file/final_train/stage_two/didemo/didemo_train_v3_filted_ok.json",
    "data_path": "",
}


EGO4D_QA_TYPE1 = {
    "annotation_path": "ICLR/train_file/final_train/stage_two/ego4d/ego4d_qa_type1_train_v3_filted_high_ok.json",
    "data_path": "",
}

EGO4D_QA_TYPE2_1 = {
    "annotation_path": "ICLR/train_file/final_train/stage_two/ego4d/ego4d_qa_type2_1_train_v3_filted_high_ok_v920.json",
    "data_path": "",
}


OFFLINE_CAP = {
    "annotation_path": "ICLR/data/LLaVA-Video-178K/all_cap_data_filtered.json",
    "data_path": "",
}

# OFFLINE_OE_QA = {
#     "annotation_path": "ICLR/data/LLaVA-Video-178K/all_oe_qa_data_split.json",
#     "data_path": "",
# }

# OFFLINE_MC_QA = {
#     "annotation_path": "ICLR/data/LLaVA-Video-178K/all_mc_qa_data_split.json",
#     "data_path": "",
# }


# 多轮对话数据
PARALLEL = {
    "annotation_path": "ICLR/train_file/final_train/multiturn/ok/multi_parallel_turn_data_train_ok_v920_v921.json",
    "data_path": "",
}

MODIFY = {
    "annotation_path": "ICLR/train_file/final_train/multiturn/ok/shot2story_type3_2_train_ok_v921.json",
    "data_path": "",
}

SEQ = {
    "annotation_path": "ICLR/train_file/final_train/multiturn/ok/multi_vript_type3_3_train_ok_v920.json",
    "data_path": "",
}




data_dict = {
    "shot2story_qa_type1": SHOT2STORY_QA_TYPE1,
    "shot2story_qa_type2": SHOT2STORY_QA_TYPE2,
    "vript_qa_type1": VRIPT_QA_TYPE1,
    "vript_qa_type2": VRIPT_QA_TYPE2,
    "charades_qa": CHARADES_QA,
    "didemo_qa": DIDEMO_QA,
    "ego4d_qa_type1": EGO4D_QA_TYPE1,
    "ego4d_qa_type2_1": EGO4D_QA_TYPE2_1,
    "vript_stage_one": VRIPT_STAGE_ONE,
    "shot2story_stage_one": SHOT2STORY_STAGE_ONE,
    "parallel": PARALLEL,
    "seq": SEQ,
    "modify": MODIFY, # 上述三个是多轮对话数据
}


def parse_sampling_rate(dataset_name):
    match = re.search(r"%(\d+)$", dataset_name)
    if match:
        return int(match.group(1)) / 100.0
    return 1.0


def data_list(dataset_names):
    config_list = []
    for dataset_name in dataset_names:
        sampling_rate = parse_sampling_rate(dataset_name)
        dataset_name = re.sub(r"%(\d+)$", "", dataset_name)
        if dataset_name in data_dict.keys():
            config = data_dict[dataset_name].copy()
            config["sampling_rate"] = sampling_rate
            config_list.append(config)
        else:
            raise ValueError(f"do not find {dataset_name}")
    return config_list


if __name__ == "__main__":
    dataset_names = ["demo"]
    configs = data_list(dataset_names)
    for config in configs:
        print(config)
