import math

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



if __name__ == "__main__":
    # 示例用法
    original_fps = 1
    target_fps = 0.6
    all_frame_paths = [f"frame_{i:04d}.jpg" for i in range(1, 17)]  # 模拟10秒的视频，共300帧

    sampled_frames = get_target_fps_frame_list(all_frame_paths, original_fps, target_fps)
    print("采样后的帧列表:")
    for frame in sampled_frames:
        print(frame)