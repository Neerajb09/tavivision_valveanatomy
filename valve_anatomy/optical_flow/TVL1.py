# tvl1.py

import os
import cv2
import numpy as np
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Add path for config
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.config import RGB_DIR, FLOW_DIR

def flow_to_color(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[..., 0], flow[..., 1]
    magnitude, angle = cv2.cartToPolar(fx, fy, angleInDegrees=True)
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 0] = angle / 2
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def initialize_main_tvl1():
    try:
        gpu_count = cv2.cuda.getCudaEnabledDeviceCount()
        if gpu_count > 0:
            print(f"Using CUDA TVL1 on GPU (detected {gpu_count} CUDA device(s)).")
            return "cuda"
    except AttributeError:
        pass

    try:
        _ = cv2.optflow.DualTVL1OpticalFlow_create()
        print("Using CPU TVL1 (opencv-contrib-python).")
        return "cpu"
    except AttributeError:
        pass

    try:
        _ = cv2.DualTVL1OpticalFlow_create()
        print("Using CPU TVL1 (opencv-python legacy interface).")
        return "cpu"
    except AttributeError:
        pass

    raise RuntimeError("TVL1 Optical Flow not available. Please install opencv-contrib-python.")

def create_tvl1_instance(mode):
    if mode == "cuda":
        return cv2.cuda_OpticalFlowDual_TVL1.create()
    else:
        try:
            return cv2.optflow.DualTVL1OpticalFlow_create()
        except AttributeError:
            return cv2.DualTVL1OpticalFlow_create()

def process_single_video(mode, rgb_root_dir, flow_root_dir, root, files):
    tvl1 = create_tvl1_instance(mode)
    frame_files = sorted([f for f in files if f.endswith(".jpg")])

    if len(frame_files) < 2:
        return "skipped", root

    relative_path = os.path.relpath(root, rgb_root_dir)
    split_dir = relative_path.split(os.sep)[0]
    folder_name = os.path.basename(root)

    flow_dir = os.path.join(flow_root_dir, split_dir)
    os.makedirs(flow_dir, exist_ok=True)

    out_video_path = os.path.join(flow_dir, f"{folder_name}_flow_vis.mp4")

    if os.path.exists(out_video_path):
        return "already_done", root

    frame_paths = [os.path.join(root, f) for f in frame_files]
    frames = []
    for fp in frame_paths:
        frame = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
        if frame is not None:
            frames.append(frame)

    if len(frames) < 2:
        return "skipped_after_read", root

    flow_vis_frames = []

    for i in range(len(frames) - 1):
        prev = frames[i]
        curr = frames[i + 1]

        if mode == "cuda":
            prev_gpu = cv2.cuda_GpuMat()
            curr_gpu = cv2.cuda_GpuMat()
            prev_gpu.upload(prev)
            curr_gpu.upload(curr)

            flow_gpu = tvl1.calc(prev_gpu, curr_gpu, None)
            flow = flow_gpu.download()
        else:
            flow = tvl1.calc(prev, curr, None)

        flow_vis = flow_to_color(flow)
        flow_vis_frames.append(flow_vis)

    if flow_vis_frames:
        height, width, _ = flow_vis_frames[0].shape
        out = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (width, height))
        for frame in flow_vis_frames:
            out.write(frame)
        out.release()
        return "generated", out_video_path
    else:
        return "no_flow", root

def compute_optical_flow_parallel(rgb_root_dir, flow_root_dir, num_threads=8):
    mode = initialize_main_tvl1()

    video_dirs = []
    for root, dirs, files in os.walk(rgb_root_dir):
        video_dirs.append((root, files))

    log_file = open("flow_log.out", "w")

    stats = {
        "generated": 0,
        "skipped": 0,
        "skipped_after_read": 0,
        "no_flow": 0,
        "already_done": 0
    }

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for root, files in video_dirs:
            futures.append(executor.submit(process_single_video, mode, rgb_root_dir, flow_root_dir, root, files))

        for f in tqdm(as_completed(futures), total=len(futures)):
            status, path = f.result()
            stats[status] += 1
            log_file.write(f"{status}: {path}\n")

    log_file.write(f"\nSummary:\n")
    for key, val in stats.items():
        log_file.write(f"{key}: {val}\n")
    log_file.close()

    print("\n--- Completed ---")
    for key, val in stats.items():
        print(f"{key}: {val}")

if __name__ == "__main__":
     compute_optical_flow_parallel(RGB_DIR, FLOW_DIR, num_threads=12)