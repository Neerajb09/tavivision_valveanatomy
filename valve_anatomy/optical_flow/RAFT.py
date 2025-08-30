import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from raft import RAFT
from utils.utils import InputPadder
from argparse import Namespace
from torchvision import transforms
from glob import glob

def flow_to_image(flow):
    """Convert flow to RGB image (OpenCV-like visualization)."""
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros((*flow.shape[:2], 3), dtype=np.uint8)
    hsv[..., 0] = angle * 180 / np.pi / 2  # Hue
    hsv[..., 1] = 255  # Saturation
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)  # Value
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def load_raft_model(device='cuda'):
    args = Namespace(model='raft-sintel.pth', small=False, mixed_precision=False, alternate_corr=False)
    model = RAFT(args)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model = model.eval().to(device)
    return model

def read_video_frames(video_path):
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames

def compute_and_save_flow_video(model, frames, output_path, device='cuda', fps=20):
    transform = transforms.ToTensor()
    flow_imgs = []

    for i in range(len(frames) - 1):
        img1 = transform(frames[i]).unsqueeze(0).to(device)
        img2 = transform(frames[i+1]).unsqueeze(0).to(device)
        padder = InputPadder(img1.shape)
        image1, image2 = padder.pad(img1, img2)

        with torch.no_grad():
            _, flow_up = model(image1, image2, iters=20, test_mode=True)

        flow_np = flow_up[0].permute(1, 2, 0).cpu().numpy()
        flow_img = flow_to_image(flow_np)
        flow_imgs.append(flow_img)

    if not flow_imgs:
        return

    h, w, _ = flow_imgs[0].shape
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    for frame in flow_imgs:
        out.write(frame)
    out.release()

def process_videos_in_folders(input_root, output_root, device='cuda', fps=20):
    model = load_raft_model(device=device)

    for class_dir in tqdm(os.listdir(input_root), desc="Processing Classes"):
        class_path = os.path.join(input_root, class_dir)
        if not os.path.isdir(class_path):
            continue

        output_class_path = os.path.join(output_root, class_dir)
        os.makedirs(output_class_path, exist_ok=True)

        for video_file in tqdm(glob(f"{class_path}/*.mp4"), desc=f"{class_dir} Videos"):
            video_name = os.path.splitext(os.path.basename(video_file))[0]
            output_path = os.path.join(output_class_path, f"{video_name}_flow.mp4")

            if os.path.exists(output_path):
                continue  # Skip already processed

            frames = read_video_frames(video_file)
            if len(frames) < 2:
                continue

            compute_and_save_flow_video(model, frames, output_path, device=device, fps=fps)

# ===== Example usage =====
if __name__ == "__main__":
    input_root = "/path/to/input_videos"
    output_root = "/path/to/output_flow_videos"
    process_videos_in_folders(input_root, output_root, device='cuda', fps=20)
