# dataset.py
import sys
import os
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.insert(0, ROOT_DIR)


import os, glob
import numpy as np
import imageio.v2 as imageio
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from config.config import RGB_DIR, FLOW_DIR, BATCH_SIZE, NUM_FRAMES, IMG_SIZE
import random

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.45, 0.45, 0.45], [0.225, 0.225, 0.225])
])

class DualStreamDataset(Dataset):
    def __init__(self, rgb_root, flow_root, split, transform=transform, num_frames=NUM_FRAMES, label_mapping=None, filter_classes=None):
        self.rgb_root = os.path.join(rgb_root, split)
        self.flow_root = os.path.join(flow_root, split)
        self.transform = transform
        self.num_frames = num_frames
        self.samples = []
        self.class_to_idx = {}

        classes = sorted(os.listdir(self.rgb_root))
        self.class_to_idx = {cls: i for i, cls in enumerate(classes)}

        for cls in classes:
            rgb_paths = glob.glob(os.path.join(self.rgb_root, cls, '**', '*.mp4'), recursive=True)
            for rgb_path in rgb_paths:
                rel_path = os.path.relpath(rgb_path, self.rgb_root)
                flow_rel_path = rel_path.replace(".mp4", "_flow_vis.mp4")
                flow_path = os.path.join(self.flow_root, flow_rel_path)

                if os.path.exists(flow_path):
                    original_label = self.class_to_idx[cls]

                    if filter_classes is not None and original_label not in filter_classes:
                        continue

                    label = label_mapping[original_label] if label_mapping else original_label
                    self.samples.append((rgb_path, flow_path, label))
        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def load_video(self, path):
        try:
            reader = imageio.get_reader(path, 'ffmpeg')
            raw = [Image.fromarray(frame) for frame in reader]
            reader.close()
        except Exception as e:
            print(f"[ERROR] Failed to read video {path}: {e}")
            raw = []

        total = len(raw)
        
        # Handle edge case: zero-length video
        if total == 0:
            sampled = [Image.new("RGB", (IMG_SIZE, IMG_SIZE))] * self.num_frames
        elif total >= self.num_frames:
            idxs = np.linspace(0, total - 1, self.num_frames, dtype=int)
            sampled = [raw[i] for i in idxs]
        else:
            sampled = raw + [Image.new("RGB", (IMG_SIZE, IMG_SIZE))] * (self.num_frames - total)

        processed = [self.transform(img) for img in sampled]
        tensor = torch.stack(processed, dim=1)
        return tensor


    def __getitem__(self, idx):
        rgb_path, flow_path, label = self.samples[idx]
        rgb = self.load_video(rgb_path)
        flow = self.load_video(flow_path)
        return (rgb, flow), label

def get_loader(split, label_mapping=None, filter_classes=None):
    ds = DualStreamDataset(RGB_DIR, FLOW_DIR, split, label_mapping=label_mapping, filter_classes=filter_classes)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=(split=='train_augmented'), num_workers=8, pin_memory=True)
    print(f"{split}: {len(ds)} samples")
    return loader
