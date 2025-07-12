# data/dataset_untrimmedNetHierarchical.py

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import random

from config.config import RGB_DIR, FLOW_DIR, BATCH_SIZE, IMG_SIZE, NUM_FRAMES as CLIP_LEN

class DualStreamDataset(Dataset):
    def __init__(self, rgb_root, flow_root, split, stage='stage1', clip_len=CLIP_LEN, n_clips=5, label_mapping=None, filter_classes=None):
        self.rgb_root = rgb_root
        self.flow_root = flow_root
        self.split = split
        self.stage = stage
        self.clip_len = clip_len
        self.n_clips = n_clips
        self.label_mapping = label_mapping
        self.filter_classes = filter_classes

        self.class_mapping_stage1 = {'Bicuspid_Type_0': 1, 'Bicuspid_Type_1': 1, 'Tricuspid': 0}
        self.class_mapping_stage2 = {'Bicuspid_Type_0': 0, 'Bicuspid_Type_1': 1}

        self.samples = self._load_file_paths()

    def _load_file_paths(self):
        samples = []
        seg_path = os.path.join(self.rgb_root, self.split)
        for class_dir in os.listdir(seg_path):
            class_dir_path = os.path.join(seg_path, class_dir)
            for file in os.listdir(class_dir_path):
                filename_wo_ext = os.path.splitext(file)[0]
                seg_file_path = os.path.join(class_dir_path, file)
                flow_file_name = filename_wo_ext + "_flow_vis.mp4"
                flow_file_path = os.path.join(self.flow_root, self.split, class_dir, flow_file_name)

                if os.path.exists(flow_file_path):
                    # apply stage logic
                    if self.stage == 'stage1':
                        label = self.class_mapping_stage1[class_dir]
                    elif self.stage == 'stage2' and class_dir != 'Tricuspid':
                        label = self.class_mapping_stage2[class_dir]
                    else:
                        continue

                    # optional filtering
                    if self.filter_classes is not None and label not in self.filter_classes:
                        continue
                    if self.label_mapping is not None:
                        label = self.label_mapping[label]

                    samples.append((seg_file_path, flow_file_path, label))
        return samples

    def __len__(self):
        return len(self.samples)

    def _read_all_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"❌ Could not open video: {video_path}")

        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            frames.append(frame)
        cap.release()

        if len(frames) == 0:
            raise ValueError(f"❌ No readable frames in: {video_path}")
        return np.array(frames)

    def _sample_clips(self, video_path):
        all_frames = self._read_all_frames(video_path)
        total_frames = len(all_frames)
        clips = []

        for _ in range(self.n_clips):
            if total_frames < self.clip_len:
                clip = list(all_frames)
                while len(clip) < self.clip_len:
                    clip.append(clip[-1])
                clip = np.stack(clip, axis=0)
            else:
                start = random.randint(0, total_frames - self.clip_len)
                clip = all_frames[start:start + self.clip_len]

            clip = clip.transpose(3, 0, 1, 2) / 255.0  # (T, H, W, C) -> (C, T, H, W)
            clips.append(torch.tensor(clip, dtype=torch.float32))

        return torch.stack(clips)  # [N, C, T, H, W]

    def __getitem__(self, idx):
        try:
            seg_path, flow_path, label = self.samples[idx]
            rgb_clips = self._sample_clips(seg_path)
            flow_clips = self._sample_clips(flow_path)
            return (rgb_clips, flow_clips), label
        except Exception as e:
            print(f"⚠️ Skipping corrupted sample [{idx}] due to error: {e}")
            return self.__getitem__((idx + 1) % len(self.samples))

def get_loader(split, stage='stage1', label_mapping=None, filter_classes=None, n_clips=5):
    dataset = DualStreamDataset(
        RGB_DIR,
        FLOW_DIR,
        split,
        stage=stage,
        label_mapping=label_mapping,
        filter_classes=filter_classes,
        n_clips=n_clips
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=(split == 'train_augmented'), num_workers=8, pin_memory=True)
    print(f"{split}: {len(dataset)} samples")
    return loader
