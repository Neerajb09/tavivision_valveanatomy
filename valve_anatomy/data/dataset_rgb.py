# ==== dataset.py ====
import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import imageio.v2 as imageio
import numpy as np

class VideoDataset(Dataset):
    def __init__(self, root_dir, transform=None, num_frames=45):
        self.transform = transform
        self.num_frames = num_frames
        self.samples = []
        self.class_to_idx = {}

        # Map class names to indices
        classes = [d for d in sorted(os.listdir(root_dir)) if os.path.isdir(os.path.join(root_dir, d))]
        self.class_to_idx = {cls: i for i, cls in enumerate(classes)}

        # Collect video paths and labels
        for cls, idx in self.class_to_idx.items():
            pattern = os.path.join(root_dir, cls, '**', '*.mp4')
            for path in glob.glob(pattern, recursive=True):
                self.samples.append((path, idx))

        if not self.samples:
            raise RuntimeError(f"No .mp4 videos found in {root_dir}.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        frames = self.load_video(path)
        return frames, label

    def load_video(self, path):
        try:
            reader = imageio.get_reader(path, 'ffmpeg')
            raw = [Image.fromarray(frame) for frame in reader]
            reader.close()
        except Exception as e:
            print(f"[ERROR] Failed to read video {path}: {e}")
            raw = []

        total = len(raw)
        if total >= self.num_frames:
            idxs = np.linspace(0, total - 1, self.num_frames, dtype=int)
            sampled = [raw[i] for i in idxs]
        else:
            sampled = raw + [Image.new("RGB", (112, 112))] * (self.num_frames - total)

        processed = [self.transform(img) for img in sampled]
        return torch.stack(processed, dim=1)  # [C, T, H, W]

# ==== DataLoader helper ====
def get_loader(root_dir, split, transform, num_frames, batch_size):
    split_dir = os.path.join(root_dir, split)
    ds = VideoDataset(split_dir, transform=transform, num_frames=num_frames)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=(split == 'train'),
                        num_workers=16, pin_memory=True)
    print(f"{split}: {len(ds)} samples, classes = {ds.class_to_idx}")
    return loader, ds
