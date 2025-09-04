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

# ------------------------------------------------------------
# Transforms (unchanged)
# ------------------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.45, 0.45, 0.45], [0.225, 0.225, 0.225]),
])

# ------------------------------------------------------------
# DualStreamDataset with frame-trimming support
# ------------------------------------------------------------
class DualStreamDataset(Dataset):
    """
    Loads paired RGB and TV-L1 flow videos. Frame selection supports:
      - 'uniform': uniformly sample self.num_frames across the whole video (original behavior)
      - 'trim_middle': drop first `skip_first` and last `skip_last` frames, then pick a
        contiguous window of length `num_frames` centered in the remaining range
        (your “use middle 16 frames” use-case).
    """
    def __init__(
        self,
        rgb_root,
        flow_root,
        split,
        transform=transform,
        num_frames=16,                 # default to 16 for your scenario
        frame_strategy="trim_middle",  # default to trim middle (skip edges)
        skip_first=8,
        skip_last=8,
        label_mapping=None,
        filter_classes=None,
        shuffle_samples=True,
    ):
        self.rgb_root = os.path.join(rgb_root, split)
        self.flow_root = os.path.join(flow_root, split)
        self.transform = transform
        self.num_frames = num_frames
        self.frame_strategy = frame_strategy
        self.skip_first = max(0, int(skip_first))
        self.skip_last  = max(0, int(skip_last))
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

        if shuffle_samples:
            random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    # ------------------------------
    # Frame reading & selection utils
    # ------------------------------
    def _read_frames(self, path):
        """Return a list of PIL.Image frames for a video. Empty list on failure."""
        frames = []
        try:
            reader = imageio.get_reader(path, 'ffmpeg')
            for frame in reader:
                frames.append(Image.fromarray(frame))
            reader.close()
        except Exception as e:
            print(f"[ERROR] Failed to read video {path}: {e}")
        return frames

    def _select_indices_uniform(self, total):
        """Uniform indices across [0, total). Pads by repeating last if needed."""
        if total <= 0:
            return [0] * self.num_frames
        if total >= self.num_frames:
            return np.linspace(0, total - 1, self.num_frames, dtype=int).tolist()
        # total < num_frames: take all then repeat last
        idxs = list(range(total))
        while len(idxs) < self.num_frames:
            idxs.append(total - 1)
        return idxs

    def _select_indices_trim_middle(self, total):
        """
        Trim first `skip_first` and last `skip_last`, then select a contiguous
        window of exactly `num_frames` centered in the remaining range.
        If the remaining range is shorter, fall back to uniform within the range
        and pad by repeating the last index.
        """
        if total <= 0:
            return [0] * self.num_frames

        start = min(self.skip_first, total)               # safe
        end   = max(start, total - self.skip_last)        # exclusive
        available = max(0, end - start)

        if available == 0:
            # nothing left after trimming; just repeat index 0
            return [0] * self.num_frames

        if available >= self.num_frames:
            # contiguous window centered in [start, end)
            center = (start + end - 1) // 2              # center index
            half = self.num_frames // 2
            s = center - half
            # clamp to valid window inside [start, end - num_frames]
            s = max(start, min(s, end - self.num_frames))
            return list(range(s, s + self.num_frames))
        else:
            # not enough frames after trim → uniformly sample what we have, then pad
            idxs = np.linspace(start, end - 1, available, dtype=int).tolist()
            while len(idxs) < self.num_frames:
                idxs.append(idxs[-1])
            return idxs

    def _select_indices(self, total):
        if self.frame_strategy == "trim_middle":
            return self._select_indices_trim_middle(total)
        else:
            return self._select_indices_uniform(total)

    def _frames_to_tensor(self, frames, idxs):
        """Gather frames by idxs (clamped), apply transform, stack to [C, T, H, W]."""
        if len(frames) == 0:
            # All black placeholders if video couldn't be read
            placeholder = Image.new("RGB", (IMG_SIZE, IMG_SIZE))
            processed = [self.transform(placeholder) for _ in range(self.num_frames)]
            return torch.stack(processed, dim=1)

        gathered = []
        last_valid = len(frames) - 1
        for i in idxs:
            j = min(max(i, 0), last_valid)
            gathered.append(frames[j])

        processed = [self.transform(img) for img in gathered]
        return torch.stack(processed, dim=1)  # [C, T, H, W]

    # ------------------------------
    # Item
    # ------------------------------
    def __getitem__(self, idx):
        rgb_path, flow_path, label = self.samples[idx]

        # Read both videos first
        rgb_frames = self._read_frames(rgb_path)
        flow_frames = self._read_frames(flow_path)

        # Use the same index plan for both streams, based on the shorter one
        total = min(len(rgb_frames), len(flow_frames))
        idxs = self._select_indices(total)

        rgb_tensor  = self._frames_to_tensor(rgb_frames,  idxs)
        flow_tensor = self._frames_to_tensor(flow_frames, idxs)

        return (rgb_tensor, flow_tensor), label

# ------------------------------------------------------------
# DataLoader factory
# ------------------------------------------------------------
def get_loader(
    split,
    label_mapping=None,
    filter_classes=None,
    # Defaults here enforce your 16-middle-frames behavior:
    num_frames=16,
    frame_strategy="trim_middle",
    skip_first=8,
    skip_last=8,
    batch_size=BATCH_SIZE,
    shuffle=None,
    num_workers=8,
    pin_memory=True,
):
    """
    Create a DataLoader. By default, it trims 8 + 8 frames and keeps the middle 16.

    You can revert to uniform sampling by:
        get_loader("train", frame_strategy="uniform", num_frames=NUM_FRAMES)
    """
    ds = DualStreamDataset(
        RGB_DIR,
        FLOW_DIR,
        split,
        num_frames=num_frames,
        frame_strategy=frame_strategy,
        skip_first=skip_first,
        skip_last=skip_last,
        label_mapping=label_mapping,
        filter_classes=filter_classes,
        shuffle_samples=True,
    )

    if shuffle is None:
        # keep your original convention, but if your split names differ, adjust here
        shuffle = (split == 'train_augmented')

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=(split.startswith('train')),  # usually helpful for training
        persistent_workers=(num_workers > 0),
    )

    print(f"{split}: {len(ds)} samples | frames={num_frames} | strategy={frame_strategy} | skip=({skip_first},{skip_last})")
    return loader
