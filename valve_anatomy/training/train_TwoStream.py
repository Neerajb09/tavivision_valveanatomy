import sys
import os
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.insert(0, ROOT_DIR)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

from data.dataset import DualStreamDataset
from config.config import RGB_DIR, FLOW_DIR, BATCH_SIZE, EPOCHS, NUM_FRAMES, IMG_SIZE, LR, PATIENCE, device
from evalution.evaluation_hierarchical import evaluate

# Model imports
from model.TwoStreamMC3_32Aff import TwoStreamCNN as MC3_AFF
from model.TwoStreamMC3_32Concat import TwoStreamCNN as MC3_CONCAT
from model.TwoStream_R3D_32Aff import TwoStreamR3DAttn as R3D_AFF
from model.TwoStream_r2plus1d_32Aff import TwoStreamR2Plus1DAttn as R2PLUS1D_AFF
from model.TwoStream_r2plus1d_32Concat import TwoStreamR2Plus1D as R2PLUS1D_CONCAT

MODEL_MAP = {
    "mc3_aff": MC3_AFF,
    "mc3_concat": MC3_CONCAT,
    "r3d_aff": R3D_AFF,
    "r2plus1d_aff": R2PLUS1D_AFF,
    "r2plus1d_concat": R2PLUS1D_CONCAT
}

def train_model(model, train_loader, val_loader, model_name, num_epochs=EPOCHS, patience_limit=PATIENCE):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.25)
    model_path = os.path.join('weights', f"best_{model_name}.pth")
    best_val_acc = 0
    patience = 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_corrects, total_samples = 0, 0
        pbar = tqdm(train_loader, desc=f"[{model_name}] Epoch {epoch}", leave=False)

        for (rgb, flow), y in pbar:
            rgb, flow, y = rgb.to(device), flow.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(rgb, flow)
            if isinstance(out, tuple):
                out = out[0]
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(out, 1)
            running_corrects += torch.sum(preds == y).item()
            total_samples += y.size(0)
            pbar.set_postfix(loss=loss.item())

        train_acc = running_corrects / total_samples
        val_acc, _, _ = evaluate(model, val_loader, f"Val Epoch {epoch}")

        print(f"[{model_name}] Epoch {epoch}: Train Acc={train_acc:.4f} | Val Acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience = 0
            torch.save(model.state_dict(), model_path)
            print("  → Best model saved")
        else:
            patience += 1
            if patience >= patience_limit:
                print("  → Early stopping triggered")
                break

        scheduler.step()

    return model_name

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True,
                        help="Choose from: mc3_aff, mc3_concat, r3d_aff, r2plus1d_aff, r2plus1d_concat")
    parser.add_argument('--num_classes', type=int, default=3)
    args = parser.parse_args()

    assert args.model in MODEL_MAP, f"Unsupported model key: {args.model}"
    model = MODEL_MAP[args.model](num_classes=args.num_classes).to(device)

    train_dataset = DualStreamDataset(RGB_DIR, FLOW_DIR, split='train_augmented', num_frames=NUM_FRAMES)
    val_dataset = DualStreamDataset(RGB_DIR, FLOW_DIR, split='val', num_frames=NUM_FRAMES)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    os.makedirs("weights", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    train_model(model, train_loader, val_loader, args.model)

if __name__ == "__main__":
    main()
