# training/train_2stream_r2plus1d.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.config import RGB_DIR, FLOW_DIR, BATCH_SIZE, NUM_FRAMES, EPOCHS, LR, PATIENCE, device
from data.dataset import DualStreamDataset
from model.TwoStream_r2plus1d_32Concat import TwoStreamR2Plus1D

train_dataset = DualStreamDataset(RGB_DIR, FLOW_DIR, split='train_augmented')
val_dataset = DualStreamDataset(RGB_DIR, FLOW_DIR, split='val')

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

model = TwoStreamR2Plus1D(num_classes=len(train_dataset.class_to_idx)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.25)

best_val_acc = 0.0
patience_counter = 0

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    print(f"\n[INFO] Starting Epoch {epoch}")
    for (rgb, flow), labels in tqdm(train_loader, desc=f"Epoch {epoch} [Train]"):
        rgb, flow, labels = rgb.to(device), flow.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(rgb, flow)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        total_loss += loss.item()

    train_acc = correct / total
    print(f"[Epoch {epoch}] Train Loss: {total_loss:.4f} | Train Accuracy: {train_acc:.4f}")

    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for (rgb, flow), labels in tqdm(val_loader, desc=f"Epoch {epoch} [Val]"):
            rgb, flow, labels = rgb.to(device), flow.to(device), labels.to(device)

            outputs = model(rgb, flow)
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_acc = val_correct / val_total
    print(f"[Epoch {epoch}] Validation Accuracy: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_two_stream_r2plus1d_concat.pth")
        print(f"[INFO] Best model saved at Epoch {epoch}")
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print("[INFO] Early stopping triggered.")
            break

    scheduler.step()
