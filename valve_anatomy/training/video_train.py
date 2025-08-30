# ==== train_mc3_finetune.py ====
import os
import sys
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.insert(0, ROOT_DIR)
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
from model.mc3_18 import MC3FineTuneModel
from data.dataset_rgb import VideoDataset, get_loader
from tqdm import tqdm

# ==== Configuration ====
DATA_DIR = '/mhgp003-v1/kanpur/data_cardiovision/valve_anatomy_classification/video_split'
BATCH_SIZE = 32
EPOCHS = 20
NUM_FRAMES = 45
IMG_SIZE = 112
LR = 1e-4
PATIENCE = 3
MODEL_NAME = 'mc3_18_finetuned_16_frames.pth'
NUM_CLASSES = 3
LOG_CSV = 'training_log.csv'

device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")

# ==== Transforms ====
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.45, 0.45, 0.45], [0.225, 0.225, 0.225])
])

# Load datasets
train_loader, _ = get_loader(DATA_DIR, 'train', transform, NUM_FRAMES, BATCH_SIZE)
val_loader, _ = get_loader(DATA_DIR, 'val', transform, NUM_FRAMES, BATCH_SIZE)

# Model setup
model = MC3FineTuneModel(num_classes=NUM_CLASSES).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.25)

# ==== Evaluation Function ====
def evaluate(model, loader, split_name):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in tqdm(loader, desc=f"Eval {split_name}", leave=False):
            X, y = X.to(device), y.to(device)
            out = model(X)
            _, p = out.max(1)
            correct += (p == y).sum().item()
            total += y.size(0)
    return correct / total

# ==== Training Loop ====
best_val_acc = 0
patience = 0
log_data = []

for epoch in range(1, EPOCHS + 1):
    model.train()
    running, seen = 0, 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
    for X, y in pbar:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        _, p = out.max(1)
        running += (p == y).sum().item()
        seen += y.size(0)
        pbar.set_postfix(loss=loss.item())

    train_acc = running / seen
    val_acc = evaluate(model, val_loader, "Val")
    log_data.append([epoch, train_acc, val_acc])

    print(f"Epoch {epoch}: Train {train_acc:.4f} | Val {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience = 0
        torch.save(model.state_dict(), MODEL_NAME)
        print("  → best model saved")
    else:
        patience += 1
        if patience >= PATIENCE:
            print("  → early stopping")
            break
    scheduler.step()

# Save training log
os.makedirs('log', exist_ok=True)
LOG_CSV = os.path.join('log', 'training_log.csv')
pd.DataFrame(log_data, columns=['Epoch', 'Train Accuracy', 'Val Accuracy']).to_csv(LOG_CSV, index=False)