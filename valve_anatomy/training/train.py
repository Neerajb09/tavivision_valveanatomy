# train.py
import sys
import os
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.insert(0, ROOT_DIR)


import torch
import torch.nn as nn
from tqdm import tqdm
from config.config import LR, EPOCHS, PATIENCE, device
from evalution.evaluation_hierarchical import evaluate

def train_model(model, train_loader, val_loader, model_name, num_epochs=EPOCHS, patience_limit=PATIENCE):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.25)

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
            torch.save(model.state_dict(), model_name)
            print("  → Best model saved")
        else:
            patience += 1
            if patience >= patience_limit:
                print("  → Early stopping triggered")
                break

        scheduler.step()

    return model_name
