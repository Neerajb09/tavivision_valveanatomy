# evaluate.py

import torch
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
from config import device

def evaluate(model, loader, split_name="Val", threshold=None):
    """
    General evaluation function with optional threshold control.
    """
    model.eval()
    correct, total = 0, 0
    preds, labels = [], []
    
    with torch.no_grad():
        for (rgb, flow), y in tqdm(loader, desc=f"Eval {split_name}", leave=False):
            rgb, flow, y = rgb.to(device), flow.to(device), y.to(device)
            out = model(rgb, flow)
            probs = torch.softmax(out, dim=1)
            print(probs)
            if threshold is None:
                # Standard argmax
                _, p = probs.max(1)
            else:
                # Apply threshold for binary classification
                if probs.shape[1] != 2:
                    raise ValueError("Threshold evaluation only supported for binary models here!")
                p = (probs[:, 0] >= threshold).long()

            correct += (p == y).sum().item()
            total += y.size(0)
            preds += p.cpu().tolist()
            labels += y.cpu().tolist()
    
    acc = correct / total
    return acc, preds, labels

def evaluate_with_report(model, loader, split_name="Val", threshold=None):
    """
    Same as above but prints report and confusion matrix directly.
    """
    acc, preds, labels = evaluate(model, loader, split_name, threshold)
    print(f"\n[{split_name}] Accuracy: {acc:.4f}")
    print(classification_report(labels, preds))
    print(confusion_matrix(labels, preds))
    return acc
