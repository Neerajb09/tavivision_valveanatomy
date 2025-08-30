# test_untrimmednet_hierarchical.py
import sys
import os

# Set project root and add to sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.insert(0, ROOT_DIR)

import torch
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix

from data.dataset_untrimmedNetHierarchical import get_loader
from model.untrimmedNet_hierarchical import TwoStreamUntrimmedNet
from evalution.evaluation_hierarchical import evaluate_with_report
from config.config import device

# ==== LABEL MAPPINGS ====
label_mapping_level1 = {0: 0, 1: 0, 2: 1}   # Bicuspid (0,1) vs Tricuspid (2)
filter_classes_level2 = [0, 1]              # Only Bicuspid classes

# ==== SETTINGS ====
n_clips = 5  # Must match training

# ==== LOAD TEST SETS ====
test_loader_lvl1 = get_loader('test', label_mapping=label_mapping_level1, n_clips=n_clips)
test_loader_full = get_loader('test', n_clips=n_clips)

# ==== LOAD MODELS ====
WEIGHTS_DIR = os.path.join(ROOT_DIR, "training", "weights")

model_lvl1 = TwoStreamUntrimmedNet(num_classes=2, n_clips=n_clips).to(device)
model_lvl1.load_state_dict(torch.load(os.path.join(WEIGHTS_DIR, "untrimmednet_level1_attention.pth")))
model_lvl1.eval()

model_lvl2 = TwoStreamUntrimmedNet(num_classes=2, n_clips=n_clips).to(device)
model_lvl2.load_state_dict(torch.load(os.path.join(WEIGHTS_DIR, "untrimmednet_level2_attention.pth")))
model_lvl2.eval()

# ==== STAGE 1 EVALUATION ====
print("\n==== LEVEL 1 Evaluation ====")
evaluate_with_report(model_lvl1, test_loader_lvl1, split_name="Test Level 1", threshold=None)

# ==== FULL HIERARCHICAL INFERENCE ====
print("\n==== HIERARCHICAL INFERENCE ====")

all_preds, all_labels = [], []
correct_total, total_total = 0, 0

with torch.no_grad():
    for (rgb, flow), y_true in tqdm(test_loader_full, desc="Hierarchical Test"):
        rgb, flow, y_true = rgb.to(device), flow.to(device), y_true.to(device)

        # Stage 1: Bicuspid vs Tricuspid
        out_lvl1 = model_lvl1(rgb, flow)
        probs_lvl1 = torch.softmax(out_lvl1, dim=1)
        pred_lvl1 = torch.argmax(probs_lvl1, dim=1)

        for i in range(len(pred_lvl1)):
            if pred_lvl1[i].item() == 1:
                final_pred = 2  # Tricuspid
            else:
                rgb_i = rgb[i].unsqueeze(0)
                flow_i = flow[i].unsqueeze(0)
                out_lvl2 = model_lvl2(rgb_i, flow_i)
                probs_lvl2 = torch.softmax(out_lvl2, dim=1)
                pred_lvl2 = torch.argmax(probs_lvl2, dim=1).item()
                final_pred = pred_lvl2  # Bicuspid Type 0 or Type 1

            all_preds.append(final_pred)
            all_labels.append(y_true[i].item())

            if final_pred == y_true[i].item():
                correct_total += 1
            total_total += 1

# ==== REPORT ====
accuracy = correct_total / total_total
print(f"\nHierarchical Test Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(all_labels, all_preds))
print("\nConfusion Matrix:")
print(confusion_matrix(all_labels, all_preds))
