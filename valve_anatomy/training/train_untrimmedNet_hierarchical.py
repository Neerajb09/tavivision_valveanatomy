import sys
import os
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.insert(0, ROOT_DIR)

import torch
from data.dataset import get_loader
from model.untrimmedNet_hierarchical import TwoStreamUntrimmedNet  # Make sure file is named untrimmednet.py
from train import train_model
from evalution.evaluation_hierarchical import evaluate_with_report
from config.config import device

# ==== LABEL MAPPINGS ====
label_mapping_level1 = {0: 0, 1: 0, 2: 1}  # Bicuspid vs Tricuspid
filter_classes_level2 = [0, 1]             # Only Bicuspid for Type 0 vs Type 1

# =========================
# LEVEL 1 TRAINING
# =========================
print("\n===== TRAINING UNTRIMMEDNET LEVEL 1 (Bicuspid vs Tricuspid) =====")

train_loader_lvl1 = get_loader('train_augmented', label_mapping=label_mapping_level1)
val_loader_lvl1 = get_loader('val', label_mapping=label_mapping_level1)

model_lvl1 = TwoStreamUntrimmedNet(num_classes=2).to(device)
model_lvl1_name = 'untrimmednet_level1_attention.pth'

train_model(model_lvl1, train_loader_lvl1, val_loader_lvl1, model_lvl1_name)

# Reload best model
model_dir = 'weights'
model_path1 = os.path.join(model_dir, model_lvl1_name)
model_lvl1.load_state_dict(torch.load(model_path1))

evaluate_with_report(model_lvl1, val_loader_lvl1, split_name="Val Level 1 (default)")
evaluate_with_report(model_lvl1, val_loader_lvl1, split_name="Val Level 1 (thresholded)", threshold=None)

# =========================
# LEVEL 2 TRAINING
# =========================
print("\n===== TRAINING UNTRIMMEDNET LEVEL 2 (Bicuspid Type 0 vs Type 1) =====")

train_loader_lvl2 = get_loader('train_augmented', filter_classes=filter_classes_level2)
val_loader_lvl2 = get_loader('val', filter_classes=filter_classes_level2)

model_lvl2 = TwoStreamUntrimmedNet(num_classes=2).to(device)
model_lvl2_name = 'untrimmednet_level2_attention.pth'

train_model(model_lvl2, train_loader_lvl2, val_loader_lvl2, model_lvl2_name)

# Reload best model
model_path2 = os.path.join(model_dir, model_lvl2_name)
model_lvl2.load_state_dict(torch.load(model_path2))

evaluate_with_report(model_lvl2, val_loader_lvl2, split_name="Val Level 2")

print("\n===== Hierarchical Training with Attention (UntrimmedNet) Completed =====")
