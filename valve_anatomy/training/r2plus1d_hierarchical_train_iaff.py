# train_hierarchical.py
import sys
import os
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.insert(0, ROOT_DIR)

import torch
from data.dataset import get_loader
from model.r2plus1d_hierarchical_iaff import DualStreamLateFusionModel  # This model includes iAFF internally
from train import train_model
from evalution.evaluation_hierarchical import evaluate_with_report
# from threshold_utils import find_best_threshold_roc
from config.config import device

# ==== LABEL MAPPINGS ====
# Mapping for level 1: merge Type 0 & 1 into bicuspid
label_mapping_level1 = {0: 0, 1: 0, 2: 1}  
# For level 2: only bicuspid samples (original labels)
filter_classes_level2 = [0, 1]

# =========================
# LEVEL 1 TRAINING
# =========================

print("\n===== TRAINING LEVEL 1 (Bicuspid vs Tricuspid) =====")

# Load training & validation data for Level 1
train_loader_lvl1 = get_loader('train_augmented', label_mapping=label_mapping_level1)
val_loader_lvl1 = get_loader('val', label_mapping=label_mapping_level1)

# Build model with iAFF fusion
model_lvl1 = DualStreamLateFusionModel(num_classes=2).to(device)
model_lvl1_name = 'dual_stream_level1_iaff_random.pth'

# Train Level 1 model
train_model(model_lvl1, train_loader_lvl1, val_loader_lvl1, model_lvl1_name)

# Reload best model after training
model_lvl1.load_state_dict(torch.load(model_lvl1_name))

# Evaluate model on validation set with default threshold
evaluate_with_report(model_lvl1, val_loader_lvl1, split_name="Val Level 1 (default)")

# Find best threshold using ROC
# best_thresh_lvl1 = find_best_threshold_roc(model_lvl1, val_loader_lvl1)

# Evaluate with best threshold for reference
evaluate_with_report(model_lvl1, val_loader_lvl1, split_name="Val Level 1 (thresholded)", threshold=None)

# =========================
# LEVEL 2 TRAINING
# =========================

print("\n===== TRAINING LEVEL 2 (Bicuspid Type 0 vs Type 1) =====")

# Load training & validation data for Level 2
train_loader_lvl2 = get_loader('train_augmented', filter_classes=filter_classes_level2)
val_loader_lvl2 = get_loader('val', filter_classes=filter_classes_level2)

# Build model with iAFF fusion
model_lvl2 = DualStreamLateFusionModel(num_classes=2).to(device)
model_lvl2_name = 'dual_stream_level2_iaff_random.pth'

# Train Level 2 model
train_model(model_lvl2, train_loader_lvl2, val_loader_lvl2, model_lvl2_name)

# Reload best model after training
model_lvl2.load_state_dict(torch.load(model_lvl2_name))

# Evaluate model on validation set
evaluate_with_report(model_lvl2, val_loader_lvl2, split_name="Val Level 2")

# =========================
print("\n===== Hierarchical Training Completed =====")
# print(f"Optimal Threshold for Level 1 = {best_thresh_lvl1:.3f}")
