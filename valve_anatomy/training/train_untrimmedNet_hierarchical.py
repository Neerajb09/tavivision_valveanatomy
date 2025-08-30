# training/train_untrimmedNet_hierarchical.py

import os
import sys

# Ensure parent folder is on sys.path to access config, model, data, evaluation
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from data.dataset_untrimmedNetHierarchical import get_loader
from model.untrimmedNet_hierarchical import TwoStreamUntrimmedNet
from train import train_model
from evalution.evaluation_hierarchical import evaluate_with_report
from config.config import device, EPOCHS, NUM_FRAMES, IMG_SIZE, LR, PATIENCE

# ========== DEVICE INFO ==========
if device.type == 'cuda':
    print(f"Using GPU: {torch.cuda.get_device_name(device)} (Device ID: {device.index})")
else:
    print("Using CPU")

# ========== CONFIG ==========
n_clips = 5
model_dir = "weights"
os.makedirs(model_dir, exist_ok=True)

# Show training configuration
print("\n===== Training Configuration =====")
print(f"NUM_FRAMES = {NUM_FRAMES}, IMG_SIZE = {IMG_SIZE}, EPOCHS = {EPOCHS}, LR = {LR}, PATIENCE = {PATIENCE}")
print("==================================\n")

# ========== LABEL MAPS ==========
label_mapping_lvl1 = {0: 0, 1: 0, 2: 1}   # Bicuspid (1) vs Tricuspid (0)
filter_classes_lvl2 = [0, 1]              # Only Bicuspid types

# ================================
# TRAINING LEVEL 1
# ================================
print("\n===== TRAINING UNTRIMMEDNET LEVEL 1 (Bicuspid vs Tricuspid) =====")

train_loader_lvl1 = get_loader('train_augmented', label_mapping=label_mapping_lvl1, n_clips=n_clips)
val_loader_lvl1 = get_loader('val', label_mapping=label_mapping_lvl1, n_clips=n_clips)

model_lvl1 = TwoStreamUntrimmedNet(num_classes=2, n_clips=n_clips).to(device)
model_lvl1_name = 'untrimmednet_level1_attention.pth'  # Only name, not path

train_model(
    model_lvl1,
    train_loader_lvl1,
    val_loader_lvl1,
    model_lvl1_name,
    num_epochs=EPOCHS,
    patience_limit=PATIENCE
)

model_path_lvl1 = os.path.join('weights', model_lvl1_name)
model_lvl1.load_state_dict(torch.load(model_path_lvl1))
evaluate_with_report(model_lvl1, val_loader_lvl1, split_name="Val Level 1 (default)")
evaluate_with_report(model_lvl1, val_loader_lvl1, split_name="Val Level 1 (thresholded)", threshold=None)

# ================================
# TRAINING LEVEL 2
# ================================
print("\n===== TRAINING UNTRIMMEDNET LEVEL 2 (Bicuspid Type 0 vs Type 1) =====")

train_loader_lvl2 = get_loader('train_augmented', filter_classes=filter_classes_lvl2, n_clips=n_clips)
val_loader_lvl2 = get_loader('val', filter_classes=filter_classes_lvl2, n_clips=n_clips)

model_lvl2 = TwoStreamUntrimmedNet(num_classes=2, n_clips=n_clips).to(device)
model_lvl2_name = 'untrimmednet_level2_attention.pth'  # Only name, not path

train_model(
    model_lvl2,
    train_loader_lvl2,
    val_loader_lvl2,
    model_lvl2_name,
    num_epochs=EPOCHS,
    patience_limit=PATIENCE
)

model_path_lvl2 = os.path.join('weights', model_lvl2_name)
model_lvl2.load_state_dict(torch.load(model_path_lvl2))
evaluate_with_report(model_lvl2, val_loader_lvl2, split_name="Val Level 2")

print("\n===== Hierarchical Training with Attention (UntrimmedNet) Completed =====")
