# ==== test_mc3_finetune.py ====
import sys
import os
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.insert(0, ROOT_DIR)
import torch
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
from model.mc3_18 import MC3FineTuneModel
from data.dataset_rgb import get_loader
from torchvision import transforms
import numpy as np

# ==== Configuration ====
DATA_DIR = '/mhgp003-v1/kanpur/data_cardiovision/valve_anatomy_classification/video_split'
NUM_FRAMES = 45
IMG_SIZE = 112
BATCH_SIZE = 32
NUM_CLASSES = 3
MODEL_NAME = '/mnt/nvme_disk2/User_data/hp927k/cardioVision/Phase2_/Valve_anatomy_classification/Models/mc3_18_finetuned.pth'

device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")

# ==== Transforms ====
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.45, 0.45, 0.45], [0.225, 0.225, 0.225])
])

# Load test dataset
test_loader, _ = get_loader(DATA_DIR, 'test', transform, NUM_FRAMES, BATCH_SIZE)

# Load trained model
model = MC3FineTuneModel(num_classes=NUM_CLASSES).to(device)
model.load_state_dict(torch.load(MODEL_NAME))
model.eval()

# Evaluation
all_preds, all_labels = [], []
with torch.no_grad():
    for X, y in tqdm(test_loader, desc="Test Eval", leave=False):
        X, y = X.to(device), y.to(device)
        out = model(X)
        _, p = out.max(1)
        all_preds.extend(p.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

# Metrics
print(f"Test Accuracy: {(np.array(all_preds) == np.array(all_labels)).mean():.4f}")
print(classification_report(all_labels, all_preds))
print(confusion_matrix(all_labels, all_preds))
