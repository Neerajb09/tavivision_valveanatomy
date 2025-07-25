# config.py

import torch

# Data directories
RGB_DIR = '/mhgp003-v1/kanpur/data_cardiovision/valve_anatomy_classification/Segmented_videos'
FLOW_DIR = '/mhgp003-v1/kanpur/data_cardiovision/valve_anatomy_classification/TVL1_OpticalflowFull'

# Hyperparameters
BATCH_SIZE = 8
EPOCHS = 30
NUM_FRAMES = 32
IMG_SIZE = 112
LR = 1e-4
PATIENCE = 7

# Device
device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
