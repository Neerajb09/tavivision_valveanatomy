# config.py

import torch

# Data directories
RGB_DIR = '/weka/kanpur/data_cardiovision/valve_anatomy_classification/Segmented_video_new_dataset'
FLOW_DIR = '/weka/kanpur/data_cardiovision/valve_anatomy_classification/TVL1_OpticalflowFullFinal_new_dataset'

# Hyperparameters
BATCH_SIZE = 8
EPOCHS = 30
NUM_FRAMES = 32
IMG_SIZE = 112
LR = 1e-4
PATIENCE = 10

# Device
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
