# model.py
import torch
import torch.nn as nn
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights

class iAFF(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = channels // reduction

        # Global Attention MLP
        self.global_fc1 = nn.Linear(channels, hidden)
        self.global_relu = nn.ReLU()
        self.global_fc2 = nn.Linear(hidden, channels)

        # Local Attention MLP
        self.local_fc1 = nn.Linear(channels, hidden)
        self.local_relu = nn.ReLU()
        self.local_fc2 = nn.Linear(hidden, channels)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        fused = x + y  # Element-wise sum

        # Global Attention
        g_att = self.global_fc2(self.global_relu(self.global_fc1(fused)))

        # Local Attention
        l_att = self.local_fc2(self.local_relu(self.local_fc1(fused)))

        # Combined attention
        attention = self.sigmoid(g_att + l_att)

        # Weighted fusion
        return attention * x + (1 - attention) * y

class DualStreamLateFusionModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        # Load pretrained R(2+1)D models
        self.rgb_stream = r2plus1d_18(weights=R2Plus1D_18_Weights.DEFAULT)
        self.flow_stream = r2plus1d_18(weights=R2Plus1D_18_Weights.DEFAULT)

        # Remove final classification layers
        self.rgb_stream.fc = nn.Identity()
        self.flow_stream.fc = nn.Identity()

        # iAFF fusion module
        self.iaff = iAFF(channels=512)

        # Classification head
        self.fusion_fc1 = nn.Linear(512, 256)
        self.relu1 = nn.ReLU()
        self.fusion_fc2 = nn.Linear(256, num_classes)

    def forward(self, rgb, flow):
        # Extract features from both streams
        rgb_feat = self.rgb_stream(rgb)    # Shape: (B, 512)
        flow_feat = self.flow_stream(flow)  # Shape: (B, 512)

        # Apply improved attention feature fusion
        fused = self.iaff(rgb_feat, flow_feat)

        # Classifier head
        x = self.relu1(self.fusion_fc1(fused))
        x = self.fusion_fc2(x)

        return x  # logits
