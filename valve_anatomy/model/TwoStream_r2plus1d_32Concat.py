# model/two_stream_r2plus1d.py

import torch
import torch.nn as nn
import torchvision.models.video as models

class TwoStreamR2Plus1D(nn.Module):
    def __init__(self, num_classes):
        super(TwoStreamR2Plus1D, self).__init__()
        
        self.rgb_model = models.r2plus1d_18(weights=models.R2Plus1D_18_Weights.DEFAULT)
        self.flow_model = models.r2plus1d_18(weights=models.R2Plus1D_18_Weights.DEFAULT)

        self.rgb_model.fc = nn.Identity()
        self.flow_model.fc = nn.Identity()

        self.classifier = nn.Sequential(
            nn.Linear(512 * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, rgb_input, flow_input):
        rgb_feat = self.rgb_model(rgb_input)
        flow_feat = self.flow_model(flow_input)
        fused = torch.cat([rgb_feat, flow_feat], dim=1)
        return self.classifier(fused)
