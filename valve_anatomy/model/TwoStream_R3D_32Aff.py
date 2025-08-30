# model/two_stream_r3d_attn.py

import torch
import torch.nn as nn
import torchvision.models.video as models
import torch.nn.functional as F

class TwoStreamR3DAttn(nn.Module):
    def __init__(self, num_classes):
        super(TwoStreamR3DAttn, self).__init__()

        self.rgb_model = models.r3d_18(weights=models.R3D_18_Weights.DEFAULT)
        self.flow_model = models.r3d_18(weights=models.R3D_18_Weights.DEFAULT)

        self.rgb_model.fc = nn.Identity()
        self.flow_model.fc = nn.Identity()

        self.attn_fc = nn.Sequential(
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, rgb_input, flow_input):
        rgb_feat = self.rgb_model(rgb_input)
        flow_feat = self.flow_model(flow_input)

        concat_feat = torch.cat([rgb_feat, flow_feat], dim=1)
        attn_logits = self.attn_fc(concat_feat)
        attn_weights = F.softmax(attn_logits, dim=1)

        attn_rgb = attn_weights[:, 0].unsqueeze(1)
        attn_flow = attn_weights[:, 1].unsqueeze(1)

        fused_feat = attn_rgb * rgb_feat + attn_flow * flow_feat
        return self.classifier(fused_feat)
