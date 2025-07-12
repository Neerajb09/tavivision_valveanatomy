# model/untrimmedNet_hierarchical.py

import torch
import torch.nn as nn
import torchvision

class AttentionModule(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.attn_fc = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.ReLU(),
            nn.Linear(in_dim // 2, 1)
        )

    def forward(self, clip_feats):
        scores = self.attn_fc(clip_feats).squeeze(-1)  # [B, N]
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)  # [B, N, 1]
        weighted_sum = (clip_feats * weights).sum(dim=1)  # [B, D]
        return weighted_sum, weights

class TwoStreamUntrimmedNet(nn.Module):
    def __init__(self, num_classes, n_clips):
        super().__init__()
        self.rgb_model = torchvision.models.video.r2plus1d_18(pretrained=True)
        self.flow_model = torchvision.models.video.r2plus1d_18(pretrained=True)

        self.rgb_model.fc = nn.Identity()
        self.flow_model.fc = nn.Identity()

        self.fusion_fc = nn.Linear(512 * 2, 512)
        self.attn = AttentionModule(512)
        self.classifier = nn.Linear(512, num_classes)

        self.n_clips = n_clips

    def forward(self, rgb_video, flow_video):
        B, N, C, T, H, W = rgb_video.shape  # [B, N, C, T, H, W]

        fused_feats = []
        for i in range(N):
            rgb_feat = self.rgb_model(rgb_video[:, i])   # [B, 512]
            flow_feat = self.flow_model(flow_video[:, i])  # [B, 512]
            fused = torch.cat([rgb_feat, flow_feat], dim=1)  # [B, 1024]
            fused = self.fusion_fc(fused)  # [B, 512]
            fused_feats.append(fused)

        feats = torch.stack(fused_feats, dim=1)  # [B, N, 512]
        video_feat, _ = self.attn(feats)  # [B, 512]
        return self.classifier(video_feat)  # [B, num_classes]
