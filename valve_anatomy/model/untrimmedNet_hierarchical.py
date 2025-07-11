import torch
import torch.nn as nn
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights

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
    def __init__(self, num_classes):
        super().__init__()
        self.rgb_model = r2plus1d_18(weights=R2Plus1D_18_Weights.DEFAULT)
        self.flow_model = r2plus1d_18(weights=R2Plus1D_18_Weights.DEFAULT)
        self.rgb_model.fc = nn.Identity()
        self.flow_model.fc = nn.Identity()
        self.fusion_fc = nn.Linear(512 * 2, 512)

        self.attn = AttentionModule(512)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, rgb_clip, flow_clip):
        # Original shape: [B, C, T, H, W] → add clip dimension → [B, 1, C, T, H, W]
        rgb_clip = rgb_clip.unsqueeze(1)
        flow_clip = flow_clip.unsqueeze(1)

        B, N, C, T, H, W = rgb_clip.shape
        rgb_feats, flow_feats = [], []
        for i in range(N):
            rgb_out = self.rgb_model(rgb_clip[:, i])
            flow_out = self.flow_model(flow_clip[:, i])
            fused = torch.cat([rgb_out, flow_out], dim=1)
            fused = self.fusion_fc(fused)
            rgb_feats.append(fused)

        feats = torch.stack(rgb_feats, dim=1)  # [B, N, 512]
        video_feat, _ = self.attn(feats)
        out = self.fc(video_feat)
        return out
