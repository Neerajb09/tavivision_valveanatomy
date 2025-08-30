import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights
import imageio.v2 as imageio
from PIL import Image
import numpy as np
from opticalflow_tvl1 import OpticalFlowTVL1Processor

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

# ==== Model Definition ====
class DualStreamLateFusionModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.rgb_stream = r2plus1d_18(weights=R2Plus1D_18_Weights.DEFAULT)
        self.flow_stream = r2plus1d_18(weights=R2Plus1D_18_Weights.DEFAULT)

        self.rgb_stream.fc = nn.Identity()
        self.flow_stream.fc = nn.Identity()
        # iAFF fusion module
        self.iaff = iAFF(channels=512)

        self.fusion_fc1 = nn.Linear(512, 256)
        self.relu1 = nn.ReLU()
        self.fusion_fc2 = nn.Linear(256, num_classes)

    def forward(self, rgb, flow):
        rgb_feat = self.rgb_stream(rgb)
        flow_feat = self.flow_stream(flow)
        # Apply improved attention feature fusion
        fused = self.iaff(rgb_feat, flow_feat)
        x = self.relu1(self.fusion_fc1(fused))
        x = self.fusion_fc2(x)
        return x

# ==== Hierarchical Inference Class ====
class HierarchicalDualStreamInference:
    def __init__(self, stage1_path, stage2_path, device='cuda:5', num_frames=32, img_size=112):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.num_frames = num_frames
        self.img_size = img_size

        self.model_stage1 = DualStreamLateFusionModel(num_classes=2).to(self.device)
        self.model_stage2 = DualStreamLateFusionModel(num_classes=2).to(self.device)
        self.model_stage1.load_state_dict(torch.load(stage1_path, map_location=self.device))
        self.model_stage2.load_state_dict(torch.load(stage2_path, map_location=self.device))
        self.model_stage1.eval()
        self.model_stage2.eval()

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.45, 0.45, 0.45], [0.225, 0.225, 0.225])
        ])

        self.class_mapping = {
            0: "Bicuspid - Type 0",
            1: "Bicuspid - Type 1",
            2: "Tricuspid"
        }

    def load_video(self, path):
        try:
            reader = imageio.get_reader(path, 'ffmpeg')
            raw = [Image.fromarray(frame) for frame in reader]
            reader.close()
        except Exception as e:
            print(f"[ERROR] Failed to read video {path}: {e}")
            raw = []

        total = len(raw)
        if total >= self.num_frames:
            idxs = np.linspace(0, total - 1, self.num_frames, dtype=int)
            sampled = [raw[i] for i in idxs]
        else:
            sampled = raw + [Image.new("RGB", (self.img_size, self.img_size))] * (self.num_frames - total)

        processed = [self.transform(img) for img in sampled]
        return torch.stack(processed, dim=1)  # [C, T, H, W]

    def predict(self, rgb_video_path):
        # Load RGB video
        rgb_tensor = self.load_video(rgb_video_path).unsqueeze(0).to(self.device)  # [1, C, T, H, W]

        # Generate flow video
        flow_video_path = './Valve_anatomy_classification/inference_pipeline/optical_flow.mp4'
        OpticalFlowTVL1Processor().process_video(rgb_video_path, output_video_path=flow_video_path)
        flow_tensor = self.load_video(flow_video_path).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # Stage 1: Bicuspid vs Tricuspid
            out1 = self.model_stage1(rgb_tensor, flow_tensor)
            probs1 = torch.softmax(out1, dim=1)
            pred1 = torch.argmax(probs1, dim=1).item()

            if pred1 == 1:  # Tricuspid
                final_class = 2
                final_probs = probs1.squeeze().cpu().numpy()
                print(f"[Stage 1] Predicted: Tricuspid | Probabilities: {final_probs}")
            else:
                out2 = self.model_stage2(rgb_tensor, flow_tensor)
                probs2 = torch.softmax(out2, dim=1)
                pred2 = torch.argmax(probs2, dim=1).item()
                final_class = pred2
                final_probs = probs2.squeeze().cpu().numpy()
                print(f"[Stage 1] Bicuspid confirmed. Passing to Stage 2")
                print(f"[Stage 2] Predicted: {self.class_mapping[pred2]} | Probabilities: {final_probs}")

        print(f"\n✅ Final Prediction: {self.class_mapping[final_class]}")
        return self.class_mapping[final_class], final_probs



# if __name__ == "__main__":
#     rgb_video_path = "temp_output_Segmented.mp4"
#     stage1_path = "/mnt/nvme_disk2/User_data/nb57077k/Phase2_/Valve_anatomy_classification/Model/hirechical/r2+1d/dual_stream_level1_r2plus1d.pth"
#     stage2_path = "/mnt/nvme_disk2/User_data/nb57077k/Phase2_/Valve_anatomy_classification/Model/hirechical/r2+1d/dual_stream_level2_r2plus1d.pth"
#     infer = HierarchicalDualStreamInference(stage1_path, stage2_path)
#     label, probs = infer.predict(rgb_video_path)