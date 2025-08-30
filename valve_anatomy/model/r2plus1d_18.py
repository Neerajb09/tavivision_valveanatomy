import torch
import torch.nn as nn
from torchvision.models.video import r2plus1d_18

class R2Plus1D18FineTuneModel(nn.Module):
    def __init__(self, num_classes):
        super(R2Plus1D18FineTuneModel, self).__init__()
        
        # Load pretrained R(2+1)D-18 model
        self.base_model = r2plus1d_18(pretrained=True)

        # Freeze all layers
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Unfreeze only layer4
        for param in self.base_model.layer4.parameters():
            param.requires_grad = True

        # Replace final FC layer for new classification task
        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.base_model(x)
