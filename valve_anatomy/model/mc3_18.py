import torch
import torch.nn as nn
from torchvision.models.video import mc3_18

class MC3FineTuneModel(nn.Module):
    def __init__(self, num_classes):
        super(MC3FineTuneModel, self).__init__()
        
        # Load pretrained MC3_18 model
        self.base_model = mc3_18(pretrained=True)

        # Freeze all layers
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Unfreeze only layer4
        for param in self.base_model.layer4.parameters():
            param.requires_grad = True

        # Replace the final fully connected layer
        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.base_model(x)
