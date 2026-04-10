import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class Model(nn.Module):
    def __init__(self, num_classes=1):
        super(Model, self).__init__()
        # Load a pre-trained model (e.g., ResNet, EfficientNet) from timm
        self.backbone = timm.create_model('resnet18', pretrained=True)
        
        # Replace the final fully connected layer to match our number of classes
        in_features = self.backbone.get_classifier().in_features
        self.backbone.reset_classifier(0)  # Remove the original classifier
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        features = self.backbone(x)  # Extract features using the backbone
        output = self.classifier(features)  # Classify using the new classifier
        return output