import torch
import torch.nn as nn
from torchvision import models, transforms

# 1. The Model Architecture (Must match training code EXACTLY)
class BiomassModel(nn.Module):
    def __init__(self, num_meta_features=2, num_outputs=5):
        super(BiomassModel, self).__init__()

        # Backbone: ConvNeXt Tiny
        # We set weights=None because we will load your custom trained weights later
        self.backbone = models.convnext_tiny(weights=None)

        # Remove the original classification head
        cnn_out_features = self.backbone.classifier[2].in_features
        self.backbone.classifier[2] = nn.Identity()

        # Metadata Branch
        self.meta_net = nn.Sequential(
            nn.Linear(num_meta_features, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU()
        )

        # Fusion Head
        self.fusion_head = nn.Sequential(
            nn.Linear(cnn_out_features + 16, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_outputs)
        )

    def forward(self, image, meta):
        img_features = self.backbone(image)
        meta_features = self.meta_net(meta)

        # Concatenate image features and metadata features
        combined = torch.cat((img_features, meta_features), dim=1)

        return self.fusion_head(combined)

# 2. The Preprocessing Transform
# This ensures user images are resized/normalized exactly like your training data
def get_prediction_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])