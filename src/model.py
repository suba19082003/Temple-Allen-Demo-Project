"""
model.py

Defines a transfer-learning model for surface defect classification.

Why pretrained ResNet18?
- Transfer learning: The early layers of a network trained on ImageNet
  already know how to detect edges, textures, and simple patterns —
  exactly the low-level features our defect images also contain.
  We only need to retrain the final classification layer for our
  6 defect classes (crazing, inclusion, patches, pitted_surface,
  rolled-in_scale, scratches).
- ResNet18 specifically: It's small (~11M parameters), fast to train
  on CPU or a single GPU, and well-suited to a small dataset like ours
  (1,416 training images). Deeper models would overfit.
"""
import torch
import torch.nn as nn
from torchvision import models


def build_model(num_classes: int) -> nn.Module:
    """
    Load a pretrained ResNet18 and replace its final fully-connected layer.

    Args:
        num_classes: Number of output classes (e.g., 6 for our defect types).

    Returns:
        Modified ResNet18 with a new classification head.
    """
    # Load pretrained weights from ImageNet (IMAGENET1K_V1)
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    # Replace the final FC layer: map from ResNet's 512-d feature vector
    # to our num_classes. This is the only layer trained from scratch.
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


if __name__ == "__main__":
    model = build_model(num_classes=6)
    print(model.fc)