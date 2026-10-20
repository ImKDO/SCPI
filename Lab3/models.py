from torch import nn
from torchvision import models


def resnet18(num_classes):
  model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
  model.fc = nn.Linear(model.fc.in_features, num_classes)
  return model
