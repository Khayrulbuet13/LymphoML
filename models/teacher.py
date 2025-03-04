import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights

class TeacherModel(nn.Module):
    """Teacher model based on ResNet50."""
    def __init__(self, num_classes=2, use_pretrained=True):
        super(TeacherModel, self).__init__()
        # Use the exact same model initialization as in teacher.py
        self.model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1 if use_pretrained else None)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)
        
    def forward(self, x):
        return self.model(x)
    
    def load_pretrained(self, path):
        """Load pretrained weights."""
        self.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        return self
