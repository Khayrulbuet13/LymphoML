import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights

class Student1(nn.Module):
    """Student model based on ResNet18."""
    def __init__(self, num_classes=2, use_pretrained=True, in_channels=1):
        super(Student1, self).__init__()
        self.model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if use_pretrained else None)
        
        # Modify the first convolutional layer to accept 1 channel input instead of 3
        if in_channels != 3:
            self.model.conv1 = nn.Conv2d(
                in_channels, 
                self.model.conv1.out_channels, 
                kernel_size=self.model.conv1.kernel_size, 
                stride=self.model.conv1.stride, 
                padding=self.model.conv1.padding, 
                bias=False
            )
            
        # Modify the fully connected layer for the number of classes
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)
        
    def forward(self, x):
        return self.model(x)
    
    def load_pretrained(self, path):
        """Load pretrained weights."""
        self.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        return self
