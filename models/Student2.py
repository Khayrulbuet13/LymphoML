import torch
import torch.nn as nn
import torch.nn.functional as F

class Student2(nn.Module):
    """Defines the student quantized CNN model architecture."""
    def __init__(self, num_classes=2, input_size=(1, 48, 48)):
        super(Student2, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )
        # Calculate the number of features after convolution layers
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_size)
            dummy_output = self.features(dummy_input)
            num_ftrs = dummy_output.numel() // dummy_output.size(0)
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    
    def load_pretrained(self, path):
        """Load pretrained weights."""
        self.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        return self
