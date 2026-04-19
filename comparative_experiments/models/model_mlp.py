"""
MLP (Multi-Layer Perceptron) for SMPL-X Parameter Prediction
Simple baseline model using fully connected layers
"""
import torch
import torch.nn as nn


class MLP(nn.Module):
    """MLP model for predicting 94D SMPL-X parameters from images"""
    
    def __init__(self, input_size=224*224*3, num_classes=94):
        super(MLP, self).__init__()
        
        self.features = nn.Sequential(
            # First extract image features through convolutional layers
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((7, 7))
        )
        
        # Calculate flattened size
        self.flatten = nn.Flatten()
        
        # MLP layers
        self.classifier = nn.Sequential(
            nn.Linear(256 * 7 * 7, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x


def create_mlp(num_classes=94, **kwargs):
    """Create MLP model"""
    return MLP(num_classes=num_classes)


if __name__ == '__main__':
    model = create_mlp()
    print(f'MLP model created')
    print(f'Total parameters: {sum(p.numel() for p in model.parameters()):,}')
    
    # Test forward pass
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    print(f'Input shape: {x.shape}')
    print(f'Output shape: {y.shape}')
