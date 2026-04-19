"""
ResNet50 for SMPL-X Parameter Prediction
Using ResNet-50 backbone with custom regression head
"""
import torch
import torch.nn as nn


class ResNet50(nn.Module):
    """ResNet50 model for predicting 94D SMPL-X parameters from images"""
    
    def __init__(self, num_classes=94, pretrained=False):
        super(ResNet50, self).__init__()
        
        # Import torchvision ResNet
        from torchvision.models import resnet50, ResNet50_Weights
        
        # Create ResNet50
        if pretrained:
            weights = ResNet50_Weights.IMAGENET1K_V1
        else:
            weights = None
        
        resnet = resnet50(weights=weights)
        
        # Remove the original classifier
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
        # Custom regression head for SMPL-X parameters
        num_features = resnet.fc.in_features
        
        self.regressor = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
        self._initialize_new_weights()
    
    def _initialize_new_weights(self):
        """Initialize weights for new layers"""
        for m in self.regressor.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.regressor(x)
        return x


def create_resnet50(num_classes=94, pretrained=True, **kwargs):
    """Create ResNet50 model"""
    return ResNet50(num_classes=num_classes, pretrained=pretrained)


if __name__ == '__main__':
    model = create_resnet50(pretrained=False)
    print(f'ResNet50 model created')
    print(f'Total parameters: {sum(p.numel() for p in model.parameters()):,}')
    
    # Test forward pass
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    print(f'Input shape: {x.shape}')
    print(f'Output shape: {y.shape}')
