"""
MobileNetV3 for SMPL-X Parameter Prediction
Using existing MobileNetV3 implementation
"""
import sys
import os

# Add mobilenetv3-master to path
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'mobilenetv3-master'))

import torch
import torch.nn as nn


def create_mobilenetv3(num_classes=94, pretrained=False, **kwargs):
    """Create MobileNetV3-Small model"""
    from mobilenetv3 import MobileNetV3_Small
    model = MobileNetV3_Small(num_classes=num_classes)
    
    if pretrained:
        ckpt_path = os.path.join(PROJECT_ROOT, 'mobilenetv3-master', 'smplify_pth', 'checkpoint-best.pth')
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
            model.load_state_dict(state_dict, strict=False)
            print(f"Loaded pretrained MobileNetV3 from {ckpt_path}")
    
    return model


if __name__ == '__main__':
    model = create_mobilenetv3()
    print(f'MobileNetV3 model created')
    print(f'Total parameters: {sum(p.numel() for p in model.parameters()):,}')
    
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    print(f'Input shape: {x.shape}')
    print(f'Output shape: {y.shape}')
