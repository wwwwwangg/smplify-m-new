"""
Model Factory for Comparative Experiments
Creates models based on configuration
"""
from .model_mlp import create_mlp
from .model_resnet import create_resnet50
from .model_convnextv2 import create_convnextv2
from .model_swin import create_swin_transformer
from .model_unet import create_unet
from .model_mobilenetv3 import create_mobilenetv3
from .model_cbam_mobilenet import create_cbam_mobilenet


# Model registry
MODEL_REGISTRY = {
    'mlp': create_mlp,
    'resnet50': create_resnet50,
    'convnextv2': create_convnextv2,
    'swin': create_swin_transformer,
    'unet': create_unet,
    'mobilenetv3': create_mobilenetv3,
    'cbam_mobilenet': create_cbam_mobilenet,
}


def create_model(model_name, num_classes=94, pretrained=False, **kwargs):
    """
    Create model by name
    
    Args:
        model_name: Name of the model (must be in MODEL_REGISTRY)
        num_classes: Number of output classes (default: 94 for SMPL-X)
        pretrained: Whether to load pretrained weights
        **kwargs: Additional arguments passed to model constructor
    
    Returns:
        PyTorch model
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}")
    
    return MODEL_REGISTRY[model_name](num_classes=num_classes, pretrained=pretrained, **kwargs)


def get_available_models():
    """Get list of available models"""
    return list(MODEL_REGISTRY.keys())


if __name__ == '__main__':
    print("Available models:")
    for name in get_available_models():
        model = create_model(name, num_classes=94, pretrained=False)
        params = sum(p.numel() for p in model.parameters())
        print(f"  {name:20s} - {params:>10,} parameters")
