"""
MobileNetV3-Small based SMPL-X initialization network
Predicts: betas (10D), body_pose (32D axis-angle), transl (3D)
"""
import sys
import os
import torch
import torch.nn as nn

# Add mobilenetv3-master to path
_mobilenet_dir = os.path.join(os.path.dirname(__file__), '..', 'mobilenetv3-master')
sys.path.insert(0, _mobilenet_dir)

from mobilenetv3 import MobileNetV3_Small


class SMPLInitNet(nn.Module):
    """
    MobileNetV3-Small wrapper for SMPL-X initialization.
    
    Input: RGB image (B, 3, 224, 224), normalized with mean=0.5, std=0.5
    Output: Tuple of (betas, body_pose_aa, transl)
        - betas: (B, 10) - Body shape parameters
        - body_pose_aa: (B, 32) - Body pose in axis-angle (from training data)
        - transl: (B, 3) - Translation (estimated)
    """
    
    def __init__(self, ckpt_path, device='cpu', num_betas=10):
        super(SMPLInitNet, self).__init__()
        self.num_betas = num_betas
        
        # Load MobileNetV3-Small with 94D output
        self.backbone = MobileNetV3_Small(num_classes=94)
        
        # Load pretrained weights if available
        if ckpt_path and os.path.exists(ckpt_path):
            print(f"Loading MobileNetV3 checkpoint from: {ckpt_path}")
            # Check torch version for weights_only support (>=1.13)
            torch_version = tuple(int(x) for x in torch.__version__.split('.')[:2])
            load_kwargs = {'map_location': 'cpu'}
            if torch_version >= (1, 13):
                load_kwargs['weights_only'] = False
            
            checkpoint = torch.load(ckpt_path, **load_kwargs)
            
            # Handle different checkpoint formats
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Remove 'module.' prefix if present
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
            
            self.backbone.load_state_dict(new_state_dict, strict=False)
            print("  Successfully loaded pretrained weights")
        else:
            print(f"Warning: Checkpoint not found at {ckpt_path}, using random initialization")
        
        # Freeze backbone parameters (no training for now)
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        self.backbone.eval()
        self.to(device)
    
    def forward(self, image):
        """
        Forward pass to predict SMPL-X initialization parameters.
        
        Args:
            image: (B, 3, 224, 224) tensor, normalized to [-1, 1]
        
        Returns:
            betas: (B, 10)
            body_pose_aa: (B, 32) - Last 32D of 94D output
            transl: (B, 3) - Placeholder, will be refined by guess_init
        """
        # Get full 94D prediction
        full_output = self.backbone(image)  # (B, 94)
        
        # Extract components based on training data layout:
        # betas(16) + global_orient(3) + left_hand(12) + right_hand(12) 
        # + jaw(3) + leye(3) + reye(3) + expression(10) + body_pose(32) = 94
        
        # betas: first 16D -> take first 10D
        betas = full_output[:, :self.num_betas]  # (B, 10)
        
        # body_pose: last 32D (indices 62-93)
        body_pose_aa = full_output[:, 62:]  # (B, 32)
        
        # transl: not in training output, use placeholder
        # Will be replaced by guess_init Z value
        transl = torch.zeros(full_output.shape[0], 3, 
                            dtype=full_output.dtype, 
                            device=full_output.device)
        
        return betas, body_pose_aa, transl


def create_init_net(ckpt_path, device='cpu'):
    """Helper function to create SMPLInitNet"""
    return SMPLInitNet(ckpt_path=ckpt_path, device=device)
