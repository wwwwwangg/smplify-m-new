"""
ConvNeXt V2 for SMPL-X Parameter Prediction
Modern CNN architecture with improved design
"""
import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """LayerNorm that supports channels_last format"""
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        self.dim = -1 if self.data_format == "channels_last" else 1
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return nn.functional.layer_norm(x, x.shape[self.dim:], self.weight, self.bias, self.eps)
        else:
            u = x.mean(self.dim, keepdim=True)
            s = (x - u).pow(2).mean(self.dim, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class GRN(nn.Module):
    """Global Response Normalization"""
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))
    
    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class Block(nn.Module):
    """ConvNeXt V2 Block"""
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = nn.Identity()
    
    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        return input + x


class ConvNeXtV2(nn.Module):
    """ConvNeXt V2 for SMPL-X parameter prediction"""
    
    def __init__(self, num_classes=94, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768]):
        super().__init__()
        
        self.downsample_layers = nn.ModuleList()
        # Stem
        stem = nn.Sequential(
            nn.Conv2d(3, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        
        # Downsampling layers
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2)
            )
            self.downsample_layers.append(downsample_layer)
        
        # Stages
        self.stages = nn.ModuleList()
        for i in range(4):
            stage = nn.Sequential(*[
                Block(dim=dims[i]) for _ in range(depths[i])
            ])
            self.stages.append(stage)
        
        # Final layers
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Sequential(
            nn.Linear(dims[-1], 1024),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1]))
    
    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


def create_convnextv2(num_classes=94, pretrained=False, **kwargs):
    """Create ConvNeXt V2 model"""
    # Tiny version
    model = ConvNeXtV2(num_classes=num_classes, depths=[3, 3, 9, 3], dims=[96, 192, 384, 764])
    return model


if __name__ == '__main__':
    model = create_convnextv2()
    print(f'ConvNeXt V2 model created')
    print(f'Total parameters: {sum(p.numel() for p in model.parameters()):,}')
    
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    print(f'Input shape: {x.shape}')
    print(f'Output shape: {y.shape}')
