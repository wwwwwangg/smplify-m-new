"""
Swin Transformer for SMPL-X Parameter Prediction
Hierarchical vision transformer with shifted windows
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Mlp(nn.Module):
    """MLP layer"""
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """Partition into windows"""
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """Reverse window partition"""
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """Window based multi-head self attention"""
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block"""
    def __init__(self, dim, num_heads, window_size=7, shift_size=0, mlp_ratio=4., drop=0., attn_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size=(window_size, window_size), 
                                     num_heads=num_heads, qkv_bias=True)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)
    
    def forward(self, x):
        H, W = self.H, self.W
        B, L, C = x.shape
        
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        
        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
        
        # Partition windows
        windows = window_partition(shifted_x, self.window_size)
        windows = windows.view(-1, self.window_size * self.window_size, C)
        
        # W-MSA
        attn_windows = self.attn(windows)
        
        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)
        
        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        
        return x


class BasicLayer(nn.Module):
    """Basic layer for Swin Transformer"""
    def __init__(self, dim, depth, num_heads, window_size, mlp_ratio=4., drop=0.):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, num_heads=num_heads, window_size=window_size,
                                shift_size=0 if (i % 2 == 0) else window_size // 2,
                                mlp_ratio=mlp_ratio, drop=drop)
            for i in range(depth)
        ])
    
    def forward(self, x):
        for blk in self.blocks:
            blk.H, blk.W = int(x.shape[1]**0.5), int(x.shape[1]**0.5)
            x = blk(x)
        return x


class SwinTransformer(nn.Module):
    """Swin Transformer for SMPL-X parameter prediction"""
    
    def __init__(self, num_classes=94, embed_dim=96, depths=[2, 2, 6, 2], 
                 num_heads=[3, 6, 12, 24], window_size=7, drop_rate=0.1):
        super().__init__()
        
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        
        # Split image into non-overlapping patches
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=4, stride=4)
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # Build layers
        self.layers = nn.ModuleList()
        for i_layer in range(len(depths)):
            layer = BasicLayer(dim=int(embed_dim * 2**i_layer),
                              depth=depths[i_layer],
                              num_heads=num_heads[i_layer],
                              window_size=window_size,
                              drop=drop_rate)
            self.layers.append(layer)
        
        # Final layers
        self.norm = nn.LayerNorm(int(embed_dim * 2**(len(depths)-1)))
        self.head = nn.Sequential(
            nn.Linear(int(embed_dim * 2**(len(depths)-1)), 1024),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Dropout(drop_rate * 0.7),
            nn.Linear(512, num_classes)
        )
    
    def forward_features(self, x):
        x = self.patch_embed(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)
        
        for layer in self.layers:
            x = layer(x)
        
        x = self.norm(x)
        return x.mean(dim=1)
    
    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


def create_swin_transformer(num_classes=94, pretrained=False, **kwargs):
    """Create Swin Transformer model (Tiny version)"""
    model = SwinTransformer(num_classes=num_classes, embed_dim=96, depths=[2, 2, 6, 2], 
                           num_heads=[3, 6, 12, 24], window_size=7)
    return model


if __name__ == '__main__':
    model = create_swin_transformer()
    print(f'Swin Transformer model created')
    print(f'Total parameters: {sum(p.numel() for p in model.parameters()):,}')
    
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    print(f'Input shape: {x.shape}')
    print(f'Output shape: {y.shape}')
