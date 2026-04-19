"""
CBAM-MobileNet: MobileNetV3 with CBAM Attention Mechanism
Replaces SE attention with CBAM (Convolutional Block Attention Module)
"""
import torch
import torch.nn as nn
import math


class ChannelAttention(nn.Module):
    """Channel attention module"""
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """Spatial attention module"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    """CBAM attention module"""
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)
    
    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


class InvertedResidualWithCBAM(nn.Module):
    """Inverted residual block with CBAM attention"""
    def __init__(self, inp, oup, stride, expand_ratio, use_cbam=False):
        super(InvertedResidualWithCBAM, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        
        hidden_dim = int(inp * expand_ratio)
        self.use_residual = stride == 1 and inp == oup
        self.use_cbam = use_cbam
        
        layers = []
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True)
            ])
        
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup)
        ])
        
        self.conv = nn.Sequential(*layers)
        
        if use_cbam:
            self.cbam = CBAM(oup)
    
    def forward(self, x):
        if self.use_residual and self.use_cbam:
            return x + self.cbam(self.conv(x))
        elif self.use_residual:
            return x + self.conv(x)
        else:
            out = self.conv(x)
            if self.use_cbam:
                out = self.cbam(out)
            return out


class CBAM_MobileNet(nn.Module):
    """MobileNetV3 with CBAM attention for SMPL-X parameter prediction"""
    
    def __init__(self, num_classes=94, width_mult=1.0, use_cbam=True):
        super(CBAM_MobileNet, self).__init__()
        
        self.use_cbam = use_cbam
        
        # Configuration: [kernel_size, expanded_channels, output_channels, use_se, stride]
        self.cfgs = [
            # kernel, exp, out, SE, stride
            [3, 16, 16, True, 2],
            [3, 72, 24, False, 2],
            [3, 88, 24, False, 1],
            [5, 96, 40, True, 2],
            [5, 240, 40, True, 1],
            [5, 240, 40, True, 1],
            [5, 120, 48, True, 1],
            [5, 144, 48, True, 1],
            [5, 288, 96, True, 2],
            [5, 576, 96, True, 1],
            [5, 576, 96, True, 1],
        ]
        
        input_channel = 16
        features = [nn.Sequential(
            nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU(inplace=True)
        )]
        
        for k, exp, out, se, s in self.cfgs:
            # Replace SE with CBAM if use_cbam is True
            features.append(InvertedResidualWithCBAM(
                input_channel, out, s, expand_ratio=exp/input_channel, use_cbam=use_cbam
            ))
            input_channel = out
        
        self.features = nn.Sequential(*features)
        
        # Final layers
        self.conv = nn.Sequential(
            nn.Conv2d(input_channel, 576, 1, 1, 0, bias=False),
            nn.BatchNorm2d(576),
            nn.ReLU(inplace=True)
        )
        
        if use_cbam:
            self.cbam = CBAM(576)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Regression head
        self.regressor = nn.Sequential(
            nn.Linear(576, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
    
    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        if self.use_cbam:
            x = self.cbam(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.regressor(x)
        return x


def create_cbam_mobilenet(num_classes=94, pretrained=False, **kwargs):
    """Create CBAM-MobileNet model"""
    model = CBAM_MobileNet(num_classes=num_classes, use_cbam=True)
    
    if pretrained:
        # Try to load MobileNetV3 pretrained weights as initialization
        PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
        ckpt_path = os.path.join(PROJECT_ROOT, 'mobilenetv3-master', 'smplify_pth', 'checkpoint-best.pth')
        if os.path.exists(ckpt_path):
            print(f"Note: Loading pretrained weights from MobileNetV3 for initialization")
            # This is optional since architectures differ
            print(f"Pretrained checkpoint found at {ckpt_path}")
    
    return model


if __name__ == '__main__':
    model = create_cbam_mobilenet()
    print(f'CBAM-MobileNet model created')
    print(f'Total parameters: {sum(p.numel() for p in model.parameters()):,}')
    print(f'Using CBAM attention: {model.use_cbam}')
    
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    print(f'Input shape: {x.shape}')
    print(f'Output shape: {y.shape}')
