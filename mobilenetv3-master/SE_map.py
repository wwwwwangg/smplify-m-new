import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from torchvision import transforms
from PIL import Image

from mobilenetv3 import MobileNetV3_Small, MobileNetV3_Large

# 加载模型（假设你已修改了 MobileNetV3 模型，添加了 features 属性）
model = MobileNetV3_Small(num_classes=94)
model.eval()

# 加载图片
image = Image.open('test2.jpg').convert('RGB')
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
img_tensor = transform(image).unsqueeze(0)  # [1, 3, 224, 224]

# 前向传播，获取特征图
with torch.no_grad():
    output, features = model(img_tensor)

# 获取第一个 SE 模块的注意力图
first_se_block = model.bneck[0].se
attention = first_se_block.attention_map  # shape: [C]

# 可视化特征图
conv1_feature_map = features['conv1'].squeeze(0).cpu().detach().numpy()  # [C, H, W]
channel_idx = 0  # 选择某个通道的特征图进行显示

# 绘制输入图像与特征图的比较
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# 显示原始输入图像
ax[0].imshow(image)
ax[0].set_title("Original Image")
ax[0].axis("off")

# 显示选定通道的特征图（经过注意力加权）
ax[1].imshow(conv1_feature_map[channel_idx], cmap='viridis')  # 显示第一个通道的特征图
ax[1].set_title(f"Feature Map (Channel {channel_idx})")
ax[1].axis("off")

plt.tight_layout()
plt.savefig("feature_map_comparison.tif")
plt.show()

# 可视化 SE 通道注意力热力图
plt.figure(figsize=(10, 1))
sns.heatmap(attention.squeeze().cpu().detach().numpy().reshape(1, -1), cmap='viridis', cbar=True, xticklabels=False)
plt.title("SE Channel Attention (Block 0)")
plt.xlabel("Channel Index")
plt.tight_layout()
plt.savefig("se_attention_block0.tif")
plt.show()
