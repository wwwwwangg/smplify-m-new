import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from mobilenetv3 import MobileNetV3_Small  # 替换成你实际使用的模型名
from datasets import SMPLXDataset,SMPLXInferenceDataset  # 如果你把 SMPLXDataset 放在了单独的文件里
import pickle
from tqdm import tqdm

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. 加载模型
model = MobileNetV3_Small()  # 替换为你的模型定义
checkpoint_path = "smplify_pth/checkpoint-best.pth"
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model'] if 'model' in checkpoint else checkpoint)
model.to(device)
model.eval()

# 🔍 添加统计参数数量的代码
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"📦 模型总参数量: {total_params:,}")
print(f"🧠 可训练参数量: {trainable_params:,}")

# 2. 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 根据训练时输入尺寸设置
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 根据训练时设置
])

# 3. 加载数据集
dataset = SMPLXInferenceDataset(root_dir="pred_file", transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# 4. 预测并保存结果
output_dir = "predictions"
os.makedirs(output_dir, exist_ok=True)

with torch.no_grad():
    for i, (image, _) in enumerate(tqdm(dataloader)):
        image = image.to(device)
        pred = model(image)  # 输出是姿态参数向量

        pred_np = pred.squeeze().cpu().numpy()
        save_path = os.path.join(output_dir, f"sample_{i:04d}.pkl")
        with open(save_path, "wb") as f:
            pickle.dump({"pred_pose": pred_np}, f)

print("✅ 预测完成，保存至:", output_dir)
