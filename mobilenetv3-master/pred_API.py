import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from mobilenetv3 import MobileNetV3_Small  # 替换成你实际使用的模型名
from datasets import SMPLXInferenceDataset  # 如果你把 SMPLXInferenceDataset 放在了单独的文件里
import pickle
from tqdm import tqdm
from PIL import Image


class PosePredictor:
    def __init__(self, model_path="/mnt/d/lxf/smplify-x-master/mobilenetv3-master/smplify_pth/checkpoint-best.pth", device=None):
        # 设置设备
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1. 加载模型
        self.model = MobileNetV3_Small()  # 替换为你的模型定义
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model'] if 'model' in checkpoint else checkpoint)
        self.model.to(self.device)
        self.model.eval()

        # 2. 数据预处理
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # 根据训练时输入尺寸设置
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 根据训练时设置
        ])

    def predict(self, image_path):
        """
        接受图片路径，返回预测的94个人体姿态参数。
        :param image_path: 图片路径
        :return: 94个姿态参数的 Tensor
        """
        # 读取并预处理图片
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image).unsqueeze(0)  # 添加 batch 维度

        # 进行预测
        with torch.no_grad():
            image = image.to(self.device)
            pred = self.model(image)  # 输出是姿态参数向量

        return pred.squeeze()  # 返回 94 个姿态参数的 Tensor

    def predict_batch(self, image_paths):
        """
        批量预测
        :param image_paths: 图片路径列表
        :return: 94个人体姿态参数的列表
        """
        predictions = []
        for image_path in tqdm(image_paths, desc="Predicting"):
            pred = self.predict(image_path)
            predictions.append(pred.cpu().numpy())
        return predictions

    def save_predictions(self, predictions, output_dir):
        """
        保存预测结果到指定目录
        :param predictions: 预测结果
        :param output_dir: 保存结果的目录
        """
        os.makedirs(output_dir, exist_ok=True)
        for i, pred in enumerate(predictions):
            save_path = os.path.join(output_dir, f"sample_{i:04d}.pkl")
            with open(save_path, "wb") as f:
                pickle.dump({"pred_pose": pred}, f)
        print("✅ 预测完成，保存至:", output_dir)
