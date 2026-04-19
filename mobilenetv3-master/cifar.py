import numpy as np
import os
from PIL import Image


def unpickle(file):
    with open(file, 'rb') as fo:
        data_dict = np.load(fo, encoding='bytes', allow_pickle=True)
    return data_dict


def convert_cifar10_bin_to_images(bin_file, output_dir):
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 读取二进制文件
    with open(bin_file, 'rb') as f:
        data = np.fromfile(f, dtype=np.uint8)

    # 解析数据（每个样本3073字节）
    data = data.reshape(-1, 3073)

    # 分离标签和图像数据
    labels = data[:, 0]
    images = data[:, 1:].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

    # 保存为PNG图像
    for idx, (img, label) in enumerate(zip(images, labels)):
        img_pil = Image.fromarray(img)
        class_dir = os.path.join(output_dir, str(label))
        os.makedirs(class_dir, exist_ok=True)
        img_pil.save(os.path.join(class_dir, f"{idx}.png"))


# 示例：转换训练集的一个批次
convert_cifar10_bin_to_images(
    bin_file="data_batch_1.bin",
    output_dir="./cifar10_train"
)

# 转换测试集
convert_cifar10_bin_to_images(
    bin_file="test_batch.bin",
    output_dir="./cifar10_test"
)