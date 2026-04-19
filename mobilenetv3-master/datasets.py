# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os, lmdb, pickle, six
from PIL import Image
import torch
from torchvision import datasets, transforms

from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.data import create_transform
from torchvision.transforms import InterpolationMode

from torch.utils.data import Dataset
from PIL import Image
import json
import os

from torch.utils.data import Dataset
import json
import os
from PIL import Image
import torch

# 推理专用数据集加载类
class SMPLXInferenceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.img_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(self.img_dir) if f.endswith(".jpg")]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.img_dir, img_name)

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, img_name



class SMPLXDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.img_dir = os.path.join(root_dir, "img")
        self.label_dir = os.path.join(root_dir, "label")
        self.transform = transform

        # 读取所有图片文件名
        self.image_files = [f for f in os.listdir(self.img_dir) if f.endswith(".jpg")]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.img_dir, img_name)

        # 获取对应的 label 文件路径
        label_folder = os.path.join(self.label_dir, img_name.replace(".jpg", ""))
        pkl_path = os.path.join(label_folder, "000.pkl")

        # 读取图像
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # 读取 pkl 文件
        with open(pkl_path, "rb") as f:
            label_data = pickle.load(f)  # 这里会返回一个字典，具体内容要检查 pkl 结构

        # 提取关键信息并转换为 Tensor
        betas = torch.tensor(label_data["betas"].reshape(-1), dtype=torch.float32)                 # (16,)
        global_orient = torch.tensor(label_data["global_orient"].reshape(-1), dtype=torch.float32) # (3,)
        left_hand_pose = torch.tensor(label_data["left_hand_pose"].reshape(-1), dtype=torch.float32) # (12,)
        right_hand_pose = torch.tensor(label_data["right_hand_pose"].reshape(-1), dtype=torch.float32) # (12,)
        jaw_pose = torch.tensor(label_data["jaw_pose"].reshape(-1), dtype=torch.float32) # (3,)
        leye_pose = torch.tensor(label_data["leye_pose"].reshape(-1), dtype=torch.float32) # (3,)
        reye_pose = torch.tensor(label_data["reye_pose"].reshape(-1), dtype=torch.float32) # (3,)
        expression = torch.tensor(label_data["expression"].reshape(-1), dtype=torch.float32) # (10,)
        body_pose = torch.tensor(label_data["body_pose"].reshape(-1), dtype=torch.float32) # (32,)

        # 拼接成最终的标签 Tensor
        label_tensor = torch.cat([betas, global_orient, left_hand_pose, right_hand_pose,
                                  jaw_pose, leye_pose, reye_pose, expression, body_pose])

        return image, label_tensor


class ImageFolderLMDB(torch.utils.data.Dataset):
    def __init__(self, db_path, transform=None):
        self.db_path = db_path
        self.env = lmdb.open(db_path, subdir=False, readonly=True, lock=False, readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = pickle.loads(txn.get(b'__len__'))
            self.keys = pickle.loads(txn.get(b'__keys__'))
        self.transform = transform

    def __getitem__(self, idx):
        env = self.env
        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[idx])
        unpacked = pickle.loads(byteflow)

        # load image
        imgbuf = unpacked[0]
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')

        # load label
        label = unpacked[1]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return self.length

def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    print("Transform = ")
    if isinstance(transform, tuple):
        for trans in transform:
            print(" - - - - - - - - - - ")
            for t in trans.transforms:
                print(t)
    else:
        for t in transform.transforms:
            print(t)
    print("---------------------------")

    if args.data_set == 'CIFAR':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform, download=True)
        nb_classes = 100
    elif args.data_set == 'IMNET':
        print("reading from datapath", args.data_path)
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif args.data_set == 'IMNET_LMDB':
        print("reading from datapath", args.data_path)
        path = os.path.join(args.data_path, 'train.lmdb' if is_train else 'val.lmdb')
        dataset = ImageFolderLMDB(path, transform=transform)
        nb_classes = 1000
    elif args.data_set == "image_folder":
        root = args.data_path if is_train else args.eval_data_path
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = args.nb_classes
        assert len(dataset.class_to_idx) == nb_classes
    # 自定义数据集
    elif args.data_set == "smplx":
        root = args.data_path if is_train else args.eval_data_path
        dataset = SMPLXDataset(root, transform=transform)
        nb_classes = 94 # 表示所有特征相加的结果

    else:
        raise NotImplementedError()
    print("Number of the class = %d" % nb_classes)

    return dataset, nb_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
    mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
    std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD

    # train模式会经过并且break接下来的代码
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter, # 颜色抖动强度
            auto_augment=args.aa, # 自动增强策略
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        if not resize_im:
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        # warping (no cropping) when evaluated at 384 or larger
        if args.input_size >= 384:  
            t.append(
            transforms.Resize((args.input_size, args.input_size), 
                            interpolation=transforms.InterpolationMode.BICUBIC), 
        )
            print(f"Warping {args.input_size} size input images...")
        else:
            if args.crop_pct is None:
                args.crop_pct = 224 / 256
            size = int(args.input_size / args.crop_pct)
            t.append(
                # to maintain same ratio w.r.t. 224 images
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),  
            )
            t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
