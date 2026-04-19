"""
MobileNetV3 Training Script for SMPL-X Parameter Prediction
Uses SMPLify-X output PKL files as ground truth labels
"""
import os
import sys
import json
import pickle
import time
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# Add paths
PROJECT_ROOT = '/mnt/e/Lxf_test/smplify-x-master'
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'mobilenetv3-master'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'smplifyx'))

from mobilenetv3 import MobileNetV3_Small

class SMPLXTrainDataset(Dataset):
    """Dataset using images and SMPLify-X output PKL files"""
    
    def __init__(self, img_dir, pkl_dirs, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.samples = []
        
        # Find all image-pkl pairs
        for pkl_dir in pkl_dirs:
            sample_name = os.path.basename(pkl_dir)
            img_path = os.path.join(img_dir, sample_name + '.jpg')
            pkl_path = os.path.join(pkl_dir, '000.pkl')
            
            if os.path.exists(img_path) and os.path.exists(pkl_path):
                self.samples.append({
                    'img': img_path,
                    'pkl': pkl_path,
                    'name': sample_name
                })
        
        print(f"Found {len(self.samples)} training samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        image = Image.open(sample['img']).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # Load PKL
        with open(sample['pkl'], 'rb') as f:
            pkl_data = pickle.load(f)
        
        # Extract SMPL-X parameters (94D)
        betas = pkl_data['betas'].flatten()[:16]  # 16D
        global_orient = pkl_data['global_orient'].flatten()[:3]  # 3D
        left_hand = pkl_data['left_hand_pose'].flatten()[:12]  # 12D
        right_hand = pkl_data['right_hand_pose'].flatten()[:12]  # 12D
        jaw = pkl_data['jaw_pose'].flatten()[:3]  # 3D
        leye = pkl_data['leye_pose'].flatten()[:3]  # 3D
        reye = pkl_data['reye_pose'].flatten()[:3]  # 3D
        expression = pkl_data['expression'].flatten()[:10]  # 10D
        body_pose = pkl_data['body_pose'].flatten()[:32]  # 32D
        
        # Concatenate all
        label = np.concatenate([betas, global_orient, left_hand, right_hand,
                                jaw, leye, reye, expression, body_pose])
        
        return image, torch.tensor(label, dtype=torch.float32)


def train_model(model, train_loader, val_loader, device, epochs=300, lr=1e-3, 
                output_dir='smplify_pth_retrain'):
    """Train MobileNetV3 model"""
    
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, 'training_log.json')
    
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    log_entries = []
    best_mae = float('inf')
    best_model_state = None
    
    print(f"\nStarting training: {epochs} epochs, lr={lr}, device={device}")
    print(f"Training samples: {len(train_loader.dataset)}")
    if val_loader:
        print(f"Validation samples: {len(val_loader.dataset)}")
    print("=" * 80)
    
    start_time = time.time()
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_losses = []
        train_maes = []
        train_mses = []
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            mae = torch.nn.functional.l1_loss(outputs, labels, reduction='mean').item()
            mse = torch.nn.functional.mse_loss(outputs, labels, reduction='mean').item()
            
            train_losses.append(loss.item())
            train_maes.append(mae)
            train_mses.append(mse)
        
        scheduler.step()
        
        avg_train_loss = np.mean(train_losses)
        avg_train_mae = np.mean(train_maes)
        avg_train_mse = np.mean(train_mses)
        
        # Validation
        avg_val_loss = avg_train_loss
        avg_val_mae = avg_train_mae
        avg_val_mse = avg_train_mse
        
        if val_loader:
            model.eval()
            val_losses, val_maes, val_mses = [], [], []
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    
                    mae = torch.nn.functional.l1_loss(outputs, labels, reduction='mean').item()
                    mse = torch.nn.functional.mse_loss(outputs, labels, reduction='mean').item()
                    
                    val_losses.append(loss.item())
                    val_maes.append(mae)
                    val_mses.append(mse)
            
            avg_val_loss = np.mean(val_losses)
            avg_val_mae = np.mean(val_maes)
            avg_val_mse = np.mean(val_mses)
        
        # Log
        log_entry = {
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'train_mae': avg_train_mae,
            'train_mse': avg_train_mse,
            'val_loss': avg_val_loss,
            'val_mae': avg_val_mae,
            'val_mse': avg_val_mse,
            'lr': scheduler.get_last_lr()[0]
        }
        log_entries.append(log_entry)
        
        # Save best model
        if avg_val_mae < best_mae:
            best_mae = avg_val_mae
            best_model_state = model.state_dict().copy()
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train Loss: {avg_train_loss:.4f} | "
                  f"Train MAE: {avg_train_mae:.4f} | "
                  f"Train MSE: {avg_train_mse:.4f} | "
                  f"Val MAE: {avg_val_mae:.4f} | "
                  f"Best MAE: {best_mae:.4f} | "
                  f"Time: {elapsed:.1f}s")
    
    # Save final model
    total_time = time.time() - start_time
    
    # Save best model
    if best_model_state is not None:
        best_path = os.path.join(output_dir, 'checkpoint-best.pth')
        torch.save(best_model_state, best_path)
        print(f"\nBest model saved to {best_path} with MAE={best_mae:.4f}")
    
    # Save last model
    last_path = os.path.join(output_dir, f'checkpoint-{epochs-1}.pth')
    torch.save(model.state_dict(), last_path)
    print(f"Last model saved to {last_path}")
    
    # Save log
    with open(log_path, 'w') as f:
        json.dump(log_entries, f, indent=2)
    print(f"Training log saved to {log_path}")
    
    # Summary
    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"Final Train MAE: {avg_train_mae:.4f}")
    print(f"Final Train MSE: {avg_train_mse:.4f}")
    print(f"Final Val MAE: {avg_val_mae:.4f}")
    print(f"Final Val MSE: {avg_val_mse:.4f}")
    print(f"Best Val MAE: {best_mae:.4f}")
    
    return log_entries, best_mae


def main():
    # Configuration
    img_dir = '/mnt/e/Lxf_test/smplify-x-master/data/train_images'
    pkl_dirs = [
        '/mnt/e/Lxf_test/smplify-x-master/smplx_mobilenet_output/results/sample_0000',
        '/mnt/e/Lxf_test/smplify-x-master/smplx_mobilenet_output/results/sample_0001',
        '/mnt/e/Lxf_test/smplify-x-master/smplx_mobilenet_output/results/sample_0002',
        '/mnt/e/Lxf_test/smplify-x-master/smplx_mobilenet_output/results/sample_0003',
    ]
    output_dir = '/mnt/e/Lxf_test/smplify-x-master/mobilenetv3-master/smplify_pth_retrain'
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Transforms
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Dataset
    full_dataset = SMPLXTrainDataset(img_dir, pkl_dirs, transform=train_transform)
    
    # Split train/val (3 train, 1 val)
    train_size = 3
    val_size = 1
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    val_dataset.dataset.transform = val_transform
    
    # DataLoaders - use full batch for small dataset to avoid BatchNorm issues
    train_loader = DataLoader(train_dataset, batch_size=3, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    # Model
    model = MobileNetV3_Small(num_classes=94)
    model = model.to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    log_entries, best_mae = train_model(
        model, train_loader, val_loader, device,
        epochs=300, lr=1e-3, output_dir=output_dir
    )
    
    print("\nTraining completed successfully!")


if __name__ == '__main__':
    main()
