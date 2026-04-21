"""
基于深度学习的智能环保监测算法研究
训练脚本 - 多模态环保监测模型训练
黄丽佳 22460525

包含：CNN图像分类、LSTM时序预测、多模态融合模型训练
支持真实数据和模拟数据两种模式
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import argparse
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.cnn_detector import WaterQualityCNN, build_model
from models.lstm_predictor import LSTMPredictor, TransformerPredictor, TCNPredictor, build_predictor
from models.fusion_model import MultiModalFusion, build_fusion_model


class WaterImageDataset(Dataset):
    """水质图像数据集 - 从目录加载真实图像"""
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []
        self.labels = []

        # 加载图像路径和标签
        if os.path.exists(data_dir):
            for cls_idx, cls_name in enumerate(sorted(os.listdir(data_dir))):
                cls_dir = os.path.join(data_dir, cls_name)
                if os.path.isdir(cls_dir):
                    for fname in os.listdir(cls_dir):
                        if fname.lower().endswith(('.jpg', '.png', '.jpeg')):
                            self.samples.append(os.path.join(cls_dir, fname))
                            self.labels.append(cls_idx)

        print(f"加载图像数据集: {len(self.samples)} 张图像, {len(set(self.labels))} 个类别")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        import cv2
        image = cv2.imread(self.samples[idx])
        if image is None:
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224)).astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)

        if self.transform:
            image = self.transform(image)

        return image, self.labels[idx]


class SensorTimeSeriesDataset(Dataset):
    """传感器时序数据集 - 从CSV加载真实数据"""
    def __init__(self, csv_path, seq_len=168, forecast_horizon=24):
        self.seq_len = seq_len
        self.forecast_horizon = forecast_horizon

        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            # 特征列
            feature_cols = ['PM2.5', 'PM10', 'SO2', 'NO2', 'O3', 'CO',
                           'temperature', 'humidity', 'wind_speed', 'pressure']
            available = [c for c in feature_cols if c in df.columns]

            # 标准化
            self.mean = df[available].mean()
            self.std = df[available].std() + 1e-8
            self.data = ((df[available] - self.mean) / self.std).values.astype(np.float32)

            # 目标列
            target_cols = ['PM2.5', 'PM10', 'SO2', 'NO2', 'O3', 'CO']
            available_targets = [c for c in target_cols if c in df.columns]
            self.targets = df[available_targets].values.astype(np.float32)

            print(f"加载时序数据集: {len(self.data)} 条记录, {len(available)} 维特征")
        else:
            self.data = None
            self.targets = None
            print(f"数据文件不存在: {csv_path}，将使用模拟数据")

    def __len__(self):
        if self.data is not None:
            return max(0, len(self.data) - self.seq_len - self.forecast_horizon)
        return 1000  # 模拟数据长度

    def __getitem__(self, idx):
        if self.data is not None:
            x = self.data[idx:idx + self.seq_len]
            y = self.targets[idx + self.seq_len:idx + self.seq_len + self.forecast_horizon]
        else:
            # 模拟数据（带周期性和趋势）
            t = np.arange(idx, idx + self.seq_len + self.forecast_horizon)
            base = 50 + 20 * np.sin(t / 24 * np.pi) + np.random.normal(0, 3, len(t))
            x = np.stack([base[:self.seq_len]] * 10, axis=1).astype(np.float32)
            y = np.stack([base[self.seq_len:]] * 6, axis=1).astype(np.float32)

        return torch.from_numpy(x), torch.from_numpy(y)


def train_image_model(args):
    """训练CNN图像分类模型"""
    print("\n" + "=" * 50)
    print("开始训练CNN图像分类模型（ResNet50 + SE注意力）")
    print("=" * 50)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

    # 构建模型
    model = build_model('classifier', num_classes=6, pretrained=True).to(device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    # 加载数据集
    data_dir = os.path.join(args.data_dir, 'water_images')
    if os.path.exists(data_dir):
        dataset = WaterImageDataset(data_dir)
        dataloader = DataLoader(dataset, batch_size=args.batch, shuffle=True, num_workers=4)
    else:
        print(f"图像数据目录不存在: {data_dir}，使用模拟数据训练")
        dataloader = [(torch.randn(args.batch, 3, 224, 224),
                       torch.randint(0, 6, (args.batch,))) for _ in range(10)]

    # 训练循环
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for images, labels in dataloader:
            if isinstance(images, torch.Tensor):
                images, labels = images.to(device), labels.to(device)
            else:
                images, labels = images.to(device), labels.to(device)

            quality_out, pollution_out = model(images)
            loss = criterion(quality_out, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = quality_out.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        scheduler.step()

        if (epoch + 1) % 10 == 0:
            acc = 100. * correct / total
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch [{epoch+1}/{args.epochs}] Loss: {avg_loss:.4f} Acc: {acc:.2f}% LR: {scheduler.get_last_lr()[0]:.6f}")

    # 保存模型
    save_path = os.path.join(args.save_dir, 'cnn_model.pth')
    torch.save(model.state_dict(), save_path)
    print(f"CNN模型已保存: {save_path}")
    return model


def train_temporal_model(args):
    """训练LSTM时序预测模型"""
    print("\n" + "=" * 50)
    print("开始训练LSTM时序预测模型")
    print("=" * 50)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = build_predictor('lstm', input_size=13, hidden_size=128,
                           num_layers=2, output_size=6, forecast_horizon=24).to(device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    criterion = nn.MSELoss()

    # 加载数据集
    csv_path = os.path.join(args.data_dir, 'sensor_data.csv')
    dataset = SensorTimeSeriesDataset(csv_path, seq_len=168, forecast_horizon=24)
    dataloader = DataLoader(dataset, batch_size=args.batch, shuffle=True, num_workers=4)

    # 训练循环
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0

        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            predictions, attn_weights = model(inputs)
            loss = criterion(predictions, targets)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()

        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(dataloader)
            rmse = np.sqrt(avg_loss)
            print(f"Epoch [{epoch+1}/{args.epochs}] Loss: {avg_loss:.4f} RMSE: {rmse:.4f}")

    save_path = os.path.join(args.save_dir, 'lstm_model.pth')
    torch.save(model.state_dict(), save_path)
    print(f"LSTM模型已保存: {save_path}")
    return model


def train_fusion_model(args):
    """训练多模态融合模型"""
    print("\n" + "=" * 50)
    print("开始训练多模态融合模型（交叉注意力融合）")
    print("=" * 50)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = build_fusion_model(num_classes=6).to(device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr * 0.5, weight_decay=1e-4)
    cls_criterion = nn.CrossEntropyLoss()
    pred_criterion = nn.MSELoss()

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0

        for batch_idx in range(10):  # 模拟训练（实际使用时替换为真实数据加载）
            images = torch.randn(args.batch, 3, 224, 224).to(device)
            temporal = torch.randn(args.batch, 168, 13).to(device)
            cls_labels = torch.randint(0, 6, (args.batch,)).to(device)

            cls_out, pred_out, alert_out = model(images, temporal)

            cls_loss = cls_criterion(cls_out, cls_labels)
            pred_loss = pred_criterion(pred_out, torch.randn_like(pred_out))
            loss = cls_loss + 0.1 * pred_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / 10
            print(f"Epoch [{epoch+1}/{args.epochs}] Loss: {avg_loss:.4f}")

    save_path = os.path.join(args.save_dir, 'fusion_model.pth')
    torch.save(model.state_dict(), save_path)
    print(f"融合模型已保存: {save_path}")
    return model


def main():
    parser = argparse.ArgumentParser(description='环保监测模型训练')
    parser.add_argument('--model', type=str, default='all',
                       choices=['cnn', 'lstm', 'fusion', 'all'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--data_dir', type=str, default='data')

    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    print(f"环保监测模型训练")
    print(f"模型: {args.model} | Epochs: {args.epochs} | Batch: {args.batch} | LR: {args.lr}")
    print(f"设备: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

    if args.model in ['cnn', 'all']:
        train_image_model(args)

    if args.model in ['lstm', 'all']:
        train_temporal_model(args)

    if args.model in ['fusion', 'all']:
        train_fusion_model(args)

    print("\n所有训练完成！")


if __name__ == '__main__':
    main()
