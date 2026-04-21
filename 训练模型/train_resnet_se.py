"""
ResNet18 + SE 注意力机制 - 水质分类模型
论文: 基于深度学习的智能环保监测算法研究
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import pandas as pd
import os

# ============= 数据准备 =============
data_dir = "PRSA_Data_20130301-20170228"  # 北京空气质量数据目录
features = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3", "TEMP", "PRES", "DEWP", "RAIN", "WSPM"]

# 加载数据
csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
dfs = [pd.read_csv(os.path.join(data_dir, f)) for f in csv_files]
data = pd.concat(dfs, ignore_index=True)
data = data.dropna(subset=features + ["PM2.5"])

# 水质等级标签 (基于PM2.5)
def get_label(pm25):
    if pm25 < 35: return 0      # 清洁
    elif pm25 < 75: return 1    # 轻度污染
    elif pm25 < 115: return 2   # 中度污染
    else: return 3              # 重度污染

data["label"] = data["PM2.5"].apply(get_label)

# ============= 数据集 =============
class WaterQualityDataset(Dataset):
    def __init__(self, data, transform=None, n=2000):
        self.data = data.sample(n=min(n, len(data)), random_state=42).reset_index(drop=True)
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        # 将11个特征转为11x11图像
        v = row[features].values.astype(np.float32)
        v = (v - v.min()) / (v.max() - v.min() + 1e-8) * 255
        img = np.zeros((11, 11), dtype=np.uint8)
        for i, val in enumerate(v):
            x, y = i % 11, i // 11
            img[y, x] = int(val)
        img = np.stack([img]*3, axis=2)  # 3通道
        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        return img, row["label"]

# ============= SE注意力模块 =============
class SE(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        return x * self.fc(y).view(b, c, 1, 1)

# ============= ResNet18 + SE 模型 =============
class ResNet18SE(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        resnet = models.resnet18(weights="IMAGENET1K_V1")
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        self.se = SE(512)
        self.fc = nn.Linear(512, num_classes)
    
    def forward(self, x):
        x = self.features(x)
        x = self.se(x)
        x = x.mean([-1, -2])  # 全局平均池化
        return self.fc(x)

# ============= 训练 =============
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

dataset = WaterQualityDataset(data, transform)
train_ds, val_ds = random_split(dataset, [int(len(dataset)*0.8), len(dataset)-int(len(dataset)*0.8)])
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64)

model = ResNet18SE(num_classes=4).cuda()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("开始训练...")
for epoch in range(10):
    # 训练
    model.train()
    correct, total = 0, 0
    for images, labels in train_loader:
        images, labels = images.cuda(), labels.cuda()
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    train_acc = 100. * correct / total
    
    # 验证
    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.cuda(), labels.cuda()
            outputs = model(images)
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()
    val_acc = 100. * val_correct / val_total
    
    print(f"Epoch {epoch+1}/10 - Train: {train_acc:.1f}%, Val: {val_acc:.1f}%")

# 保存模型
torch.save(model.state_dict(), "resnet_se_best.pth")
print("模型保存到 resnet_se_best.pth")