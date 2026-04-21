"""
LSTM + 时序注意力机制 - 空气质量预测模型
论文: 基于深度学习的智能环保监测算法研究
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import os

# ============= 数据加载 =============
data_dir = "PRSA_Data_20130301-20170228"  # 北京空气质量数据目录
csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
dfs = [pd.read_csv(os.path.join(data_dir, f)) for f in csv_files]
data = pd.concat(dfs, ignore_index=True)

# 特征列
feature_cols = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3", "TEMP", "PRES", "DEWP", "RAIN", "WSPM"]
target_col = "PM2.5"

# 数据清洗
data = data.dropna(subset=feature_cols + [target_col])
data["datetime"] = pd.to_datetime(data[["year", "month", "day", "hour"]])
data = data.sort_values(["station", "datetime"]).reset_index(drop=True)

# 标准化
scaler = StandardScaler()
data[feature_cols] = scaler.fit_transform(data[feature_cols])

# ============= 序列创建 =============
def create_sequences(data, seq_len=24, pred_len=6):
    X, y = [], []
    for station in data["station"].unique():
        station_data = data[data["station"] == station]
        values = station_data[feature_cols].values
        target_idx = feature_cols.index(target_col)
        for i in range(len(values) - seq_len - pred_len + 1):
            X.append(values[i:i+seq_len])
            y.append(values[i+seq_len:i+seq_len+pred_len, target_idx].mean())
    return np.array(X), np.array(y)

X, y = create_sequences(data, seq_len=24, pred_len=6)
print(f"数据形状: X={X.shape}, y={y.shape}")

# 划分训练集/测试集
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# ============= LSTM + 注意力模型 =============
class LSTM_Predictor(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout)
        self.attention = nn.Linear(hidden_size, 1)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # 时序注意力
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        return self.fc(context)

class AirDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(AirDataset(X_train, y_train), batch_size=256, shuffle=True)
test_loader = DataLoader(AirDataset(X_test, y_test), batch_size=256, shuffle=False)

# ============= 训练 =============
device = torch.device("cuda")
model = LSTM_Predictor(len(feature_cols)).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=5, factor=0.5)

print("开始训练...")
epochs = 30
best_loss = float("inf")

for epoch in range(epochs):
    model.train()
    train_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X_batch).squeeze(), y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            loss = criterion(model(X_batch).squeeze(), y_batch)
            val_loss += loss.item()
    
    train_loss /= len(train_loader)
    val_loss /= len(test_loader)
    scheduler.step(val_loss)
    
    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), "lstm_best.pth")
    
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}/{epochs} - Train: {train_loss:.4f}, Val: {val_loss:.4f}")

print(f"模型保存到 lstm_best.pth")