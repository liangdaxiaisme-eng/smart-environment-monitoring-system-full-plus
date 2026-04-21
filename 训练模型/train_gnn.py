"""
图神经网络 (GNN) - 多站点空气质量预测
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

# 数据路径
data_dir = "/hy-tmp/env_monitoring/data/PRSA_Data_20130301-20170228"

# 加载数据
csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
dfs = [pd.read_csv(os.path.join(data_dir, f)) for f in csv_files]
data = pd.concat(dfs, ignore_index=True)

# 特征列
features = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3", "TEMP", "PRES", "DEWP", "RAIN", "WSPM"]
target_col = "PM2.5"
data = data.dropna(subset=features + [target_col])

# 获取站点列表
stations = sorted(data["station"].unique())
num_nodes = len(stations)
print(f"站点数: {num_nodes}")

# 构建邻接矩阵 (每个站点与相邻站点连接)
adj = torch.eye(num_nodes)
for i in range(num_nodes):
    for j in range(i+1, min(i+3, num_nodes)):
        adj[i, j] = adj[j, i] = 1

# 标准化
scaler = StandardScaler()
data[features] = scaler.fit_transform(data[features])

# 创建序列数据
def create_sequences(data, seq_len=12):
    X, y = [], []
    for station in stations:
        station_data = data[data["station"] == station].reset_index(drop=True)
        values = station_data[features].values
        targets = station_data[target_col].values
        for i in range(len(values) - seq_len):
            X.append(values[i:i+seq_len])
            y.append(targets[i+seq_len])
    return np.array(X), np.array(y)

X, y = create_sequences(data)
y_mean, y_std = y.mean(), y.std()
y = (y - y_mean) / y_std

print(f"数据形状: X={X.shape}, y={y.shape}")

# 图卷积层
class GraphConvLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
    
    def forward(self, x, adj):
        # x: [batch, nodes, features]
        # adj: [nodes, nodes]
        batch_size = x.size(0)
        adj_batch = adj.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, nodes, nodes]
        support = self.linear(x)
        output = torch.bmm(adj_batch, support)
        return F.relu(output)

# GNN模型
class GNNModel(nn.Module):
    def __init__(self, num_nodes, feature_dim, hidden_dim=64):
        super().__init__()
        self.gcn1 = GraphConvLayer(feature_dim, hidden_dim)
        self.gcn2 = GraphConvLayer(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, 1)
    
    def forward(self, x, adj):
        # x: [batch, seq_len, feature_dim]
        batch_size, seq_len, feature_dim = x.size()
        
        # 取最后时刻的特征，然后扩展到所有节点
        x = x[:, -1, :]  # [batch, feature_dim]
        x = x.unsqueeze(1).expand(-1, num_nodes, -1)  # [batch, num_nodes, feature_dim]
        
        # 图卷积
        x = self.gcn1(x, adj)
        x = self.gcn2(x, adj)
        
        # 全局池化
        x = x.mean(dim=1)  # [batch, hidden_dim]
        
        return self.fc(x).squeeze(-1)

# 数据集
class GNNDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 划分数据
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

train_loader = DataLoader(GNNDataset(X_train, y_train), batch_size=256, shuffle=True)
test_loader = DataLoader(GNNDataset(X_test, y_test), batch_size=256)

# 模型
device = torch.device("cuda")
model = GNNModel(num_nodes, len(features)).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("开始训练 GNN...")
epochs = 15
best_loss = float('inf')

for epoch in range(epochs):
    # 训练
    model.train()
    train_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        output = model(X_batch, adj.to(device))
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    # 验证
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            output = model(X_batch, adj.to(device))
            loss = criterion(output, y_batch)
            val_loss += loss.item()
    
    train_loss /= len(train_loader)
    val_loss /= len(test_loader)
    
    if val_loss < best_loss:
        best_loss = val_loss
        torch.save({
            "model": model.state_dict(),
            "y_mean": y_mean,
            "y_std": y_std
        }, "/hy-tmp/env_monitoring/models/gnn_best.pth")
    
    if (epoch + 1) % 3 == 0:
        print(f"Epoch {epoch+1}/{epochs} - Train: {train_loss:.4f}, Val: {val_loss:.4f}")

print(f"GNN模型保存到 /hy-tmp/env_monitoring/models/gnn_best.pth")