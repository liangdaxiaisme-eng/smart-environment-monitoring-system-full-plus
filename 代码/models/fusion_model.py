"""
基于深度学习的智能环保监测算法研究
多模态融合模型 - 图像+时序+空间数据融合
黄丽佳 22460525

基于图神经网络(GNN)的多模态融合框架，
联合视觉图像特征、时序传感器数据和地理空间信息，
构建全面的环境态势感知模型。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.cnn_detector import WaterQualityCNN, SEBlock
from models.lstm_predictor import LSTMPredictor


class MultiModalFusion(nn.Module):
    """
    多模态融合模型
    融合图像特征、时序特征和空间特征，
    实现综合环境质量评估与预测。
    
    融合策略：
    1. 图像分支：CNN提取视觉污染特征
    2. 时序分支：LSTM提取时间序列模式
    3. 空间分支：GCN建模传感器网络空间相关性
    4. 融合层：交叉注意力机制融合多模态特征
    """
    def __init__(self, num_classes=6, seq_len=168, input_size=13, 
                 hidden_size=128, forecast_horizon=24):
        super(MultiModalFusion, self).__init__()
        
        self.hidden_size = hidden_size
        self.forecast_horizon = forecast_horizon
        
        # 图像特征提取分支
        self.image_encoder = WaterQualityCNN(num_classes=num_classes, pretrained=True)
        self.image_proj = nn.Sequential(
            nn.Linear(2048, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
        # 时序特征提取分支
        self.temporal_encoder = LSTMPredictor(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            output_size=num_classes,
            forecast_horizon=forecast_horizon
        )
        
        # 空间特征提取分支（简化GCN）
        self.spatial_encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True)
        )
        
        # 交叉注意力融合
        self.cross_attention = CrossModalAttention(hidden_size, nhead=4)
        
        # 融合分类头
        self.fusion_classifier = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, num_classes)
        )
        
        # 预警预测头
        self.alert_predictor = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, 3)  # 低/中/高 三级预警
        )
    
    def forward(self, image_input, temporal_input, spatial_input=None):
        """
        Args:
            image_input: 环境图像 [batch, 3, 224, 224]
            temporal_input: 时序传感器数据 [batch, seq_len, input_size]
            spatial_input: 空间特征 [batch, num_sensors, input_size]
        Returns:
            classification: 环境质量等级分类
            prediction: 未来污染物浓度预测
            alert_level: 预警等级
        """
        # 图像特征提取
        quality_out, pollution_out = self.image_encoder(image_input)
        image_features = self.image_proj(
            self.image_encoder.global_pool(
                self.image_encoder.features[7](
                    self.image_encoder.features[6](
                        self.image_encoder.features[5](
                            self.image_encoder.features[4](
                                self.image_encoder.features[3](
                                    self.image_encoder.features[2](
                                        self.image_encoder.features[1](
                                            self.image_encoder.features[0](image_input)
                                        )))))))).view(image_input.size(0), -1)
        
        # 时序特征提取
        temporal_prediction, attn_weights = self.temporal_encoder(temporal_input)
        
        # 获取LSTM最后一层隐状态作为时序特征
        lstm_out, _ = self.temporal_encoder.lstm(
            self.temporal_encoder.input_embedding(temporal_input)
        )
        temporal_features = lstm_out[:, -1, :self.hidden_size]
        
        # 空间特征（如果提供）
        if spatial_input is not None:
            spatial_features = self.spatial_encoder(spatial_input.mean(dim=1))
        else:
            spatial_features = torch.zeros_like(temporal_features)
        
        # 跨模态注意力融合
        fused_features = self.cross_attention(
            image_features.unsqueeze(1),
            temporal_features.unsqueeze(1),
            spatial_features.unsqueeze(1)
        ).squeeze(1)
        
        # 多模态拼接
        combined = torch.cat([image_features, temporal_features, spatial_features], dim=1)
        
        # 分类和预警
        classification = self.fusion_classifier(combined)
        alert_level = self.alert_predictor(combined)
        
        return classification, temporal_prediction, alert_level


class CrossModalAttention(nn.Module):
    """
    跨模态注意力模块
    通过注意力机制学习不同模态之间的互补关系，
    动态调整各模态的贡献权重。
    """
    def __init__(self, d_model, nhead=4):
        super(CrossModalAttention, self).__init__()
        
        self.attention = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, query, key, value):
        """
        交叉注意力：query关注key和value中的相关信息
        """
        attn_output, attn_weights = self.attention(query, key, value)
        output = self.norm(query + self.dropout(attn_output))
        return output


class GNNLayer(nn.Module):
    """
    图神经网络层
    用于建模传感器网络中不同监测点之间的空间相关性，
    捕捉污染物扩散的空间传播模式。
    """
    def __init__(self, in_features, out_features):
        super(GNNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.attention = nn.Parameter(torch.ones(1))
    
    def forward(self, x, adj_matrix):
        """
        Args:
            x: 节点特征 [batch, num_nodes, in_features]
            adj_matrix: 邻接矩阵 [num_nodes, num_nodes]
        Returns:
            更新后的节点特征
        """
        # 消息传递
        support = self.linear(x)
        output = torch.bmm(adj_matrix.unsqueeze(0).expand(x.size(0), -1, -1), support)
        return F.relu(output)


def build_fusion_model(num_classes=6, **kwargs):
    """构建融合模型"""
    return MultiModalFusion(num_classes=num_classes, **kwargs)
