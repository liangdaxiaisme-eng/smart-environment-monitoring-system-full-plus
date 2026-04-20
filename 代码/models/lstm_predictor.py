"""
基于深度学习的智能环保监测算法研究
LSTM时序预测模型 - 空气/水质质量预测
黄丽佳 22460525

基于LSTM + 注意力机制的时序预测模型，
用于预测PM2.5、PM10、SO₂、NO₂等污染物浓度。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TemporalAttention(nn.Module):
    """
    时序注意力模块
    自动学习不同时间步的重要性权重，
    关注对预测结果影响最大的历史时刻。
    """
    def __init__(self, hidden_size):
        super(TemporalAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )
    
    def forward(self, lstm_output):
        """
        Args:
            lstm_output: LSTM输出 [batch, seq_len, hidden_size*2]
        Returns:
            context: 加权上下文向量 [batch, hidden_size*2]
            weights: 注意力权重 [batch, seq_len]
        """
        weights = self.attention(lstm_output).squeeze(-1)  # [batch, seq_len]
        weights = F.softmax(weights, dim=1)
        
        context = torch.bmm(weights.unsqueeze(1), lstm_output).squeeze(1)
        return context, weights


class LSTMPredictor(nn.Module):
    """
    LSTM时序预测模型
    多层LSTM + 时序注意力 + 全连接输出，
    用于预测未来24h/48h/72h的污染物浓度。
    
    输入特征：
    - PM2.5, PM10, SO₂, NO₂, O₃, CO 浓度
    - 温度、湿度、风速、气压 气象数据
    - 时间特征（小时、星期、月份）
    
    输出：
    - 未来N小时各污染物浓度预测值
    """
    def __init__(self, input_size=13, hidden_size=128, num_layers=2, 
                 output_size=6, forecast_horizon=24, dropout=0.3):
        super(LSTMPredictor, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.forecast_horizon = forecast_horizon
        
        # 输入特征嵌入（将不同量纲的特征映射到统一空间）
        self.input_embedding = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # 双向LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 时序注意力
        self.attention = TemporalAttention(hidden_size)
        
        # 预测头
        self.predictor = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size // 2, output_size * forecast_horizon)
        )
        
        self.output_size = output_size
    
    def forward(self, x):
        """
        Args:
            x: 输入序列 [batch, seq_len, input_size]
        Returns:
            predictions: 预测结果 [batch, forecast_horizon, output_size]
        """
        batch_size = x.size(0)
        
        # 特征嵌入
        x = self.input_embedding(x)
        
        # LSTM编码
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # 时序注意力
        context, attn_weights = self.attention(lstm_out)
        
        # 预测
        output = self.predictor(context)
        output = output.view(batch_size, self.forecast_horizon, self.output_size)
        
        return output, attn_weights


class TransformerPredictor(nn.Module):
    """
    Transformer时序预测模型
    基于自注意力机制，捕捉长期依赖关系，
    用于中长期（7天/30天）环境质量趋势预测。
    """
    def __init__(self, input_size=13, d_model=128, nhead=8, 
                 num_layers=4, output_size=6, forecast_horizon=24, dropout=0.1):
        super(TransformerPredictor, self).__init__()
        
        self.d_model = d_model
        self.forecast_horizon = forecast_horizon
        self.output_size = output_size
        
        # 输入嵌入
        self.input_embedding = nn.Linear(input_size, d_model)
        
        # 位置编码
        self.pos_encoding = self._generate_positional_encoding(512, d_model)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # 预测头
        self.predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_size * forecast_horizon)
        )
    
    def _generate_positional_encoding(self, max_len, d_model):
        """生成正弦位置编码"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)
    
    def forward(self, x):
        """
        Args:
            x: 输入序列 [batch, seq_len, input_size]
        Returns:
            predictions: [batch, forecast_horizon, output_size]
        """
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        # 嵌入 + 位置编码
        x = self.input_embedding(x) * np.sqrt(self.d_model)
        x = x + self.pos_encoding[:, :seq_len, :]
        
        # Transformer编码
        encoded = self.transformer_encoder(x)
        
        # 取最后一个时间步
        last_step = encoded[:, -1, :]
        
        # 预测
        output = self.predictor(last_step)
        output = output.view(batch_size, self.forecast_horizon, self.output_size)
        
        return output


class TCNPredictor(nn.Module):
    """
    时序卷积网络(TCN)预测模型
    使用因果膨胀卷积捕捉时序模式，
    计算效率高于LSTM，适合实时预测场景。
    """
    def __init__(self, input_size=13, num_channels=[64, 128, 256], 
                 kernel_size=3, output_size=6, forecast_horizon=24, dropout=0.2):
        super(TCNPredictor, self).__init__()
        
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            padding = (kernel_size - 1) * dilation
            
            layers.append(nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size,
                         dilation=dilation, padding=padding),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Conv1d(out_channels, out_channels, kernel_size,
                         dilation=dilation, padding=padding),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ))
        
        self.network = nn.ModuleList(layers)
        
        # 预测头
        self.predictor = nn.Sequential(
            nn.Linear(num_channels[-1], 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, output_size * forecast_horizon)
        )
        
        self.output_size = output_size
        self.forecast_horizon = forecast_horizon
    
    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, input_size]
        Returns:
            predictions: [batch, forecast_horizon, output_size]
        """
        batch_size = x.size(0)
        
        # 转换为 [batch, channels, seq_len]
        x = x.transpose(1, 2)
        
        # TCN层
        for layer in self.network:
            residual = x
            x = layer(x)
            x = x[:, :, :residual.size(2)]  # 因果卷积裁剪
            if x.size(1) == residual.size(1):
                x = x + residual  # 残差连接
        
        # 取最后一个时间步
        x = x[:, :, -1]
        
        # 预测
        output = self.predictor(x)
        output = output.view(batch_size, self.forecast_horizon, self.output_size)
        
        return output


def build_predictor(model_type='lstm', **kwargs):
    """预测模型工厂函数"""
    if model_type == 'lstm':
        return LSTMPredictor(**kwargs)
    elif model_type == 'transformer':
        return TransformerPredictor(**kwargs)
    elif model_type == 'tcn':
        return TCNPredictor(**kwargs)
    else:
        raise ValueError(f"Unknown predictor type: {model_type}")
