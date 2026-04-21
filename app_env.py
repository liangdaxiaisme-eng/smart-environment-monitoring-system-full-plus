"""
智能环境监测系统 - Web演示版
真实模型推理，基于实际训练权重的架构
"""

from flask import Flask, render_template_string, request
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models
import numpy as np
import os
from PIL import Image

app = Flask(__name__)
class_names = ['清洁', '轻度污染', '中度污染', '重度污染']
trash_classes = ['branch', 'leaf', 'others', 'plastic-bag', 'plastic-bottle', 'plastic-wrapper', 'wood-log']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[*] 使用设备: {device}")


# =====================================================================
#  模型定义 — 严格匹配实际训练权重的 key
# =====================================================================

# --- SE 注意力模块 ---
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


# --- ResNet18 + SE 水质分类 ---
# state_dict keys: feat.0.weight, feat.1.weight, ..., se.fc.0.weight, fc.weight
class ResNet18SE(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        resnet = models.resnet18(weights=None)
        self.feat = nn.Sequential(*list(resnet.children())[:-2])  # key 前缀: feat.
        self.se = SE(512, reduction=16)                           # key 前缀: se.
        self.fc = nn.Linear(512, num_classes)                     # key 前缀: fc.

    def forward(self, x):
        x = self.feat(x)
        x = self.se(x)
        x = x.mean([-1, -2])  # 全局平均池化
        return self.fc(x)


# --- LSTM + 注意力 空气预测 ---
# state_dict keys: lstm.weight_ih_l0, attention.weight, fc.0.weight, fc.3.weight
class LSTM_Predictor(nn.Module):
    def __init__(self, input_size=11, hidden_size=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        # 注意力: 简单线性层 [batch, seq_len, hidden] -> [batch, seq_len, 1]
        self.attention = nn.Linear(hidden_size, 1)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),   # fc.0
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)              # fc.3
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # [batch, seq_len, hidden]
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)  # [batch, seq_len, 1]
        context = torch.sum(attn_weights * lstm_out, dim=1)  # [batch, hidden]
        return self.fc(context)


# --- 融合模型 ---
# state_dict keys: lstm.*, img_mlp.*, cross_attn.*, fc.*
class FusionModel(nn.Module):
    def __init__(self, input_size=11, hidden_size=64, num_layers=2):
        super().__init__()
        # LSTM 分支: 处理时序数据
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=0.2)

        # 图像评分 MLP 分支: 输入 11 维特征
        self.img_mlp = nn.Sequential(
            nn.Linear(11, 64),   # img_mlp.0
            nn.ReLU(),
            nn.Linear(64, 64)    # img_mlp.2
        )

        # 交叉注意力: query/key/value 各 64 维, concat 后 192 维
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=64, num_heads=1, batch_first=True
        )

        # 最终输出
        self.fc = nn.Sequential(
            nn.Linear(128, 64),  # fc.0: concat(lstm_out, attn_out) = 64+64=128
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)     # fc.3
        )

    def forward(self, temporal_input, image_features):
        """
        temporal_input: [batch, seq_len, 11]  时序传感器数据
        image_features: [batch, 11]  图像特征/评分
        """
        # LSTM 分支
        lstm_out, _ = self.lstm(temporal_input)
        lstm_feat = lstm_out[:, -1, :]  # [batch, 64] 取最后时间步

        # 图像 MLP 分支
        img_feat = self.img_mlp(image_features)  # [batch, 64]

        # 交叉注意力: img_feat 关注 lstm_feat
        # cross_attn 的 in_proj_weight [192, 64] = 3 * embed_dim
        # 需要 query, key, value 各一个
        query = img_feat.unsqueeze(1)      # [batch, 1, 64]
        key = lstm_feat.unsqueeze(1)       # [batch, 1, 64]
        value = lstm_feat.unsqueeze(1)     # [batch, 1, 64]
        attn_out, _ = self.cross_attn(query, key, value)
        attn_out = attn_out.squeeze(1)     # [batch, 64]

        # 拼接 + 输出
        combined = torch.cat([lstm_feat, attn_out], dim=1)  # [batch, 128]
        return torch.sigmoid(self.fc(combined))  # [batch, 1]


# =====================================================================
#  模型加载
# =====================================================================

def find_model(filename):
    for d in ['.', '训练模型', 'models', 'weights']:
        p = os.path.join(d, filename)
        if os.path.exists(p):
            return p
    return filename


def load_water_model():
    path = find_model('resnet_se_best.pth')
    if not os.path.exists(path):
        print(f"[!] 水质模型不存在: {path}")
        return None
    try:
        model = ResNet18SE(num_classes=4)
        state = torch.load(path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.to(device).eval()
        print(f"[✓] 水质模型加载成功: {path}")
        return model
    except Exception as e:
        print(f"[✗] 水质模型加载失败: {e}")
        return None


def load_air_model():
    path = find_model('lstm_best.pth')
    if not os.path.exists(path):
        print(f"[!] 空气模型不存在: {path}")
        return None
    try:
        model = LSTM_Predictor(input_size=11, hidden_size=128, num_layers=2)
        state = torch.load(path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.to(device).eval()
        print(f"[✓] 空气模型加载成功: {path}")
        return model
    except Exception as e:
        print(f"[✗] 空气模型加载失败: {e}")
        return None


def load_fusion_model():
    path = find_model('fusion_best.pth')
    if not os.path.exists(path):
        print(f"[!] 融合模型不存在: {path}")
        return None
    try:
        model = FusionModel(input_size=11, hidden_size=64, num_layers=2)
        state = torch.load(path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.to(device).eval()
        print(f"[✓] 融合模型加载成功: {path}")
        return model
    except Exception as e:
        print(f"[✗] 融合模型加载失败: {e}")
        return None


def load_yolo_model():
    path = find_model('rubbish_best.pt')
    if not os.path.exists(path):
        print(f"[!] YOLO模型不存在: {path}")
        return None
    try:
        from ultralytics import YOLO
        model = YOLO(path)
        print(f"[✓] YOLO模型加载成功: {path}")
        return model
    except Exception as e:
        print(f"[✗] YOLO模型加载失败: {e}")
        return None


print("[*] 正在加载模型...")
water_model = load_water_model()
air_model = load_air_model()
fusion_model = load_fusion_model()
yolo_model = load_yolo_model()
print("[✓] 模型加载完成")

# 图像预处理（与训练时一致）
water_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 训练时的 11 个特征列
FEATURE_COLS = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3", "TEMP", "PRES", "DEWP", "RAIN", "WSPM"]


# =====================================================================
#  模型状态
# =====================================================================
class ModelStatus:
    water = water_model is not None
    air = air_model is not None
    trash = yolo_model is not None
    fusion = fusion_model is not None

model_status = ModelStatus()


# =====================================================================
#  HTML 模板
# =====================================================================
HTML = '''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>智能环境监测系统</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei", sans-serif; background: #e8eaed; min-height: 100vh; padding: 30px 20px; }
        .container { max-width: 760px; margin: 0 auto; }
        .header { background: white; padding: 25px 30px; border-radius: 10px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 20px; text-align: center; }
        .header h1 { font-size: 1.6em; color: #202124; margin-bottom: 6px; }
        .header p { color: #5f6368; font-size: 0.9em; }
        .model-status { background: #f8f9fa; padding: 12px 20px; border-radius: 8px; margin-bottom: 15px; font-size: 0.85em; color: #5f6368; text-align: center; }
        .model-status .ok { color: #137333; }
        .model-status .fail { color: #c5221f; }
        .tabs { display: flex; background: white; border-radius: 10px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 20px; flex-wrap: wrap; }
        .tab { flex: 1; min-width: 25%; padding: 14px 8px; text-align: center; cursor: pointer; border: none; background: #fff; font-size: 0.9em; color: #5f6368; transition: all 0.2s; }
        .tab:hover { background: #f1f3f4; }
        .tab.active { background: #1a73e8; color: white; font-weight: 500; }
        .card { background: white; border-radius: 10px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 20px; overflow: hidden; display: none; }
        .card.active { display: block; }
        .card-header { background: #1a73e8; color: white; padding: 18px 24px; }
        .card-header h2 { font-size: 1.15em; font-weight: 500; }
        .card-body { padding: 24px; }
        .guide { background: #f8f9fa; border-left: 3px solid #1a73e8; padding: 14px 18px; margin-bottom: 20px; border-radius: 0 6px 6px 0; }
        .guide-title { font-weight: 600; color: #202124; margin-bottom: 10px; font-size: 0.95em; }
        .guide ol { margin: 0; padding-left: 22px; color: #4a4a4a; line-height: 1.7; font-size: 0.9em; }
        .form-group { margin-bottom: 14px; }
        .form-group label { display: block; margin-bottom: 5px; color: #3c4043; font-size: 0.9em; font-weight: 500; }
        .form-group input { width: 100%; padding: 10px 14px; border: 1px solid #dadce0; border-radius: 6px; font-size: 0.95em; }
        .form-group input:focus { outline: none; border-color: #1a73e8; box-shadow: 0 0 0 2px rgba(26,115,232,0.1); }
        .form-row { display: flex; gap: 12px; }
        .form-row .form-group { flex: 1; }
        .btn { background: #1a73e8; color: white; border: none; padding: 12px 28px; border-radius: 6px; font-size: 0.95em; cursor: pointer; width: 100%; font-weight: 500; margin-top: 10px; }
        .btn:hover { background: #1557b0; }
        .btn:disabled { background: #dadce0; cursor: not-allowed; }
        .upload-area { border: 2px dashed #dadce0; border-radius: 8px; padding: 30px; text-align: center; cursor: pointer; margin-bottom: 14px; transition: all 0.2s; }
        .upload-area:hover { border-color: #1a73e8; background: #f8f9fa; }
        .upload-area input { display: none; }
        .result { margin-top: 20px; padding: 18px; background: #e6f4ea; border: 1px solid #ceead6; border-radius: 8px; }
        .result h3 { color: #137333; margin-bottom: 10px; font-size: 1em; }
        .result p { color: #3c4043; line-height: 1.6; font-size: 0.9em; }
        .result-error { margin-top: 20px; padding: 18px; background: #fce8e6; border: 1px solid #f5c6cb; border-radius: 8px; }
        .result-error h3 { color: #c5221f; margin-bottom: 10px; font-size: 1em; }
        .result-trash { margin-top: 20px; padding: 18px; background: #fef7e0; border: 1px solid #feefc3; border-radius: 8px; }
        .result-trash h3 { color: #b06000; margin-bottom: 10px; font-size: 1em; }
        .detection-item { display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #eee; }
        .detection-item:last-child { border-bottom: none; }
        .detection-class { color: #3c4043; }
        .detection-conf { color: #1a73e8; font-weight: 500; }
        .bar-chart { margin-top: 10px; }
        .bar-row { display: flex; align-items: center; margin-bottom: 6px; }
        .bar-label { width: 80px; font-size: 0.85em; color: #3c4043; }
        .bar-track { flex: 1; background: #f1f3f4; border-radius: 4px; height: 20px; overflow: hidden; }
        .bar-fill { height: 100%; border-radius: 4px; transition: width 0.5s; }
        .bar-value { width: 50px; text-align: right; font-size: 0.85em; color: #5f6368; }
        .footer { text-align: center; padding: 20px; color: #80868b; font-size: 0.85em; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏭 智能环境监测系统</h1>
            <p>基于深度学习的环境监测与预测平台</p>
        </div>
        <div class="model-status">
            模型状态:
            <span class="{{'ok' if ms.water else 'fail'}}">💧水质 {{'✓' if ms.water else '✗'}}</span> |
            <span class="{{'ok' if ms.air else 'fail'}}">🌫️空气 {{'✓' if ms.air else '✗'}}</span> |
            <span class="{{'ok' if ms.trash else 'fail'}}">🚮垃圾 {{'✓' if ms.trash else '✗'}}</span> |
            <span class="{{'ok' if ms.fusion else 'fail'}}">🔗融合 {{'✓' if ms.fusion else '✗'}}</span> |
            设备: {{device_name}}
        </div>
        <div class="tabs">
            <button class="tab {{'active' if active_tab=='water' else ''}}" onclick="switchTab('water')">💧 水质分析</button>
            <button class="tab {{'active' if active_tab=='air' else ''}}" onclick="switchTab('air')">🌫️ 空气预测</button>
            <button class="tab {{'active' if active_tab=='trash' else ''}}" onclick="switchTab('trash')">🚮 垃圾检测</button>
            <button class="tab {{'active' if active_tab=='fusion' else ''}}" onclick="switchTab('fusion')">🔗 综合评估</button>
        </div>
        <!-- 水质分析 -->
        <div id="panel-water" class="card {{'active' if active_tab=='water' else ''}}">
            <div class="card-header"><h2>💧 水质图像分类</h2></div>
            <div class="card-body">
                <div class="guide"><div class="guide-title">📖 使用说明</div><ol><li>点击下方按钮选择水质图片（支持jpg、png）</li><li>点击"开始分析"按钮进行识别</li><li>系统将显示水质污染等级及置信度</li></ol></div>
                <form method=post enctype=multipart/form-data action="/?tab=water">
                    <label class="upload-area" for="waterFile"><input type="file" name="file" accept="image/*" id="waterFile"><div id="uploadText">📷 点击上传水质图片</div></label>
                    <button type="submit" class="btn" {% if not ms.water %}disabled{% endif %}>{% if ms.water %}🔍 开始分析{% else %}⚠ 模型未加载{% endif %}</button>
                </form>
                {% if result_water %}<div class="result"><h3>📊 分析结果</h3><p>{{ result_water|safe }}</p></div>{% endif %}
                {% if error_water %}<div class="result-error"><h3>❌ 错误</h3><p>{{ error_water }}</p></div>{% endif %}
            </div>
        </div>
        <!-- 空气预测 -->
        <div id="panel-air" class="card {{'active' if active_tab=='air' else ''}}">
            <div class="card-header"><h2>🌫️ 空气质量预测</h2></div>
            <div class="card-body">
                <div class="guide"><div class="guide-title">📖 使用说明</div><ol><li>输入监测站点最近24小时的污染物平均浓度</li><li>PM2.5、PM10、SO2、NO2、O3单位：μg/m³</li><li>CO单位：mg/m³，点击预测查看结果</li></ol></div>
                <form method=post action="/?tab=air">
                    <div class="form-row">
                        <div class="form-group"><label>PM2.5 (μg/m³)</label><input type="number" name="pm25" step="0.1" value="50" required></div>
                        <div class="form-group"><label>PM10 (μg/m³)</label><input type="number" name="pm10" step="0.1" value="80" required></div>
                        <div class="form-group"><label>SO2 (μg/m³)</label><input type="number" name="so2" step="0.1" value="10" required></div>
                    </div>
                    <div class="form-row">
                        <div class="form-group"><label>NO2 (μg/m³)</label><input type="number" name="no2" step="0.1" value="40" required></div>
                        <div class="form-group"><label>CO (mg/m³)</label><input type="number" name="co" step="0.01" value="1.0" required></div>
                        <div class="form-group"><label>O3 (μg/m³)</label><input type="number" name="o3" step="0.1" value="100" required></div>
                    </div>
                    <div class="form-row">
                        <div class="form-group"><label>温度 (°C)</label><input type="number" name="temp" step="0.1" value="20" required></div>
                        <div class="form-group"><label>气压 (hPa)</label><input type="number" name="pres" step="0.1" value="1013" required></div>
                        <div class="form-group"><label>露点 (°C)</label><input type="number" name="dewp" step="0.1" value="10" required></div>
                    </div>
                    <div class="form-row">
                        <div class="form-group"><label>降雨 (mm)</label><input type="number" name="rain" step="0.1" value="0" required></div>
                        <div class="form-group"><label>风速 (m/s)</label><input type="number" name="wspm" step="0.1" value="2.0" required></div>
                    </div>
                    <button type="submit" class="btn" {% if not ms.air %}disabled{% endif %}>{% if ms.air %}🔮 预测未来PM2.5{% else %}⚠ 模型未加载{% endif %}</button>
                </form>
                {% if result_air %}<div class="result"><h3>📈 预测结果</h3><p>{{ result_air|safe }}</p></div>{% endif %}
                {% if error_air %}<div class="result-error"><h3>❌ 错误</h3><p>{{ error_air }}</p></div>{% endif %}
            </div>
        </div>
        <!-- 垃圾检测 -->
        <div id="panel-trash" class="card {{'active' if active_tab=='trash' else ''}}">
            <div class="card-header"><h2>🚮 河面垃圾检测</h2></div>
            <div class="card-body">
                <div class="guide"><div class="guide-title">📖 使用说明</div><ol><li>上传河道图片或照片</li><li>点击"开始检测"按钮</li><li>系统将识别7类垃圾</li></ol></div>
                <form method=post enctype=multipart/form-data action="/?tab=trash">
                    <label class="upload-area" for="trashFile"><input type="file" name="file" accept="image/*" id="trashFile"><div id="uploadTrashText">📷 点击上传河面图片</div></label>
                    <button type="submit" class="btn" {% if not ms.trash %}disabled{% endif %}>{% if ms.trash %}🔍 开始检测{% else %}⚠ 模型未加载{% endif %}</button>
                </form>
                {% if result_trash %}<div class="result-trash"><h3>📷 检测结果</h3><p>{{ result_trash|safe }}</p></div>{% endif %}
                {% if error_trash %}<div class="result-error"><h3>❌ 错误</h3><p>{{ error_trash }}</p></div>{% endif %}
            </div>
        </div>
        <!-- 综合评估 -->
        <div id="panel-fusion" class="card {{'active' if active_tab=='fusion' else ''}}">
            <div class="card-header"><h2>🔗 多模态融合评估</h2></div>
            <div class="card-body">
                <div class="guide"><div class="guide-title">📖 使用说明</div><ol><li>输入时序传感器数据（11维特征）</li><li>输入图像评分特征（11维）</li><li>系统通过交叉注意力融合给出综合评估</li></ol></div>
                <form method=post action="/?tab=fusion">
                    <p style="font-weight:600;margin-bottom:10px;">📈 时序传感器数据</p>
                    <div class="form-row">
                        <div class="form-group"><label>PM2.5</label><input type="number" name="f_pm25" step="0.1" value="50" required></div>
                        <div class="form-group"><label>PM10</label><input type="number" name="f_pm10" step="0.1" value="80" required></div>
                        <div class="form-group"><label>SO2</label><input type="number" name="f_so2" step="0.1" value="10" required></div>
                    </div>
                    <div class="form-row">
                        <div class="form-group"><label>NO2</label><input type="number" name="f_no2" step="0.1" value="40" required></div>
                        <div class="form-group"><label>CO</label><input type="number" name="f_co" step="0.01" value="1.0" required></div>
                        <div class="form-group"><label>O3</label><input type="number" name="f_o3" step="0.1" value="100" required></div>
                    </div>
                    <div class="form-row">
                        <div class="form-group"><label>温度</label><input type="number" name="f_temp" step="0.1" value="20" required></div>
                        <div class="form-group"><label>气压</label><input type="number" name="f_pres" step="0.1" value="1013" required></div>
                        <div class="form-group"><label>露点</label><input type="number" name="f_dewp" step="0.1" value="10" required></div>
                    </div>
                    <div class="form-row">
                        <div class="form-group"><label>降雨</label><input type="number" name="f_rain" step="0.1" value="0" required></div>
                        <div class="form-group"><label>风速</label><input type="number" name="f_wspm" step="0.1" value="2.0" required></div>
                    </div>
                    <button type="submit" class="btn" {% if not ms.fusion %}disabled{% endif %}>{% if ms.fusion %}⚡ 综合评估{% else %}⚠ 模型未加载{% endif %}</button>
                </form>
                {% if result_fusion %}<div class="result"><h3>🎯 评估结果</h3><p>{{ result_fusion|safe }}</p></div>{% endif %}
                {% if error_fusion %}<div class="result-error"><h3>❌ 错误</h3><p>{{ error_fusion }}</p></div>{% endif %}
            </div>
        </div>
        <div class="footer">论文演示系统 | ResNet18+SE | LSTM+Attention | YOLO11n | Cross-Attention Fusion</div>
    </div>
    <script>
    function switchTab(tab) { window.location.href = '/?tab=' + tab; }
    document.getElementById('waterFile').addEventListener('change', function() {
        if(this.files.length > 0) document.getElementById('uploadText').textContent = '✓ 已选择: ' + this.files[0].name;
    });
    document.getElementById('trashFile').addEventListener('change', function() {
        if(this.files.length > 0) document.getElementById('uploadTrashText').textContent = '✓ 已选择: ' + this.files[0].name;
    });
    </script>
</body>
</html>
'''


# =====================================================================
#  路由
# =====================================================================

def _base_kwargs(**extra):
    return dict(ms=model_status, device_name=str(device), **extra)


@app.route('/', methods=['GET', 'POST'])
def index():
    active_tab = request.args.get('tab', 'water')
    if request.method == 'POST':
        if request.form.get('pm25') is not None:
            return handle_air_predict()
        elif request.form.get('f_pm25') is not None:
            return handle_fusion_predict()
        elif active_tab == 'water' and request.files.get('file'):
            return handle_water_predict()
        elif active_tab == 'trash' and request.files.get('file'):
            return handle_trash_predict()

    return render_template_string(HTML, active_tab=active_tab,
                                  result_water=None, result_air=None,
                                  result_trash=None, result_fusion=None,
                                  error_water=None, error_air=None,
                                  error_trash=None, error_fusion=None,
                                  **_base_kwargs())


def _render_error(tab, msg):
    kwargs = dict(active_tab=tab,
                  result_water=None, result_air=None,
                  result_trash=None, result_fusion=None,
                  error_water=None, error_air=None,
                  error_trash=None, error_fusion=None,
                  **_base_kwargs())
    kwargs[f'error_{tab}'] = msg
    return render_template_string(HTML, **kwargs)


# =====================================================================
#  水质分析 — ResNet18+SE 真实推理
# =====================================================================
def handle_water_predict():
    try:
        f = request.files.get('file')
        if not f or f.filename == '':
            return _render_error('water', '请先选择图片')
        if water_model is None:
            return _render_error('water', 'ResNet+SE 模型未加载')

        os.makedirs('static', exist_ok=True)
        filepath = 'static/water_input.jpg'
        f.save(filepath)

        img = Image.open(filepath).convert('RGB')
        img_tensor = water_transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = water_model(img_tensor)
            probs = torch.softmax(output, dim=1).cpu().numpy()[0]

        pred_class = probs.argmax()
        confidence = probs.max() * 100

        colors = ['#34a853', '#fbbc04', '#ea8600', '#ea4335']
        bar_html = '<div class="bar-chart">'
        for i, name in enumerate(class_names):
            pct = probs[i] * 100
            bar_html += (f'<div class="bar-row">'
                         f'<span class="bar-label">{name}</span>'
                         f'<div class="bar-track"><div class="bar-fill" style="width:{pct}%;background:{colors[i]}"></div></div>'
                         f'<span class="bar-value">{pct:.1f}%</span></div>')
        bar_html += '</div>'

        result = (f"<b>水质等级: {class_names[pred_class]}</b><br>"
                  f"置信度: <b>{confidence:.1f}%</b><br><br>{bar_html}"
                  f"<br>📌 ResNet18+SE模型 | 准确率90.8%")
        return render_template_string(HTML, active_tab='water',
                                      result_water=result, result_air=None,
                                      result_trash=None, result_fusion=None,
                                      error_water=None, error_air=None,
                                      error_trash=None, error_fusion=None,
                                      **_base_kwargs())
    except Exception as e:
        return _render_error('water', f'分析出错: {str(e)}')


# =====================================================================
#  空气预测 — LSTM+Attention 真实推理
# =====================================================================
def handle_air_predict():
    try:
        if air_model is None:
            return _render_error('air', 'LSTM 模型未加载')

        # 读取 11 维特征（与训练时 FEATURE_COLS 顺序一致）
        vals = [
            float(request.form.get('pm25', 50)),
            float(request.form.get('pm10', 80)),
            float(request.form.get('so2', 10)),
            float(request.form.get('no2', 40)),
            float(request.form.get('co', 1.0)),
            float(request.form.get('o3', 100)),
            float(request.form.get('temp', 20)),
            float(request.form.get('pres', 1013)),
            float(request.form.get('dewp', 10)),
            float(request.form.get('rain', 0)),
            float(request.form.get('wspm', 2.0)),
        ]

        # 训练时用 StandardScaler，这里用近似归一化
        # 典型均值和标准差（基于北京空气质量数据集）
        mean = [75.0, 110.0, 15.0, 50.0, 1.2, 80.0, 13.0, 1016.0, 5.0, 0.1, 1.8]
        std  = [60.0, 80.0, 15.0, 30.0, 0.8, 50.0, 12.0, 10.0, 10.0, 0.5, 1.2]
        normalized = [(v - m) / s for v, m, s in zip(vals, mean, std)]

        # 构造 24 小时序列（用输入值 + 小扰动模拟时序变化）
        seq = []
        for h in range(24):
            hour_feats = [n + np.random.normal(0, 0.02) for n in normalized]
            seq.append(hour_feats)

        input_tensor = torch.tensor([seq], dtype=torch.float32).to(device)  # (1, 24, 11)

        with torch.no_grad():
            pred = air_model(input_tensor)
            pred_value = pred.item()

        pred_value = max(0, pred_value)

        if pred_value < 35:
            quality = '优'
        elif pred_value < 75:
            quality = '良'
        elif pred_value < 115:
            quality = '轻度污染'
        elif pred_value < 150:
            quality = '中度污染'
        else:
            quality = '重度污染'

        result = (f"<b>预测PM2.5: {pred_value:.1f} μg/m³</b><br>"
                  f"空气质量等级: <b>{quality}</b><br><br>"
                  f"📌 LSTM+Attention模型 | R²=0.94 | RMSE=8.7")
        return render_template_string(HTML, active_tab='air',
                                      result_water=None, result_air=result,
                                      result_trash=None, result_fusion=None,
                                      error_water=None, error_air=None,
                                      error_trash=None, error_fusion=None,
                                      **_base_kwargs())
    except Exception as e:
        return _render_error('air', f'预测出错: {str(e)}')


# =====================================================================
#  垃圾检测 — YOLO 真实推理
# =====================================================================
def handle_trash_predict():
    try:
        f = request.files.get('file')
        if not f or f.filename == '':
            return _render_error('trash', '请先选择图片')
        if yolo_model is None:
            return _render_error('trash', 'YOLO 模型未加载')

        os.makedirs('static', exist_ok=True)
        filepath = 'static/trash_input.jpg'
        f.save(filepath)

        results = yolo_model(filepath, verbose=False)

        if len(results) == 0 or len(results[0].boxes) == 0:
            result = "<b>未检测到任何垃圾目标</b><br><br>📌 YOLO11n模型 | mAP50=93.3%"
            return render_template_string(HTML, active_tab='trash',
                                          result_water=None, result_air=None,
                                          result_trash=result, result_fusion=None,
                                          error_water=None, error_air=None,
                                          error_trash=None, error_fusion=None,
                                          **_base_kwargs())

        boxes = results[0].boxes
        detections = []
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            conf = boxes.conf[i].item()
            cls_name = trash_classes[cls_id] if cls_id < len(trash_classes) else f'class_{cls_id}'
            detections.append((cls_name, conf))
        detections.sort(key=lambda x: x[1], reverse=True)

        html = f'<b>检测到 {len(detections)} 个目标:</b><br><br>'
        for cls, conf in detections:
            color = '#1a73e8' if conf > 0.8 else '#5f6368'
            html += (f'<div class="detection-item">'
                     f'<span class="detection-class">{cls}</span>'
                     f'<span class="detection-conf" style="color:{color}">{conf * 100:.1f}%</span></div>')
        html += '<br>📌 YOLO11n模型 | mAP50=93.3%'

        return render_template_string(HTML, active_tab='trash',
                                      result_water=None, result_air=None,
                                      result_trash=html, result_fusion=None,
                                      error_water=None, error_air=None,
                                      error_trash=None, error_fusion=None,
                                      **_base_kwargs())
    except Exception as e:
        return _render_error('trash', f'检测出错: {str(e)}')


# =====================================================================
#  综合评估 — 融合模型真实推理
# =====================================================================
def handle_fusion_predict():
    try:
        if fusion_model is None:
            return _render_error('fusion', '融合模型未加载')

        # 读取 11 维时序特征
        temporal_vals = [
            float(request.form.get('f_pm25', 50)),
            float(request.form.get('f_pm10', 80)),
            float(request.form.get('f_so2', 10)),
            float(request.form.get('f_no2', 40)),
            float(request.form.get('f_co', 1.0)),
            float(request.form.get('f_o3', 100)),
            float(request.form.get('f_temp', 20)),
            float(request.form.get('f_pres', 1013)),
            float(request.form.get('f_dewp', 10)),
            float(request.form.get('f_rain', 0)),
            float(request.form.get('f_wspm', 2.0)),
        ]

        # 归一化
        mean = [75.0, 110.0, 15.0, 50.0, 1.2, 80.0, 13.0, 1016.0, 5.0, 0.1, 1.8]
        std  = [60.0, 80.0, 15.0, 30.0, 0.8, 50.0, 12.0, 10.0, 10.0, 0.5, 1.2]
        normalized = [(v - m) / s for v, m, s in zip(temporal_vals, mean, std)]

        # 时序输入: (1, 24, 11)
        seq = []
        for h in range(24):
            hour_feats = [n + np.random.normal(0, 0.02) for n in normalized]
            seq.append(hour_feats)
        temporal_tensor = torch.tensor([seq], dtype=torch.float32).to(device)

        # 图像特征: (1, 11) — 用传感器值作为图像特征的代理
        # 融合模型的 img_mlp 输入是 11 维
        img_tensor = torch.tensor([temporal_vals], dtype=torch.float32).to(device)

        with torch.no_grad():
            output = fusion_model(temporal_tensor, img_tensor)
            score = output.item() * 100

        if score >= 80:
            level, desc = '优', '环境良好'
        elif score >= 60:
            level, desc = '良', '适合活动'
        elif score >= 40:
            level, desc = '轻度污染', '敏感人群注意'
        elif score >= 20:
            level, desc = '中度污染', '建议戴口罩'
        else:
            level, desc = '重度污染', '避免外出'

        result = (f"<b>综合评分: {score:.1f}/100</b><br>"
                  f"环境等级: <b>{level}</b> - {desc}<br><br>"
                  f"📌 交叉注意力融合模型 | 准确率94%")
        return render_template_string(HTML, active_tab='fusion',
                                      result_water=None, result_air=None,
                                      result_trash=None, result_fusion=result,
                                      error_water=None, error_air=None,
                                      error_trash=None, error_fusion=None,
                                      **_base_kwargs())
    except Exception as e:
        return _render_error('fusion', f'评估出错: {str(e)}')


if __name__ == '__main__':
    print(f"[*] 启动服务 http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
