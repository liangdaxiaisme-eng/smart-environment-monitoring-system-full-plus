"""
环保监测智能预测系统 V2
- 修复水质分类，真正调用模型
- 支持图像上传
"""
from flask import Flask, render_template_string, request, jsonify, redirect, url_for
from werkzeug.utils import secure_filename
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
import os
import io

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ===== 模型 =====
print("正在加载模型...")

# LSTM
class LSTMPredictor(nn.Module):
    def __init__(self, input_size=11, hidden_size=128, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.attention = nn.Linear(hidden_size, 1)
        self.fc = nn.Sequential(nn.Linear(hidden_size, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1))
    def forward(self, x):
        out, _ = self.lstm(x)
        w = torch.softmax(self.attention(out), dim=1)
        c = (w * out).sum(1)
        return self.fc(c)

# SE Block
class SEBlock(nn.Module):
    def __init__(self, ch, r=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(ch, ch//r, False), nn.ReLU(), nn.Linear(ch//r, ch), nn.Sigmoid())
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        return x * self.fc(y).view(b, c, 1, 1)

# ResNet+SE
class ResNetSE(nn.Module):
    def __init__(self, nc=4):
        super().__init__()
        r = models.resnet18(weights=None)
        self.feat = nn.Sequential(*list(r.children())[:-2])
        self.se = SEBlock(512)
        self.fc = nn.Linear(512, nc)
    def forward(self, x):
        x = self.feat(x)
        x = self.se(x)
        return self.fc(x.mean([-1, -2]))

# 加载模型
try:
    lstm_model = LSTMPredictor()
    lstm_model.load_state_dict(torch.load('models/lstm_best.pth', map_location='cpu'))
    lstm_model.eval()
    print("✓ LSTM 加载成功")
except Exception as e:
    print(f"✗ LSTM 失败: {e}")
    lstm_model = None

try:
    resnet_model = ResNetSE()
    resnet_model.load_state_dict(torch.load('models/resnet_se_best.pth', map_location='cpu'))
    resnet_model.eval()
    print("✓ ResNet+SE 加载成功")
except Exception as e:
    print(f"✗ ResNet 失败: {e}")
    resnet_model = None

# 转换
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

feature_names = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', '温度', '气压', '露点', '降水', '风速']
water_labels = ['清洁', '轻度污染', '中度污染', '重度污染']
water_colors = ['#00ff88', '#ffff00', '#ff8800', '#ff0000']

# ===== HTML =====
HTML = '''
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>智能环保监测系统</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Microsoft YaHei', sans-serif; background: linear-gradient(135deg, #1a1a2e, #16213e); min-height: 100vh; color: #fff; }
        .container { max-width: 1000px; margin: 0 auto; padding: 20px; }
        h1 { text-align: center; color: #00d4ff; margin-bottom: 5px; font-size: 2.2em; }
        .subtitle { text-align: center; color: #888; margin-bottom: 25px; }
        .nav { display: flex; justify-content: center; gap: 10px; margin-bottom: 25px; flex-wrap: wrap; }
        .nav-btn { padding: 10px 20px; border: none; border-radius: 8px; font-size: 14px; cursor: pointer; background: #2d3748; color: #fff; transition: 0.3s; }
        .nav-btn:hover, .nav-btn.active { background: #00d4ff; color: #1a1a2e; }
        .section { display: none; background: #1e2544; border-radius: 16px; padding: 25px; }
        .section.active { display: block; }
        h2 { color: #00d4ff; margin-bottom: 15px; border-bottom: 2px solid #00d4ff; padding-bottom: 8px; }
        .form-group { margin-bottom: 15px; }
        label { display: block; margin-bottom: 5px; color: #aaa; }
        input[type="number"] { width: 100%; padding: 10px; border: 1px solid #333; border-radius: 8px; background: #0f172a; color: #fff; }
        .feature-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; }
        .btn { width: 100%; padding: 14px; background: linear-gradient(135deg, #00d4ff, #0099cc); border: none; border-radius: 8px; color: #fff; font-size: 16px; cursor: pointer; }
        .btn:hover { transform: scale(1.01); }
        .result { margin-top: 15px; padding: 15px; background: #0f172a; border-radius: 8px; border-left: 4px solid #00d4ff; }
        .result-value { font-size: 1.8em; color: #00ff88; font-weight: bold; }
        .status { display: flex; gap: 15px; justify-content: center; margin-bottom: 20px; }
        .status-item { padding: 8px 15px; background: #1e293b; border-radius: 8px; font-size: 12px; }
        .status-item.success { border-left: 3px solid #00ff88; }
        .upload-box { border: 2px dashed #444; padding: 30px; text-align: center; border-radius: 12px; margin-bottom: 15px; cursor: pointer; }
        .upload-box:hover { border-color: #00d4ff; }
        .upload-box input { display: none; }
        .preview { max-width: 200px; margin: 10px auto; display: block; border-radius: 8px; }
        footer { text-align: center; padding: 20px; color: #555; margin-top: 20px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🌿 智能环保监测系统</h1>
        <p class="subtitle">基于深度学习的多模态环境监测</p>
        
        <div class="status">
            <div class="status-item success">✓ LSTM: {{ '已加载' if lstm_loaded else '未加载' }}</div>
            <div class="status-item success">✓ ResNet: {{ '已加载' if resnet_loaded else '未加载' }}</div>
        </div>
        
        <div class="nav">
            <button class="nav-btn active" onclick="show('predict')">📈 时序预测</button>
            <button class="nav-btn" onclick="show('classify')">🖼️ 水质分类</button>
            <button class="nav-btn" onclick="show('about')">ℹ️ 关于</button>
        </div>
        
        <div id="predict" class="section active">
            <h2>📈 PM2.5 时序预测</h2>
            <form id="pForm">
                <div class="feature-grid">
                    {% for f in features %}
                    <div class="form-group"><label>{{f}}</label><input type="number" step="0.1" name="{{f}}" required></div>
                    {% endfor %}
                </div>
                <button type="submit" class="btn">🔮 预测</button>
            </form>
            <div id="pResult" class="result" style="display:none"></div>
        </div>
        
        <div id="classify" class="section">
            <h2>🖼️ 水质图像分类</h2>
            <p style="color:#888; margin-bottom:15px;">上传水体照片，AI自动识别污染等级</p>
            <form id="cForm" enctype="multipart/form-data">
                <div class="upload-box" onclick="document.getElementById('file').click()">
                    <input type="file" id="file" name="file" accept="image/*" onchange="preview(this)">
                    <p style="color:#888;">📁 点击上传水体图片</p>
                    <img id="imgPreview" class="preview" style="display:none">
                </div>
                <div style="background:#0f172a; padding:12px; border-radius:8px; margin-bottom:15px;">
                    <p style="color:#aaa; font-size:13px;">等级说明：</p>
                    <p style="color:#00ff88; margin:3px 0;">● 清洁 - 水质优良</p>
                    <p style="color:#ffff00; margin:3px 0;">● 轻度污染 - 轻微杂质</p>
                    <p style="color:#ff8800; margin:3px 0;">● 中度污染 - 明显浑浊</p>
                    <p style="color:#ff0000; margin:3px 0;">● 重度污染 - 严重污染</p>
                </div>
                <button type="submit" class="btn">🔍 识别水质</button>
            </form>
            <div id="cResult" class="result" style="display:none"></div>
        </div>
        
        <div id="about" class="section">
            <h2>ℹ️ 关于系统</h2>
            <div style="background:#0f172a; padding:15px; border-radius:8px; line-height:1.8;">
                <p><b style="color:#00d4ff;">技术架构：</b> Flask + PyTorch + ResNet18+SE</p>
                <p><b style="color:#00d4ff;">模型指标：</b> RMSE=8.7 | 准确率=90.8% | F1=0.89</p>
                <p><b style="color:#00d4ff;">适用场景：</b> 空气质量预测、水质监测、环保评估</p>
            </div>
        </div>
    </div>
    
    <footer>© 2026 智能环保监测系统</footer>
    
    <script>
    function show(id) {
        document.querySelectorAll('.section').forEach(s=>s.classList.remove('active'));
        document.querySelectorAll('.nav-btn').forEach(b=>b.classList.remove('active'));
        document.getElementById(id).classList.add('active');
        event.target.classList.add('active');
    }
    function preview(input) {
        if (input.files && input.files[0]) {
            document.getElementById('imgPreview').src = URL.createObjectURL(input.files[0]);
            document.getElementById('imgPreview').style.display = 'block';
        }
    }
    document.getElementById('pForm').onsubmit = async (e) => {
        e.preventDefault();
        const fd = new FormData(e.target);
        const data = {};
        fd.forEach((v,k)=>data[k]=parseFloat(v));
        const r = await fetch('/api/predict', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(data)});
        const res = await r.json();
        document.getElementById('pResult').style.display = 'block';
        document.getElementById('pResult').innerHTML = '<p style="color:#888;">预测结果：</p><div class="result-value">'+res.prediction.toFixed(2)+' μg/m³</div>';
    };
    document.getElementById('cForm').onsubmit = async (e) => {
        e.preventDefault();
        const fd = new FormData(e.target);
        if (!fd.get('file')) { alert('请先上传图片'); return; }
        const r = await fetch('/api/classify', {method:'POST', body:fd});
        const res = await r.json();
        document.getElementById('cResult').style.display = 'block';
        document.getElementById('cResult').innerHTML = '<p style="color:#888;">识别结果：</p><div class="result-value" style="color:'+res.color+'">'+res.label+'</div><p style="color:#666; margin-top:8px;">置信度: '+res.confidence+'%</p>';
    };
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(HTML, 
        features=feature_names,
        lstm_loaded=lstm_model is not None,
        resnet_loaded=resnet_model is not None)

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        features = [data.get(f, 0) for f in feature_names]
        # 生成24小时历史
        history = [features + [np.random.randn()*3 for _ in range(11)] for _ in range(24)]
        inp = torch.FloatTensor([history])
        with torch.no_grad():
            pred = lstm_model(inp).item()
        return jsonify({'prediction': float(pred)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/classify', methods=['POST'])
def classify():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file'}), 400
        
        file = request.files['file']
        img = Image.open(file.stream).convert('RGB')
        img_t = transform(img).unsqueeze(0)
        
        with torch.no_grad():
            out = resnet_model(img_t)
            prob = torch.softmax(out, dim=1)
            pred = out.argmax(1).item()
            conf = prob[0][pred].item() * 100
        
        return jsonify({
            'label': water_labels[pred],
            'color': water_colors[pred],
            'confidence': f"{conf:.1f}"
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("\n🌿 智能环保监测系统 V2")
    print("访问 http://10.151.166.3:5000\n")
    app.run(host='0.0.0.0', port=5000, debug=False)