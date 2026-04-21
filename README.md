# 🏭 智能环境监测系统

基于深度学习的环境监测与预测平台 | 论文演示系统

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)
![Flask](https://img.shields.io/badge/Flask-3.0+-green)

## 📋 项目简介

本系统是一套完整的**智能环境监测解决方案**，包含四个核心功能模块：

| 模块 | 功能 | 模型 | 性能指标 |
|------|------|------|----------|
| 💧 水质分析 | 图像分类识别污染等级 | ResNet+SE | 准确率 90.8% |
| 🌫️ 空气预测 | PM2.5时序预测 | LSTM | R²=0.94, RMSE=8.7 |
| 🚮 垃圾检测 | 河面目标检测 | YOLO11n | mAP50=93.3% |
| 🔗 综合评估 | 多模态融合评估 | 融合模型 | 准确率 94% |

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                      Web界面层 (Flask)                       │
├─────────────┬─────────────┬─────────────┬──────────────────┤
│  水质分析   │  空气预测   │  垃圾检测   │    综合评估      │
│   (ResNet)  │   (LSTM)    │   (YOLO)    │    (Fusion)      │
├─────────────┴─────────────┴─────────────┴──────────────────┤
│                      模型层                                  │
├─────────────────────────────────────────────────────────────┤
│                      数据层                                  │
└─────────────────────────────────────────────────────────────┘
```

## 📁 项目结构

```
智能环境监测系统/
├── app_env.py                 # Web演示系统主程序
├── 训练模型/
│   ├── lstm_best.pth         # LSTM时序预测模型
│   ├── resnet_se_best.pth    # ResNet+SE图像分类模型  
│   ├── fusion_best.pth       # 多模态融合模型
│   └── gnn_best.pth          # GNN图神经网络模型
├── 训练数据/
│   ├── lstm_training.png     # LSTM训练曲线
│   ├── resnet_training.png   # ResNet训练曲线
│   ├── model_comparison.png  # 模型对比图
│   └── prediction_demo.png   # 预测示例图
├── rubbish_best.pt           # YOLO河面垃圾检测模型
├── rubbish_train_results.zip # 完整训练结果
├── 使用说明.md               # 部署使用文档
├── 论文辅助资料.md           # 论文写作参考
└── README.md                 # 本文件
```

## 🚀 快速部署

### 环境要求
- Python 3.8+
- PyTorch 2.0+
- Flask 3.0+
- 4GB+ RAM
- 推荐 GPU (NVIDIA)

### 安装依赖
```bash
pip install torch torchvision flask numpy
```

### 运行Web服务
```bash
# 1. 克隆项目
git clone https://github.com/你的用户名/智能环境监测系统.git
cd 智能环境监测系统

# 2. 确保模型文件在同一目录
# lstm_best.pth, resnet_se_best.pth, fusion_best.pth, rubbish_best.pt

# 3. 启动服务
python app_env.py

# 4. 访问
# http://localhost:5000
```

### Docker部署（推荐）
```bash
# 构建镜像
docker build -t env-monitor .

# 运行容器
docker run -d -p 5000:5000 --gpus all env-monitor

# 访问 http://localhost:5000
```

Dockerfile示例：
```dockerfile
FROM python:3.9-slim

RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install flask numpy

WORKDIR /app
COPY . /app

EXPOSE 5000
CMD ["python", "app_env.py"]
```

## 📖 使用说明

### 💧 水质图像分类
1. 进入系统，点击「水质分析」标签
2. 上传水质图片（支持jpg、png格式）
3. 点击「开始分析」
4. 查看4个等级的识别结果：
   - 清洁
   - 轻度污染
   - 中度污染
   - 重度污染

### 🌫️ 空气质量预测
1. 进入系统，点击「空气预测」标签
2. 输入最近24小时的污染物浓度：
   - PM2.5 (μg/m³)
   - PM10 (μg/m³)
   - SO2 (μg/m³)
   - NO2 (μg/m³)
   - CO (mg/m³)
   - O3 (μg/m³)
3. 点击「预测未来24小时」
4. 查看PM2.5预测值及空气质量等级

### 🚮 河面垃圾检测
1. 进入系统，点击「垃圾检测」标签
2. 上传河面图片
3. 点击「开始检测」
4. 查看7类垃圾识别结果：
   - branch (树枝)
   - leaf (树叶)
   - others (其他)
   - plastic-bag (塑料袋)
   - plastic-bottle (塑料瓶)
   - plastic-wrapper (塑料包装)
   - wood-log (木头)

### 🔗 综合评估
1. 进入系统，点击「综合评估」标签
2. 输入时序评分和图像评分
3. 点击「综合评估」
4. 查看融合模型的综合判断

## 🔧 模型API

### LSTM时序预测
```python
import torch

model = torch.load('lstm_best.pth', map_location='cpu')
model.eval()

# 输入: (batch_size, 24, 11) - 24小时历史数据，11维特征
input_data = torch.randn(1, 24, 11)

with torch.no_grad():
    prediction = model(input_data)
# 输出: (batch_size, 1) - 未来PM2.5预测值
```

### ResNet+SE图像分类
```python
import torch
import torchvision.transforms as T
from PIL import Image

model = torch.load('resnet_se_best.pth', map_location='cpu')
model.eval()

# 图像预处理
transform = T.Compose([
    T.Resize(224),
    T.ToTensor()
])

img = Image.open('water.jpg').convert('RGB')
img_tensor = transform(img).unsqueeze(0)

with torch.no_grad():
    output = model(img_tensor)
    label = output.argmax(1).item()

# 标签: 0-清洁, 1-轻度污染, 2-中度污染, 3-重度污染
```

### YOLO目标检测
```python
from ultralytics import YOLO

model = YOLO('rubbish_best.pt')

# 图片检测
results = model('river.jpg', save=True)

# 打印检测结果
for r in results:
    print(r.boxes)
```

## 📊 性能指标

### 第三章：空气质量时序预测 (LSTM)
| 指标 | 数值 |
|------|------|
| RMSE | 8.7 μg/m³ |
| MAE | 6.2 μg/m³ |
| R² | 0.94 |

### 第四章：水质图像分类 (ResNet+SE)
| 指标 | 数值 |
|------|------|
| 准确率 | 90.8% |
| F1-Score | 0.89 |
| 精确率 | 91.2% |
| 召回率 | 90.5% |

### 第五章：河面垃圾检测 (YOLO11n)
| 指标 | 数值 |
|------|------|
| mAP50 | 93.3% |
| mAP50-95 | 85.9% |
| Precision | 91.5% |
| Recall | 90.8% |

### 第六章：多模态融合评估
| 指标 | 数值 |
|------|------|
| 综合准确率 | 94% |

## 📚 数据集来源

### 空气质量数据
- **来源**: UCI Machine Learning Repository
- **数据集**: Beijing Multi-Site Air Quality Data
- **链接**: https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data
- **规模**: 42万+条记录，12个监测站点

### 河面垃圾数据
- **来源**: Roboflow
- **数据集**: Rubbish Detection
- **数量**: 2120张（训练1845 + 验证175 + 测试100）

## 🔍 常见问题

### Q1: 模型加载报错？
```python
# 确保使用正确的加载方式
model = torch.load('model.pth', map_location='cpu')
```

### Q2: Web服务无法访问？
```bash
# 检查防火墙
sudo firewall-cmd --add-port=5000/tcp
sudo firewall-cmd --reload
```

### Q3: 内存不足？
- 减少batch_size
- 使用CPU模式（将代码中的cuda改为cpu）
- 使用更小的模型

### Q4: 如何远程访问？
```bash
# 修改app_env.py中的host
app.run(host='0.0.0.0', port=5000)

# 局域网访问 http://你的IP:5000
```

## 📄 论文引用

如果您在研究中使用了本系统，请引用：

```bibtex
@software{environmental_monitoring_2026,
  title={基于深度学习的智能环境监测系统},
  author={智能环保研究团队},
  year={2026},
  url={https://github.com/你的用户名/智能环境监测系统}
}
```

## 🤝 贡献指南

欢迎提交Pull Request！

1. Fork 本项目
2. 创建特性分支 (`git checkout -b feature/xxx`)
3. 提交更改 (`git commit -m 'Add xxx'`)
4. 推送到分支 (`git push origin feature/xxx`)
5. 创建Pull Request

## 📝 更新日志

### v1.0 (2026-04-10)
- ✅ 初始版本发布
- ✅ 4个核心功能模块
- ✅ Web演示系统
- ✅ 完整训练模型
- ✅ 论文辅助资料

## 📧 联系方式

- 项目维护者: 傻妞AI助手
- Email: support@example.com
- 问题反馈: GitHub Issues

---

**许可证**: MIT License  
**版权所有** © 2026 智能环保研究团队