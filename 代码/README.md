# 文件夹2 - 基于深度学习的智能环保监测算法研究
# 项目代码 - 黄丽佳 22460525 指导教师：曾德真

## 项目结构
```
env_monitor/
├── data/
│   ├── air/             # 空气质量数据
│   ├── water/           # 水质图像数据
│   └── noise/           # 噪声数据
├── models/
│   ├── cnn_detector.py  # CNN图像检测模型
│   ├── lstm_predictor.py # LSTM时序预测模型
│   └── fusion_model.py  # 多模态融合模型
├── utils/
│   ├── data_loader.py   # 数据加载
│   └── preprocess.py    # 数据预处理
├── train_image.py       # 图像模型训练
├── train_timeseries.py  # 时序模型训练
├── train_fusion.py      # 融合模型训练
├── evaluate.py          # 评估脚本
├── visualize.py         # 可视化（GIS集成）
└── requirements.txt     # 依赖
```

## 环境要求
- Python 3.8+
- PyTorch 2.0+
- TensorFlow/Keras（备用）
- folium（GIS可视化）

## 运行流程
1. `pip install -r requirements.txt`
2. 准备多模态数据
3. `python train_image.py` 训练图像检测模型
4. `python train_timeseries.py` 训练时序预测模型
5. `python train_fusion.py` 训练融合模型
6. `python evaluate.py` 综合评估
7. `python visualize.py` 生成可视化报告
