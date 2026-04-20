"""
基于深度学习的智能环保监测算法研究
CNN图像检测模型 - 水质/大气污染视觉识别
黄丽佳 22460525

基于ResNet50 + 注意力机制的图像分类与检测模型，
用于识别水体漂浮物、油污、藻华等视觉污染特征。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation注意力模块
    通过通道注意力机制增强重要特征通道的响应
    """
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class WaterQualityCNN(nn.Module):
    """
    水质图像分类模型
    基于ResNet50骨干网络，集成SE注意力模块，
    用于识别水质等级（I-V类及劣V类）和污染类型。
    
    输入：224x224 RGB水体图像
    输出：水质等级分类 + 污染类型分类
    """
    def __init__(self, num_classes=6, num_pollution_types=5, pretrained=True):
        super(WaterQualityCNN, self).__init__()
        
        # 骨干网络：ResNet50
        backbone = models.resnet50(pretrained=pretrained)
        self.features = nn.Sequential(*list(backbone.children())[:-2])
        
        # SE注意力模块
        self.se1 = SEBlock(256)
        self.se2 = SEBlock(512)
        self.se3 = SEBlock(1024)
        self.se4 = SEBlock(2048)
        
        # 全局池化
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # 水质等级分类头
        self.quality_classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
        # 污染类型分类头
        self.pollution_classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_pollution_types)
        )
    
    def forward(self, x):
        # 特征提取（分阶段加入SE注意力）
        x = self.features[0](x)   # Conv1
        x = self.features[1](x)   # BN1
        x = self.features[2](x)   # ReLU
        x = self.features[3](x)   # MaxPool
        
        x = self.features[4](x)   # Layer1
        x = self.se1(x)
        
        x = self.features[5](x)   # Layer2
        x = self.se2(x)
        
        x = self.features[6](x)   # Layer3
        x = self.se3(x)
        
        x = self.features[7](x)   # Layer4
        x = self.se4(x)
        
        # 全局池化
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # 多任务输出
        quality_out = self.quality_classifier(x)
        pollution_out = self.pollution_classifier(x)
        
        return quality_out, pollution_out


class PollutionDetector(nn.Module):
    """
    污染源目标检测模型
    基于YOLO架构，用于检测和定位水体中的漂浮物、油污、排放口等污染源。
    """
    def __init__(self, num_classes=5):
        super(PollutionDetector, self).__init__()
        
        # Backbone: CSPDarknet-lite
        self.backbone = nn.Sequential(
            # Stem
            nn.Conv2d(3, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            
            # Stage 1
            nn.Conv2d(32, 64, 3, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            
            # Stage 2
            nn.Conv2d(64, 128, 3, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),
            
            # Stage 3
            nn.Conv2d(128, 256, 3, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.SiLU(inplace=True),
            
            # Stage 4
            nn.Conv2d(256, 512, 3, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.SiLU(inplace=True),
        )
        
        # 检测头
        self.detect_head = nn.Sequential(
            nn.Conv2d(512, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.SiLU(inplace=True),
            nn.Conv2d(256, (5 + num_classes) * 3, 1)  # 3 anchors per cell
        )
        
        self.num_classes = num_classes
    
    def forward(self, x):
        features = self.backbone(x)
        detection = self.detect_head(features)
        return detection


def build_model(model_type='classifier', num_classes=6, pretrained=True):
    """模型工厂函数"""
    if model_type == 'classifier':
        return WaterQualityCNN(num_classes=num_classes, pretrained=pretrained)
    elif model_type == 'detector':
        return PollutionDetector(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
