#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
损失函数工具
版本: v0.2
日期: 20260121

提供自定义损失函数类
v0.2新增: SMAPELoss - 对称平均绝对百分比误差
"""

import torch
import torch.nn as nn


class MAPELoss(nn.Module):
    """
    平均绝对百分比误差损失函数 (Mean Absolute Percentage Error)
    
    公式: MAPE = mean(|pred - target| / (|target| + epsilon))
    
    特点:
    - 对相对误差敏感，适合不同量级的数据
    - 当target接近0时，使用epsilon避免除零
    - 支持裁剪相对误差上限，避免极端值主导训练
    
    示例:
        >>> criterion = MAPELoss(reduction='mean', epsilon=1e-8, max_relative_error=5.0)
        >>> pred = torch.tensor([10.0, 20.0, 30.0])
        >>> target = torch.tensor([9.0, 21.0, 29.0])
        >>> loss = criterion(pred, target)
    """
    def __init__(self, reduction='mean', epsilon=1e-8, max_relative_error=None):
        """
        初始化MAPE损失函数
        
        Args:
            reduction: 损失缩减方式，可选 'mean', 'sum', 'none'
            epsilon: 防止除零的小值，默认1e-8
            max_relative_error: 相对误差的上限（可选）
                               例如5.0表示限制单个样本的相对误差最大为5.0（500%）
                               None表示不裁剪，使用原始MAPE
        """
        super(MAPELoss, self).__init__()
        self.reduction = reduction
        self.epsilon = epsilon
        self.max_relative_error = max_relative_error
    
    def forward(self, pred, target):
        """
        前向传播
        
        Args:
            pred: 预测值，形状为 (batch_size, ...)
            target: 真实值，形状与pred相同
        
        Returns:
            损失值（标量或与输入相同形状）
        """
        # 计算绝对误差
        abs_error = torch.abs(pred - target)
        
        # 计算分母（避免除零）
        denominator = torch.abs(target) + self.epsilon
        
        # 计算相对误差
        relative_error = abs_error / denominator
        
        # 裁剪相对误差（如果设置了max_relative_error）
        if self.max_relative_error is not None:
            relative_error = torch.clamp(relative_error, max=self.max_relative_error)
        
        # 根据reduction参数返回结果
        if self.reduction == 'mean':
            return torch.mean(relative_error)
        elif self.reduction == 'sum':
            return torch.sum(relative_error)
        elif self.reduction == 'none':
            return relative_error
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")


class SMAPELoss(nn.Module):
    """
    对称平均绝对百分比误差损失函数 (Symmetric Mean Absolute Percentage Error)
    
    公式: SMAPE = mean(2 * |pred - target| / (|target| + |pred| + epsilon))
    
    特点:
    - 对称性：pred和target的位置可以互换，结果一致
    - 对接近0的值更加鲁棒：当target和pred都接近0时，损失不会爆炸
    - 有界性：相对误差的值域为[0, 2]（0%-200%），比MAPE更稳定
    - 适合处理包含接近0值的时间序列预测任务
    
    优势对比MAPE:
    - MAPE问题：当target接近0时，|pred-target|/|target|会变得很大，导致模型过度关注接近0的样本
    - SMAPE解决：分母包含pred和target，当两者都小时分子也小，相对误差不会爆炸
    
    示例:
        >>> criterion = SMAPELoss(reduction='mean', epsilon=1e-8, max_relative_error=2.0)
        >>> pred = torch.tensor([10.0, 20.0, 30.0])
        >>> target = torch.tensor([9.0, 21.0, 29.0])
        >>> loss = criterion(pred, target)
    """
    def __init__(self, reduction='mean', epsilon=1e-8, max_relative_error=None):
        """
        初始化SMAPE损失函数
        
        Args:
            reduction: 损失缩减方式，可选 'mean', 'sum', 'none'
            epsilon: 防止除零的小值，默认1e-8
            max_relative_error: 相对误差的上限（可选）
                               例如2.0表示限制单个样本的相对误差最大为2.0（200%）
                               None表示不裁剪，使用原始SMAPE
                               注意：SMAPE的理论上界为2.0
        """
        super(SMAPELoss, self).__init__()
        self.reduction = reduction
        self.epsilon = epsilon
        self.max_relative_error = max_relative_error
    
    def forward(self, pred, target):
        """
        前向传播
        
        Args:
            pred: 预测值，形状为 (batch_size, ...)
            target: 真实值，形状与pred相同
        
        Returns:
            损失值（标量或与输入相同形状）
        """
        # 计算绝对误差
        abs_error = torch.abs(pred - target)
        
        # 计算对称分母（避免除零）
        denominator = torch.abs(target) + torch.abs(pred) + self.epsilon
        
        # 计算对称相对误差（乘以2是为了保持与传统SMAPE定义一致）
        relative_error = 2.0 * abs_error / denominator
        
        # 裁剪相对误差（如果设置了max_relative_error）
        if self.max_relative_error is not None:
            relative_error = torch.clamp(relative_error, max=self.max_relative_error)
        
        # 根据reduction参数返回结果
        if self.reduction == 'mean':
            return torch.mean(relative_error)
        elif self.reduction == 'sum':
            return torch.sum(relative_error)
        elif self.reduction == 'none':
            return relative_error
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")

