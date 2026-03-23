#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
评估指标工具
版本: v0.1
日期: 20251212

提供回归任务的评估指标计算功能
"""

import numpy as np
from typing import Dict, Union, Optional
import torch


def _to_numpy(x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """
    将输入转换为numpy数组
    
    Args:
        x: numpy数组或torch tensor
    
    Returns:
        numpy数组
    """
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    elif isinstance(x, np.ndarray):
        return x
    else:
        return np.array(x)


def _clean_values(y_true: np.ndarray, y_pred: np.ndarray) -> tuple:
    """
    清理异常值（NaN、无穷值）
    
    Args:
        y_true: 真实值
        y_pred: 预测值
    
    Returns:
        清理后的(y_true, y_pred)
    """
    # 转换为numpy数组
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)
    
    # 展平为一维数组
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    # 检查长度是否一致
    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}")
    
    # 找出有效值（非NaN、非无穷）
    valid_mask = np.isfinite(y_true) & np.isfinite(y_pred)
    
    if not np.any(valid_mask):
        raise ValueError("No valid values found after cleaning")
    
    y_true_clean = y_true[valid_mask]
    y_pred_clean = y_pred[valid_mask]
    
    return y_true_clean, y_pred_clean


def mse(y_true: Union[np.ndarray, torch.Tensor], 
        y_pred: Union[np.ndarray, torch.Tensor]) -> float:
    """
    计算均方误差 (Mean Squared Error)
    
    Args:
        y_true: 真实值
        y_pred: 预测值
    
    Returns:
        MSE值
    """
    y_true, y_pred = _clean_values(y_true, y_pred)
    return float(np.mean((y_true - y_pred) ** 2))


def mae(y_true: Union[np.ndarray, torch.Tensor], 
        y_pred: Union[np.ndarray, torch.Tensor]) -> float:
    """
    计算平均绝对误差 (Mean Absolute Error)
    
    Args:
        y_true: 真实值
        y_pred: 预测值
    
    Returns:
        MAE值
    """
    y_true, y_pred = _clean_values(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: Union[np.ndarray, torch.Tensor], 
         y_pred: Union[np.ndarray, torch.Tensor]) -> float:
    """
    计算均方根误差 (Root Mean Squared Error)
    
    Args:
        y_true: 真实值
        y_pred: 预测值
    
    Returns:
        RMSE值
    """
    return np.sqrt(mse(y_true, y_pred))


def mape(y_true: Union[np.ndarray, torch.Tensor], 
         y_pred: Union[np.ndarray, torch.Tensor],
         epsilon: float = 1e-8,
         max_relative_error: Optional[float] = None) -> float:
    """
    计算平均绝对百分比误差 (Mean Absolute Percentage Error)
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        epsilon: 防止除零的小值
        max_relative_error: 相对误差的上限（可选）
                           例如5.0表示限制单个样本的相对误差最大为5.0（500%）
                           None表示不裁剪，使用原始MAPE
    
    Returns:
        MAPE值（百分比形式，例如10.5表示10.5%）
    """
    y_true, y_pred = _clean_values(y_true, y_pred)
    
    # 避免除零
    denominator = np.abs(y_true) + epsilon
    relative_errors = np.abs((y_true - y_pred) / denominator)
    
    # 裁剪相对误差（如果设置了max_relative_error）
    if max_relative_error is not None:
        relative_errors = np.clip(relative_errors, a_min=None, a_max=max_relative_error)
    
    # 转换为百分比
    percentage_errors = relative_errors * 100
    
    return float(np.mean(percentage_errors))


def r2_score(y_true: Union[np.ndarray, torch.Tensor], 
             y_pred: Union[np.ndarray, torch.Tensor]) -> float:
    """
    计算R²决定系数 (Coefficient of Determination)
    
    Args:
        y_true: 真实值
        y_pred: 预测值
    
    Returns:
        R²值
    """
    y_true, y_pred = _clean_values(y_true, y_pred)
    
    # 计算总平方和
    ss_res = np.sum((y_true - y_pred) ** 2)  # 残差平方和
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)  # 总平方和
    
    # 避免除零
    if ss_tot < 1e-10:
        return 0.0
    
    r2 = 1 - (ss_res / ss_tot)
    return float(r2)


def compute_metrics(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
    metrics: Optional[list] = None,
    max_relative_error: Optional[float] = None,
    epsilon: float = 1e-8
) -> Dict[str, float]:
    """
    一次性计算多个评估指标
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        metrics: 要计算的指标列表，如果为None则计算所有指标
                 可选值: ["mse", "mae", "rmse", "mape", "r2"]
        max_relative_error: MAPE的相对误差上限（可选）
                           例如5.0表示限制单个样本的相对误差最大为5.0（500%）
                           None表示不裁剪，使用原始MAPE
        epsilon: MAPE计算中防止除零的小值（默认1e-8）
    
    Returns:
        指标字典，格式为 {metric_name: metric_value}
    """
    if metrics is None:
        metrics = ["mse", "mae", "rmse", "mape", "r2"]
    
    results = {}
    
    metric_functions = {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "r2": r2_score
    }
    
    for metric_name in metrics:
        if metric_name not in metric_functions:
            raise ValueError(f"Unknown metric: {metric_name}. Available: {list(metric_functions.keys())}")
        
        try:
            # 为MAPE传递epsilon和max_relative_error参数
            if metric_name == "mape":
                metric_value = metric_functions[metric_name](y_true, y_pred, 
                                                             epsilon=epsilon,
                                                             max_relative_error=max_relative_error)
            else:
                metric_value = metric_functions[metric_name](y_true, y_pred)
            results[metric_name] = metric_value
        except (ValueError, ZeroDivisionError, RuntimeError) as e:
            # 如果某个指标计算失败（如数据问题、除零等），记录错误但继续计算其他指标
            results[metric_name] = float('nan')
            print(f"Warning: Failed to compute {metric_name}: {e}")
    
    return results


def format_metrics(metrics: Dict[str, float], precision: int = 4) -> str:
    """
    格式化指标字典为字符串
    
    Args:
        metrics: 指标字典
        precision: 小数位数
    
    Returns:
        格式化后的字符串
    """
    lines = []
    for metric_name, metric_value in metrics.items():
        if np.isnan(metric_value) or np.isinf(metric_value):
            lines.append(f"{metric_name}: NaN/Inf")
        else:
            lines.append(f"{metric_name}: {metric_value:.{precision}f}")
    
    return ", ".join(lines)
