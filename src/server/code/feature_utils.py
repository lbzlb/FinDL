#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
特征处理工具
版本: v0.1
日期: 20251212

提供特征标准化、统计量计算等功能
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Tuple, Union


def get_feature_columns(df: pd.DataFrame, exclude_columns: List[str]) -> List[str]:
    """
    从DataFrame中获取特征列名列表（排除元数据列和非数值列）
    
    Args:
        df: 输入DataFrame
        exclude_columns: 需要排除的列名列表
    
    Returns:
        特征列名列表（只包含数值类型）
    """
    all_columns = df.columns.tolist()
    # 先排除指定的列
    candidate_columns = [col for col in all_columns if col not in exclude_columns]
    
    # 只保留数值类型的列
    feature_columns = []
    for col in candidate_columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            feature_columns.append(col)
        else:
            # 尝试转换为数值类型
            try:
                test_series = pd.to_numeric(df[col], errors='coerce')
                # 如果转换后至少有一些数值，则认为是数值列
                if test_series.notna().any():
                    feature_columns.append(col)
            except Exception:
                # 如果无法转换，跳过该列
                pass
    
    return feature_columns


def compute_feature_stats(
    df: pd.DataFrame,
    feature_columns: List[str],
    method: str = "standard"  # 保留参数以便将来扩展
) -> Dict[str, Dict[str, float]]:
    """
    计算特征统计量（均值、方差、最小值、最大值）
    
    Args:
        df: 输入DataFrame（通常是训练集）
        feature_columns: 特征列名列表
        method: 标准化方法，"standard"或"minmax"（当前未使用，保留以便将来扩展）
    
    Returns:
        字典，格式为 {column_name: {"mean": ..., "std": ..., "min": ..., "max": ...}}
    """
    stats = {}
    
    for col in feature_columns:
        if col not in df.columns:
            continue
        
        # 只处理数值类型的列
        if not pd.api.types.is_numeric_dtype(df[col]):
            # 尝试转换为数值类型
            try:
                series = pd.to_numeric(df[col], errors='coerce')
            except Exception:
                # 如果无法转换，跳过该列
                print(f"警告: 列 '{col}' 无法转换为数值类型，跳过统计量计算")
                continue
        else:
            series = df[col]
        
        # 排除NaN和0值（因为0是填充的NaN，没有实际意义）
        # 使用pandas的notna()方法过滤NaN值，同时排除0值
        valid_series = series[(series.notna()) & (series != 0)]
        
        if len(valid_series) == 0:
            # 如果列全为NaN或全为0，使用默认值
            stats[col] = {
                "mean": 0.0,
                "std": 1.0,
                "min": 0.0,
                "max": 1.0
            }
            print(f"警告: 列 '{col}' 全为NaN或0，使用默认统计量")
            continue
        
        # 转换为numpy数组进行计算
        valid_values = valid_series.values
        
        col_stats = {
            "mean": float(np.mean(valid_values)),
            "std": float(np.std(valid_values)),
            "min": float(np.min(valid_values)),
            "max": float(np.max(valid_values))
        }
        
        # 防止标准差为0
        if col_stats["std"] < 1e-8:
            col_stats["std"] = 1.0
        
        # 防止min和max相等
        if col_stats["max"] - col_stats["min"] < 1e-8:
            col_stats["max"] = col_stats["min"] + 1.0
        
        stats[col] = col_stats
    
    return stats


def standardize_features(
    df: pd.DataFrame,
    feature_columns: List[str],
    stats: Dict[str, Dict[str, float]],
    inplace: bool = False
) -> pd.DataFrame:
    """
    Z-score标准化特征：(x - mean) / std
    
    Args:
        df: 输入DataFrame
        feature_columns: 特征列名列表
        stats: 统计量字典（从compute_feature_stats获取）
        inplace: 是否原地修改
    
    Returns:
        标准化后的DataFrame
    """
    if not inplace:
        df = df.copy()
    
    for col in feature_columns:
        if col not in df.columns or col not in stats:
            continue
        
        # 确保列是数值类型
        if not pd.api.types.is_numeric_dtype(df[col]):
            # 尝试转换为数值类型
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except Exception:
                # 如果无法转换，跳过该列
                print(f"警告: 列 '{col}' 无法转换为数值类型，跳过标准化")
                continue
        
        # 确保列是float类型（归一化后会产生float值）
        if df[col].dtype != 'float64' and df[col].dtype != 'float32':
            df[col] = df[col].astype('float64')
        
        mean = stats[col]["mean"]
        std = stats[col]["std"]
        
        # 检查标准差是否为0或太小
        if std < 1e-8:
            # 如果标准差太小，说明该特征值几乎不变，标准化后设为0
            # 但保留原有的0值（填充的NaN）不变
            mask = df[col] != 0
            df.loc[mask, col] = 0.0
        else:
            # 标准化时排除0值（因为0是填充的NaN，没有实际意义）
            # 只对非0值进行标准化，0值保持为0
            mask = df[col] != 0
            if mask.any():
                df.loc[mask, col] = (df.loc[mask, col] - mean) / std
            # 0值保持为0，不做归一化
    
    return df


def normalize_features(
    df: pd.DataFrame,
    feature_columns: List[str],
    stats: Dict[str, Dict[str, float]],
    inplace: bool = False
) -> pd.DataFrame:
    """
    MinMax归一化特征：(x - min) / (max - min)
    
    Args:
        df: 输入DataFrame
        feature_columns: 特征列名列表
        stats: 统计量字典（从compute_feature_stats获取）
        inplace: 是否原地修改
    
    Returns:
        归一化后的DataFrame
    """
    if not inplace:
        df = df.copy()
    
    for col in feature_columns:
        if col not in df.columns or col not in stats:
            continue
        
        # 确保列是数值类型
        if not pd.api.types.is_numeric_dtype(df[col]):
            # 尝试转换为数值类型
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except Exception:
                # 如果无法转换，跳过该列
                print(f"警告: 列 '{col}' 无法转换为数值类型，跳过归一化")
                continue
        
        # 确保列是float类型（归一化后会产生float值）
        if df[col].dtype != 'float64' and df[col].dtype != 'float32':
            df[col] = df[col].astype('float64')
        
        min_val = stats[col]["min"]
        max_val = stats[col]["max"]
        range_val = max_val - min_val
        
        if range_val < 1e-8:
            # 如果范围太小，保持原值（但0值保持为0）
            mask = df[col] != 0
            df.loc[mask, col] = 0.0
            continue
        
        # 归一化时排除0值（因为0是填充的NaN，没有实际意义）
        # 只对非0值进行归一化，0值保持为0
        mask = df[col] != 0
        if mask.any():
            df.loc[mask, col] = (df.loc[mask, col] - min_val) / range_val
        # 0值保持为0，不做归一化
    
    return df


def apply_normalization(
    df: pd.DataFrame,
    feature_columns: List[str],
    stats: Dict[str, Dict[str, float]],
    method: str = "standard",
    inplace: bool = False
) -> pd.DataFrame:
    """
    应用标准化或归一化（根据method参数选择）
    
    Args:
        df: 输入DataFrame
        feature_columns: 特征列名列表
        stats: 统计量字典
        method: "standard"（Z-score）或"minmax"
        inplace: 是否原地修改
    
    Returns:
        处理后的DataFrame
    """
    if method == "standard":
        return standardize_features(df, feature_columns, stats, inplace)
    elif method == "minmax":
        return normalize_features(df, feature_columns, stats, inplace)
    else:
        raise ValueError(f"Unknown normalization method: {method}. Use 'standard' or 'minmax'.")


def save_feature_stats(stats: Dict[str, Dict[str, float]], filepath: Union[str, Path]) -> None:
    """
    保存特征统计量到JSON文件
    
    Args:
        stats: 统计量字典
        filepath: 保存路径
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)


def load_feature_stats(filepath: Union[str, Path]) -> Dict[str, Dict[str, float]]:
    """
    从JSON文件加载特征统计量
    
    Args:
        filepath: 文件路径
    
    Returns:
        统计量字典
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Stats file not found: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        stats = json.load(f)
    
    return stats


def extract_features_and_target(
    df: pd.DataFrame,
    feature_columns: List[str],
    target_column: str
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    从DataFrame中提取特征和目标
    
    Args:
        df: 输入DataFrame
        feature_columns: 特征列名列表
        target_column: 目标列名
    
    Returns:
        (特征DataFrame, 目标Series)
    """
    # 检查列是否存在
    missing_features = [col for col in feature_columns if col not in df.columns]
    if missing_features:
        raise ValueError(f"Missing feature columns: {missing_features}")
    
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame")
    
    features = df[feature_columns].copy()
    target = df[target_column].copy()
    
    return features, target
