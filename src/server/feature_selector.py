#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
特征选择器
版本: v0.1
日期: 20251212

用于从DataFrame中识别和提取特征列和目标列
"""

import pandas as pd
from typing import List, Tuple, Optional


def _get_feature_columns(df: pd.DataFrame, exclude_columns: List[str]) -> List[str]:
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


def _extract_features_and_target(
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
    features = df[feature_columns].copy()
    target = df[target_column].copy()
    return features, target


class FeatureSelector:
    """
    特征选择器类
    用于从DataFrame中提取特征列和目标列
    """
    
    def __init__(
        self,
        exclude_columns: List[str],
        target_column: str
    ):
        """
        初始化特征选择器
        
        Args:
            exclude_columns: 需要排除的元数据列列表（不用于训练）
            target_column: 目标列名（要预测的列）
        """
        self.exclude_columns = exclude_columns
        self.target_column = target_column
        self._feature_columns: Optional[List[str]] = None
    
    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """
        获取特征列列表（排除元数据列）
        
        Args:
            df: 输入DataFrame
        
        Returns:
            特征列名列表
        """
        if self._feature_columns is None:
            self._feature_columns = _get_feature_columns(df, self.exclude_columns)
        return self._feature_columns
    
    def get_target_column(self) -> str:
        """
        获取目标列名
        
        Returns:
            目标列名
        """
        return self.target_column
    
    def validate_columns(self, df: pd.DataFrame) -> None:
        """
        验证DataFrame中是否包含必需的列
        
        Args:
            df: 输入DataFrame
        
        Raises:
            ValueError: 如果缺少必需的列
        """
        feature_columns = self.get_feature_columns(df)
        
        # 检查特征列是否存在
        missing_features = [col for col in feature_columns if col not in df.columns]
        if missing_features:
            raise ValueError(f"Missing feature columns: {missing_features}")
        
        # 检查目标列是否存在
        if self.target_column not in df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in DataFrame")
    
    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        从DataFrame中提取特征列
        
        Args:
            df: 输入DataFrame
        
        Returns:
            特征DataFrame
        """
        self.validate_columns(df)
        feature_columns = self.get_feature_columns(df)
        return df[feature_columns].copy()
    
    def extract_target(self, df: pd.DataFrame) -> pd.Series:
        """
        从DataFrame中提取目标列
        
        Args:
            df: 输入DataFrame
        
        Returns:
            目标Series
        """
        self.validate_columns(df)
        return df[self.target_column].copy()
    
    def extract_features_and_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        从DataFrame中同时提取特征和目标
        
        Args:
            df: 输入DataFrame
        
        Returns:
            (特征DataFrame, 目标Series)
        """
        self.validate_columns(df)
        feature_columns = self.get_feature_columns(df)
        return _extract_features_and_target(df, feature_columns, self.target_column)
    
    def get_num_features(self, df: pd.DataFrame) -> int:
        """
        获取特征数量
        
        Args:
            df: 输入DataFrame
        
        Returns:
            特征数量
        """
        feature_columns = self.get_feature_columns(df)
        return len(feature_columns)
    
    def reset(self) -> None:
        """
        重置特征列缓存（当DataFrame结构发生变化时调用）
        """
        self._feature_columns = None
