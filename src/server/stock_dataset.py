#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
股票数据集类（支持mask）
版本: v0.2
日期: 20260118

实现PyTorch Dataset类，用于加载股票数据
v0.2新增：返回mask标识空白数据（-1000标记）
"""

import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, Dict, List
import importlib.util

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent
utils_path = project_root / "src" / "utils" / "v0.1_20251212"
data_path = project_root / "src" / "data" / "v0.2_20260118"

# 动态导入模块
def _load_module(module_path: Path, module_name: str):
    """动态加载模块"""
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# 导入工具模块
try:
    data_utils_module = _load_module("src/server/code/data_utils.py", "data_utils")
    feature_utils_module = _load_module("src/server/code/feature_utils.py", "feature_utils")
    feature_selector_module = _load_module( "src/server/code/feature_selector.py", "feature_selector")
    
    # 导入需要的函数和类
    load_parquet_file = data_utils_module.load_parquet_file
    get_data_by_rows = data_utils_module.get_data_by_rows
    get_data_by_single_row = data_utils_module.get_data_by_single_row
    get_file_cache = data_utils_module.get_file_cache
    apply_normalization = feature_utils_module.apply_normalization
    load_feature_stats = feature_utils_module.load_feature_stats
    FeatureSelector = feature_selector_module.FeatureSelector
except Exception as e:
    raise ImportError(f"Failed to import required modules: {e}") from e


class StockDataset(Dataset):
    """
    股票数据集类（支持mask）
    从索引文件中读取样本信息，加载对应的数据文件，返回训练样本
    
    v0.2新增：返回mask标识空白数据（值为-1000的位置）
    """
    
    def __init__(
        self,
        index_file: str,
        exclude_columns: List[str],
        target_column: str,
        normalize: bool = True,
        normalize_method: str = "standard",
        feature_stats: Optional[Dict] = None,
        stats_file: Optional[str] = None,
        cache_enabled: bool = True,
        cache_size: int = 100,
        blank_value: float = -1000.0,  # 空白数据标记值
        return_mask: bool = True  # 是否返回mask
    ):
        """
        初始化数据集
        
        Args:
            index_file: 索引文件路径（parquet格式）
            exclude_columns: 需要排除的元数据列列表
            target_column: 目标列名
            normalize: 是否标准化特征
            normalize_method: 标准化方法，"standard"或"minmax"
            feature_stats: 特征统计量字典（如果为None，需要从stats_file加载）
            stats_file: 统计量文件路径（如果feature_stats为None）
            cache_enabled: 是否启用文件缓存
            cache_size: 缓存文件数量上限
            blank_value: 空白数据标记值（默认-1000）
            return_mask: 是否返回mask（默认True）
        """
        # 读取索引文件
        self.index_df = pd.read_parquet(index_file)
        self.num_samples = len(self.index_df)
        
        # 初始化特征选择器
        self.feature_selector = FeatureSelector(
            exclude_columns=exclude_columns,
            target_column=target_column
        )
        
        # 标准化配置
        self.normalize = normalize
        self.normalize_method = normalize_method
        
        # mask配置
        self.blank_value = blank_value
        self.return_mask = return_mask
        
        # 加载特征统计量（用于标准化）
        if normalize:
            if feature_stats is not None:
                self.feature_stats = feature_stats
            elif stats_file is not None:
                self.feature_stats = load_feature_stats(stats_file)
            else:
                raise ValueError("Either feature_stats or stats_file must be provided when normalize=True")
        else:
            self.feature_stats = None
        
        # 文件缓存
        self.cache_enabled = cache_enabled
        if cache_enabled:
            self.cache = get_file_cache(max_size=cache_size)
        else:
            self.cache = None
        
        # 缓存特征列数量（避免重复计算）
        self._num_features: Optional[int] = None
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return self.num_samples
    
    def __getitem__(self, idx: int) -> tuple:
        """
        获取单个样本
        
        Args:
            idx: 样本索引
        
        Returns:
            如果return_mask=True:
                (X, y, mask) 元组
                - X: torch.Tensor, 形状为 (seq_len, num_features)
                - y: torch.Tensor, 形状为 (1,)
                - mask: torch.Tensor (bool), 形状为 (seq_len, num_features)
                        True表示有效数据，False表示空白数据
            如果return_mask=False:
                (X, y) 元组（向后兼容）
        """
        # 获取样本信息
        sample = self.index_df.iloc[idx]
        
        # 加载数据
        X, y = self._load_sample_data(sample)
        
        # 转换为torch tensor
        X_tensor = torch.FloatTensor(X.values)
        y_tensor = torch.FloatTensor([y])
        
        if self.return_mask:
            # 生成mask: True表示有效数据，False表示空白数据
            mask = torch.BoolTensor((X.values != self.blank_value))
            return X_tensor, y_tensor, mask
        else:
            return X_tensor, y_tensor
    
    def _load_sample_data(self, sample: pd.Series) -> tuple:
        """
        加载单个样本的数据
        
        Args:
            sample: 样本信息（索引文件中的一行）
        
        Returns:
            (特征DataFrame, 目标值)
        """
        # 读取原始数据文件
        source_file = sample['source_file']
        df = load_parquet_file(
            filepath=source_file,
            use_cache=self.cache_enabled,
            cache=self.cache
        )
        
        # 提取输入数据（根据行号范围）
        input_start = sample['input_row_start']
        input_end = sample['input_row_end']
        input_df = get_data_by_rows(df, input_start, input_end)
        
        # 提取目标数据（根据行号）
        target_row = sample['target_row']
        target_series = get_data_by_single_row(df, target_row)
        target_column = self.feature_selector.get_target_column()
        target_value = target_series[target_column]
        
        # 特征选择
        feature_columns = self.feature_selector.get_feature_columns(input_df)
        X = input_df[feature_columns].copy()
        
        # 检查并处理NaN值（在标准化之前）
        # 注意：数据应该已经预处理过，理论上不应该有NaN
        # 如果仍有NaN，直接填充为blank_value（默认-1000）
        nan_count_before = X.isna().sum().sum()
        if nan_count_before > 0:
            # 将NaN填充为空白标记值
            X = X.fillna(self.blank_value)
        
        # 特征标准化
        if self.normalize and self.feature_stats is not None:
            X = apply_normalization(
                X,
                feature_columns=feature_columns,
                stats=self.feature_stats,
                method=self.normalize_method,
                inplace=False
            )
        
        # 再次检查NaN（标准化后可能产生NaN）
        nan_count_after = X.isna().sum().sum()
        if nan_count_after > 0:
            # 标准化后如果还有NaN，填充为blank_value
            X = X.fillna(self.blank_value)
        
        return X, target_value
    
    def get_num_features(self) -> int:
        """
        获取特征数量
        
        Returns:
            特征数量
        """
        if self._num_features is None:
            # 从第一个样本获取特征数量
            if len(self) > 0:
                sample = self.index_df.iloc[0]
                source_file = sample['source_file']
                df = load_parquet_file(
                    filepath=source_file,
                    use_cache=self.cache_enabled,
                    cache=self.cache
                )
                input_start = sample['input_row_start']
                input_end = sample['input_row_end']
                input_df = get_data_by_rows(df, input_start, input_end)
                feature_columns = self.feature_selector.get_feature_columns(input_df)
                self._num_features = len(feature_columns)
            else:
                raise ValueError("Dataset is empty")
        
        return self._num_features
    
    def get_seq_len(self) -> int:
        """
        获取序列长度（输入窗口大小）
        
        Returns:
            序列长度
        """
        if len(self) > 0:
            sample = self.index_df.iloc[0]
            seq_len = sample['input_row_end'] - sample['input_row_start'] + 1
            return int(seq_len)
        else:
            raise ValueError("Dataset is empty")
