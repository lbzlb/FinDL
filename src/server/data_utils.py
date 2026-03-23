#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据工具函数
版本: v0.1
日期: 20251212

提供文件缓存、数据加载等功能
"""

import pandas as pd
from pathlib import Path
from typing import Optional, Union
from collections import OrderedDict
import threading


class FileCache:
    """
    LRU文件缓存类
    使用OrderedDict实现LRU缓存策略
    """
    
    def __init__(self, max_size: int = 100):
        """
        初始化文件缓存
        
        Args:
            max_size: 最大缓存文件数量
        """
        self.max_size = max_size
        self.cache: OrderedDict[str, pd.DataFrame] = OrderedDict()
        self.lock = threading.Lock()  # 线程锁，保证线程安全
    
    def get(self, filepath: str) -> Optional[pd.DataFrame]:
        """
        从缓存中获取文件
        
        Args:
            filepath: 文件路径
        
        Returns:
            DataFrame或None（如果不在缓存中）
        """
        filepath = str(Path(filepath).resolve())  # 标准化路径
        
        with self.lock:
            if filepath in self.cache:
                # 移动到末尾（表示最近使用）
                self.cache.move_to_end(filepath)
                return self.cache[filepath].copy()  # 返回副本，避免外部修改影响缓存
            return None
    
    def put(self, filepath: str, df: pd.DataFrame) -> None:
        """
        将文件放入缓存
        
        Args:
            filepath: 文件路径
            df: DataFrame数据
        """
        filepath = str(Path(filepath).resolve())  # 标准化路径
        
        with self.lock:
            # 如果已存在，先移除
            if filepath in self.cache:
                self.cache.move_to_end(filepath)
            else:
                # 如果缓存已满，移除最旧的项
                if len(self.cache) >= self.max_size:
                    self.cache.popitem(last=False)  # 移除最旧的项
            
            # 添加到缓存末尾
            self.cache[filepath] = df.copy()  # 存储副本
    
    def clear(self) -> None:
        """清空缓存"""
        with self.lock:
            self.cache.clear()
    
    def size(self) -> int:
        """返回当前缓存大小"""
        with self.lock:
            return len(self.cache)
    
    def contains(self, filepath: str) -> bool:
        """检查文件是否在缓存中"""
        filepath = str(Path(filepath).resolve())
        with self.lock:
            return filepath in self.cache


# 全局文件缓存实例（单例模式）
_global_cache: Optional[FileCache] = None
_cache_lock = threading.Lock()


def get_file_cache(max_size: int = 100) -> FileCache:
    """
    获取全局文件缓存实例（单例模式）
    
    Args:
        max_size: 最大缓存文件数量
    
    Returns:
        文件缓存实例
    """
    global _global_cache
    
    with _cache_lock:
        if _global_cache is None:
            _global_cache = FileCache(max_size=max_size)
        return _global_cache


def load_parquet_file(
    filepath: Union[str, Path],
    use_cache: bool = True,
    cache: Optional[FileCache] = None
) -> pd.DataFrame:
    """
    加载parquet文件（带缓存）
    
    Args:
        filepath: 文件路径
        use_cache: 是否使用缓存
        cache: 缓存实例（如果为None，使用全局缓存）
    
    Returns:
        DataFrame
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    # 如果使用缓存
    if use_cache:
        if cache is None:
            cache = get_file_cache()
        
        # 尝试从缓存获取
        cached_df = cache.get(str(filepath))
        if cached_df is not None:
            return cached_df
    
    # 加载文件
    df = pd.read_parquet(filepath)
    
    # 如果使用缓存，存入缓存
    if use_cache:
        if cache is None:
            cache = get_file_cache()
        cache.put(str(filepath), df)
    
    return df


def get_data_by_rows(
    df: pd.DataFrame,
    start_row: int,
    end_row: int
) -> pd.DataFrame:
    """
    根据行号范围提取数据（包含两端）
    
    Args:
        df: 输入DataFrame
        start_row: 起始行号（包含）
        end_row: 结束行号（包含）
    
    Returns:
        提取的数据DataFrame
    """
    if start_row < 0:
        start_row = 0
    if end_row >= len(df):
        end_row = len(df) - 1
    
    if start_row > end_row:
        raise ValueError(f"Invalid row range: start_row ({start_row}) > end_row ({end_row})")
    
    return df.iloc[start_row:end_row + 1].copy()


def get_data_by_single_row(
    df: pd.DataFrame,
    row: int
) -> pd.Series:
    """
    根据单个行号提取数据
    
    Args:
        df: 输入DataFrame
        row: 行号
    
    Returns:
        提取的数据Series
    """
    if row < 0 or row >= len(df):
        raise IndexError(f"Row index {row} out of range [0, {len(df)})")
    
    return df.iloc[row].copy()


def validate_dataframe(df: pd.DataFrame, required_columns: Optional[list] = None) -> bool:
    """
    验证DataFrame是否符合要求
    
    Args:
        df: 输入DataFrame
        required_columns: 必需的列名列表
    
    Returns:
        是否通过验证
    
    Raises:
        ValueError: 如果验证失败
    """
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    if required_columns is not None:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
    
    return True


def normalize_path(path: Union[str, Path]) -> Path:
    """
    标准化路径
    
    Args:
        path: 路径字符串或Path对象
    
    Returns:
        标准化后的Path对象
    """
    return Path(path).resolve()
