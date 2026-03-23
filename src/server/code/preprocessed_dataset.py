#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
预处理数据集类（支持mask + 内存映射）
版本: v0.3
日期: 20260119

功能:
- 支持内存映射加载，避免一次性加载大文件到内存
- 直接加载预处理后的.pt文件
- 无需任何运行时数据处理
- 极快的数据访问速度
- 支持返回mask标识空白数据（-1000标记）
- 优化大数据集的内存使用

v0.3新增：
- 内存映射加载模式（mmap_mode）
- 批量加载模式（batch_load_mode）
- 更灵活的内存管理策略
"""

import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, Dict
import warnings


class PreprocessedStockDataset(Dataset):
    """
    预处理股票数据集类（支持mask + 内存映射）
    
    直接加载预处理好的tensor数据，避免运行时的数据加载和处理开销
    适用于训练速度要求高的场景
    
    v0.2: 支持mask标识空白数据
    v0.3: 支持内存映射加载大数据集
    """
    
    def __init__(
        self,
        pt_file_path: str,
        device: Optional[str] = None,
        blank_value: float = -1000.0,
        return_mask: bool = True,
        mmap_mode: bool = False,  # v0.3新增：是否使用内存映射
        precompute_mask: bool = False  # v0.3新增：是否预计算mask（仅mmap模式）
    ):
        """
        初始化预处理数据集
        
        Args:
            pt_file_path: 预处理的.pt文件路径
            load_to_memory: 是否将数据加载到内存（mmap_mode=True时忽略）
            device: 数据加载到的设备（None表示CPU，'cuda'表示GPU）
                   注意：mmap_mode=True时，device必须为None或'cpu'
            blank_value: 空白数据标记值（默认-1000）
            return_mask: 是否返回mask（默认True）
            mmap_mode: 是否使用内存映射模式（默认False）
                      True: 数据保留在磁盘，按需加载（适合大数据集）
                      False: 一次性加载到内存（适合小数据集，速度更快）
            precompute_mask: 是否预计算mask（仅在mmap_mode=True时有效）
                           True: 预先计算整个mask（需要遍历数据，慢但训练时快）
                           False: 运行时动态计算mask（快速启动，训练时略慢）
        """
        pt_file_path = Path(pt_file_path)
        
        if not pt_file_path.exists():
            raise FileNotFoundError(f"预处理文件不存在: {pt_file_path}")
        
        # 保存配置
        self.blank_value = blank_value
        self.return_mask = return_mask
        self.mmap_mode = mmap_mode
        self.device_name = device
        
        print(f"加载预处理数据: {pt_file_path}")
        
        if mmap_mode:
            # ===== 内存映射模式：数据保留在磁盘，按需加载 =====
            if device is not None and device != 'cpu':
                raise ValueError("内存映射模式只支持CPU设备（device=None或'cpu'）")
            
            print("  模式: 内存映射（数据保留在磁盘）")
            
            # 使用mmap加载（PyTorch 2.1+）
            try:
                # 尝试使用mmap参数（较新的PyTorch版本）
                data = torch.load(
                    pt_file_path, 
                    map_location='cpu',
                    weights_only=False,
                    mmap=True  # 关键参数：启用内存映射
                )
                print("  使用torch.load的mmap参数加载")
            except TypeError:
                # 旧版本PyTorch不支持mmap参数，回退到普通加载
                warnings.warn(
                    "当前PyTorch版本不支持mmap参数，回退到普通加载模式。"
                    "建议升级到PyTorch 2.1+以支持内存映射。",
                    UserWarning
                )
                data = torch.load(pt_file_path, map_location='cpu', weights_only=False)
            
            self.X = data['X']  # [N, seq_len, num_features]
            self.y = data['y']  # [N, 1]
            self.metadata = data.get('metadata', {})
            self.feature_stats = data.get('feature_stats', None)
            
            # mask处理
            if self.return_mask:
                if precompute_mask:
                    # 预计算mask（需要遍历整个数据集）
                    print("  预计算mask中...")
                    self.mask = (self.X != self.blank_value)
                    print(f"  已生成mask (空白数据标记为{self.blank_value})")
                    
                    # 统计空白数据比例
                    total_elements = self.mask.numel()
                    valid_elements = self.mask.sum().item()
                    blank_ratio = 1.0 - (valid_elements / total_elements)
                    print(f"  空白数据比例: {blank_ratio * 100:.2f}%")
                else:
                    # 运行时动态计算mask（快速启动）
                    self.mask = None
                    print(f"  mask将在运行时动态计算 (空白数据标记为{self.blank_value})")
            else:
                self.mask = None
        
        else:
            # ===== 普通模式：一次性加载到内存 =====
            print("  模式: 内存加载（数据全部加载到内存）")
            
            # 加载数据
            if device is not None:
                data = torch.load(pt_file_path, map_location=device, weights_only=False)
            else:
                data = torch.load(pt_file_path, weights_only=False)
            
            self.X = data['X']
            self.y = data['y']
            self.metadata = data.get('metadata', {})
            self.feature_stats = data.get('feature_stats', None)
            
            # 预计算mask
            if self.return_mask:
                self.mask = (self.X != self.blank_value)
                print(f"  已生成mask (空白数据标记为{self.blank_value})")
                
                # 统计空白数据比例
                total_elements = self.mask.numel()
                valid_elements = self.mask.sum().item()
                blank_ratio = 1.0 - (valid_elements / total_elements)
                print(f"  空白数据比例: {blank_ratio * 100:.2f}%")
            else:
                self.mask = None
        
        # 数据信息
        self.num_samples = len(self.X)
        self.num_features = self.X.shape[2]
        self.seq_len = self.X.shape[1]
        
        print(f"  样本数: {self.num_samples}")
        print(f"  特征数: {self.num_features}")
        print(f"  序列长度: {self.seq_len}")
        
        # 计算内存占用（仅估算）
        data_memory_mb = (self.X.element_size() * self.X.nelement() + 
                         self.y.element_size() * self.y.nelement()) / (1024 * 1024)
        
        if self.mask is not None:
            mask_memory_mb = self.mask.element_size() * self.mask.nelement() / (1024 * 1024)
            total_memory_mb = data_memory_mb + mask_memory_mb
            print(f"  数据内存: {data_memory_mb:.2f} MB")
            print(f"  Mask内存: {mask_memory_mb:.2f} MB")
            print(f"  总内存: {total_memory_mb:.2f} MB")
        else:
            print(f"  数据内存: {data_memory_mb:.2f} MB")
        
        if mmap_mode:
            print("  实际物理内存占用: 远小于上述值（按需加载）")
        
        if device is not None:
            print(f"  数据位置: {device}")
        else:
            print("  数据位置: CPU")
    
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
            如果return_mask=False:
                (X, y) 元组
        """
        X_sample = self.X[idx]
        y_sample = self.y[idx]
        
        if self.return_mask:
            if self.mask is not None:
                # 使用预计算的mask
                mask_sample = self.mask[idx]
            else:
                # 动态计算mask
                mask_sample = (X_sample != self.blank_value)
            
            return X_sample, y_sample, mask_sample
        else:
            return X_sample, y_sample
    
    def get_num_features(self) -> int:
        """获取特征数量"""
        return self.num_features
    
    def get_seq_len(self) -> int:
        """获取序列长度"""
        return self.seq_len
    
    def get_metadata(self) -> Dict:
        """获取元数据"""
        return self.metadata
    
    def get_feature_stats(self) -> Optional[Dict]:
        """获取特征统计量"""
        return self.feature_stats
    
    def to(self, device: str):
        """
        将数据移动到指定设备（仅非mmap模式）
        
        Args:
            device: 设备名称（'cpu' 或 'cuda'）
        """
        if self.mmap_mode:
            raise RuntimeError("内存映射模式不支持to()操作，数据必须保持在CPU")
        
        self.X = self.X.to(device)
        self.y = self.y.to(device)
        if self.mask is not None:
            self.mask = self.mask.to(device)
        print(f"数据已移动到: {device}")
        return self
