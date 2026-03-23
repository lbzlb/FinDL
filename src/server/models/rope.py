#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
旋转位置编码 (Rotary Position Embedding, RoPE)
版本: v0.42
日期: 20260118

实现RoPE位置编码，用于相对位置编码
（从v0.4复制，无需修改）
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class RotaryPositionEmbedding(nn.Module):
    """
    旋转位置编码 (RoPE)
    
    通过旋转矩阵对Query和Key进行位置编码，实现相对位置编码
    """
    
    def __init__(
        self,
        d_model: int,
        max_seq_len: int = 5000,
        base: int = 10000
    ):
        """
        初始化RoPE
        
        Args:
            d_model: 模型维度（必须是偶数）
            max_seq_len: 最大序列长度
            base: 旋转基数（默认10000）
        """
        super(RotaryPositionEmbedding, self).__init__()
        
        assert d_model % 2 == 0, f"d_model ({d_model}) must be even for RoPE"
        
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.base = base
        
        # 计算频率
        inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer('inv_freq', inv_freq)
        
        # 预计算cos和sin缓存
        self._build_cache(max_seq_len)
    
    def _build_cache(self, max_len: int):
        """构建cos和sin缓存"""
        t = torch.arange(max_len, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        # 拼接cos和sin
        emb = torch.cat([freqs, freqs], dim=-1)
        cos_cache = emb.cos()[None, None, :, :]  # (1, 1, max_len, d_model)
        sin_cache = emb.sin()[None, None, :, :]  # (1, 1, max_len, d_model)
        self.register_buffer('cos_cache', cos_cache)
        self.register_buffer('sin_cache', sin_cache)
    
    def forward(
        self,
        x: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        应用旋转位置编码
        
        Args:
            x: 输入张量，形状为 (batch, n_heads, seq_len, d_k)
            position_ids: 位置索引，形状为 (batch, seq_len) 或 (1, seq_len)
                         如果为None，则使用默认位置 [0, 1, 2, ..., seq_len-1]
        
        Returns:
            旋转后的张量，形状与输入相同
        """
        batch_size, n_heads, seq_len, d_k = x.shape
        
        # 如果没有传入位置索引，使用默认位置
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=x.device).unsqueeze(0)  # (1, seq_len)
        
        # 确保position_ids形状正确
        if position_ids.dim() == 1:
            position_ids = position_ids.unsqueeze(0)  # (1, seq_len)
        
        # 从缓存中提取对应的cos和sin
        # 如果position_ids是连续的[0,1,2,...]，直接使用缓存的前seq_len个
        # 否则，根据position_ids从缓存中提取对应的位置
        
        # 检查position_ids是否是连续的
        default_pos = torch.arange(seq_len, device=x.device).unsqueeze(0)
        is_continuous = (position_ids.shape[0] == 1 and 
                        torch.allclose(position_ids.squeeze(0), default_pos.squeeze(0)))
        
        # RoPE只需要d_k//2维，因为x会被分成两部分
        d_half = d_k // 2
        
        if is_continuous:
            # 连续位置，直接使用缓存（只取前d_k//2维）
            cos = self.cos_cache[:, :, :seq_len, :d_half].expand(batch_size, n_heads, seq_len, d_half)
            sin = self.sin_cache[:, :, :seq_len, :d_half].expand(batch_size, n_heads, seq_len, d_half)
        else:
            # 非连续位置，需要根据position_ids索引
            # position_ids: (batch, seq_len) -> (batch, 1, seq_len, 1)
            pos_ids_expanded = position_ids.unsqueeze(1).unsqueeze(-1)  # (batch, 1, seq_len, 1)
            # 扩展维度以匹配缓存形状（最后一个维度是d_half）
            pos_ids_for_gather = pos_ids_expanded.expand(batch_size, n_heads, seq_len, d_half)
            # 从缓存中提取（只取前d_k//2维）
            cos = torch.gather(
                self.cos_cache[:, :, :, :d_half].expand(batch_size, n_heads, self.max_seq_len, d_half),
                dim=2,
                index=pos_ids_for_gather
            )
            sin = torch.gather(
                self.sin_cache[:, :, :, :d_half].expand(batch_size, n_heads, self.max_seq_len, d_half),
                dim=2,
                index=pos_ids_for_gather
            )
        
        # 将x分成两部分（实部和虚部）
        x1, x2 = x.chunk(2, dim=-1)  # 每个都是 (batch, n_heads, seq_len, d_k//2)
        
        # 应用旋转：旋转后的x = [x1*cos - x2*sin, x1*sin + x2*cos]
        rotated_x = torch.cat([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos
        ], dim=-1)
        
        return rotated_x
