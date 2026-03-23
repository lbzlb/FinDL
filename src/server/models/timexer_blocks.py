#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TimeXer 基础模块
版本: v0.4
日期: 20260106

实现TimeXer的核心组件：
- 5层残差模块（复用）
- 精简时间混合块（共享）
- 精简特征混合块
- 精简TSMixer块
- 交叉注意力层（带FFN）
- 自注意力层（带FFN）
- 融合层
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# 导入RoPE
try:
    from .rope import RotaryPositionEmbedding
except ImportError as exc:
    # 如果相对导入失败，尝试绝对导入
    from pathlib import Path
    rope_path = Path(__file__).parent / "rope.py"
    if rope_path.exists():
        import importlib.util
        spec = importlib.util.spec_from_file_location("rope", rope_path)
        rope_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rope_module)
        RotaryPositionEmbedding = rope_module.RotaryPositionEmbedding
    else:
        raise ImportError(f"Cannot find rope.py at {rope_path}") from exc


class ResidualBlock(nn.Module):
    """
    5层残差模块
    
    结构：
    输入 -> Linear(D→D) + GELU -> ... (5层) -> 输出 + 输入（残差连接）
    
    特点：
    - 5层全连接，维度保持不变
    - 每层使用GELU激活
    - 最后一层输出与输入进行残差连接
    - 模块内部不使用Dropout
    """
    
    def __init__(self, dim: int, activation: str = "gelu"):
        """
        初始化5层残差模块
        
        Args:
            dim: 输入输出维度（保持不变）
            activation: 激活函数，"gelu"或"relu"
        """
        super(ResidualBlock, self).__init__()
        
        # 5层全连接
        self.layers = nn.ModuleList([
            nn.Linear(dim, dim) for _ in range(5)
        ])
        
        # 激活函数
        if activation.lower() == "gelu":
            self.activation = nn.GELU()
        elif activation.lower() == "relu":
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (..., dim)
        
        Returns:
            输出张量，形状为 (..., dim)
        """
        identity = x  # 保存输入用于残差连接
        
        # 5层前馈，每层都有激活
        for layer in self.layers:
            x = layer(x)
            x = self.activation(x)
        
        # 残差连接
        x = x + identity
        
        return x


class LightweightTimeMixingBlock(nn.Module):
    """
    精简时间混合块（共享版本）
    
    在时间维度上应用多层残差网络，用于捕捉复杂的时间依赖关系
    
    结构：
      seq_len → 256 (Linear + GELU) → res1(256) → res2(256) → seq_len (Linear + GELU + Dropout)
    """
    
    def __init__(
        self,
        seq_len: int,
        dropout: float = 0.1,
        activation: str = "gelu"
    ):
        """
        初始化精简时间混合块
        
        Args:
            seq_len: 序列长度
            dropout: Dropout比率（仅用于下降路径）
            activation: 激活函数，"gelu"或"relu"
        """
        super(LightweightTimeMixingBlock, self).__init__()
        
        # 激活函数
        if activation.lower() == "gelu":
            self.activation = nn.GELU()
        elif activation.lower() == "relu":
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # ===== 平直路径 =====
        # seq_len → 256
        self.up1 = nn.Linear(seq_len, 256)
        self.res1 = ResidualBlock(256, activation)
        
        # 256 → 256 (平直，无维度变化)
        self.res2 = ResidualBlock(256, activation)
        
        # 256 → seq_len
        self.down2 = nn.Linear(256, seq_len)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (batch, seq_len, n_features)
        
        Returns:
            输出张量，形状为 (batch, seq_len, n_features)
        """
        # x: (batch, seq_len, n_features)
        # 转置以便在时间维度上应用MLP
        x = x.transpose(1, 2)  # (batch, n_features, seq_len)
        
        # ===== 平直路径 =====
        # seq_len → 256
        x = self.up1(x)  # (batch, n_features, 256)
        x = self.activation(x)
        x = self.res1(x)  # ResidualBlock(256)
        
        # 256 → 256 (平直，无维度变化)
        x = self.res2(x)  # ResidualBlock(256)
        
        # 256 → seq_len
        x = self.down2(x)  # (batch, n_features, seq_len)
        x = self.activation(x)
        x = self.dropout2(x)
        
        # 转置回来
        x = x.transpose(1, 2)  # (batch, seq_len, n_features)
        
        return x


class TimeAttentionBlock(nn.Module):
    """
    时间注意力块（使用RoPE）
    
    在时间维度上应用自注意力机制，替代MLP时间混合
    使用RoPE位置编码来捕捉时间顺序
    """
    
    def __init__(
        self,
        seq_len: int,
        n_features: int,
        n_heads: int = 8,
        dropout: float = 0.1,
        activation: str = "gelu",
        use_rope: bool = True
    ):
        """
        初始化时间注意力块
        
        Args:
            seq_len: 序列长度
            n_features: 特征数量
            n_heads: 注意力头数（默认8）
            dropout: Dropout比率
            activation: 激活函数，"gelu"或"relu"
            use_rope: 是否使用RoPE位置编码（默认True）
        """
        super(TimeAttentionBlock, self).__init__()
        
        assert n_features % n_heads == 0, f"n_features ({n_features}) must be divisible by n_heads ({n_heads})"
        assert (n_features // n_heads) % 2 == 0, f"d_k ({n_features // n_heads}) must be even for RoPE"
        
        self.seq_len = seq_len
        self.n_features = n_features
        self.n_heads = n_heads
        self.d_k = n_features // n_heads
        self.use_rope = use_rope
        
        # 激活函数
        if activation.lower() == "gelu":
            self.activation = nn.GELU()
        elif activation.lower() == "relu":
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # 时间维度投影：将n_features投影到d_model（保持n_features不变）
        # 实际上不需要投影，直接在时间维度上应用注意力
        
        # Q, K, V投影（在特征维度上）
        self.W_q = nn.Linear(n_features, n_features)
        self.W_k = nn.Linear(n_features, n_features)
        self.W_v = nn.Linear(n_features, n_features)
        
        # 输出投影
        self.W_o = nn.Linear(n_features, n_features)
        
        # RoPE位置编码
        if use_rope:
            self.rope = RotaryPositionEmbedding(
                d_model=self.d_k,
                max_seq_len=seq_len,
                base=10000
            )
        
        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (batch, seq_len, n_features)
        
        Returns:
            输出张量，形状为 (batch, seq_len, n_features)
        """
        batch_size, seq_len, n_features = x.shape
        
        # 保存残差连接的输入
        residual = x
        
        # 线性投影并分离多头
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        # Q, K, V: (batch, n_heads, seq_len, d_k)
        
        # 应用RoPE位置编码
        if self.use_rope:
            # 生成位置索引：时间步索引 [0, 1, 2, ..., seq_len-1]
            position_ids = torch.arange(seq_len, device=x.device).unsqueeze(0)  # (1, seq_len)
            Q = self.rope(Q, position_ids)
            K = self.rope(K, position_ids)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        # scores: (batch, n_heads, seq_len, seq_len)
        
        # Softmax归一化
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # 计算注意力输出
        attn_output = torch.matmul(attn_weights, V)
        # attn_output: (batch, n_heads, seq_len, d_k)
        
        # 合并多头
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, n_features
        )
        
        # 输出投影
        output = self.W_o(attn_output)
        output = self.dropout(output)
        
        # 残差连接
        output = output + residual
        
        return output


class LightweightFeatureMixingBlock(nn.Module):
    """
    精简特征混合块
    
    在特征维度上应用多层残差网络，用于捕捉复杂的特征交互关系
    
    结构：
    上升路径（无dropout）：
      n_features → 64 (Linear + GELU) → res1(64) → 128 (Linear + GELU) → res2(128) 
      → 256 (Linear + GELU) → res3(256)
    
    下降路径（有dropout）：
      → 128 (Linear + GELU + Dropout) → res4(128) → 64 (Linear + GELU + Dropout) 
      → res5(64) → n_features (Linear + GELU + Dropout)
    """
    
    def __init__(
        self,
        n_features: int,
        dropout: float = 0.1,
        activation: str = "gelu"
    ):
        """
        初始化精简特征混合块
        
        Args:
            n_features: 特征数量
            dropout: Dropout比率（仅用于下降路径）
            activation: 激活函数，"gelu"或"relu"
        """
        super(LightweightFeatureMixingBlock, self).__init__()
        
        # 激活函数
        if activation.lower() == "gelu":
            self.activation = nn.GELU()
        elif activation.lower() == "relu":
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # ===== 上升路径（无dropout） =====
        # n_features → 64
        self.up1 = nn.Linear(n_features, 64)
        self.res1 = ResidualBlock(64, activation)
        
        # 64 → 128
        self.up2 = nn.Linear(64, 128)
        self.res2 = ResidualBlock(128, activation)
        
        # 128 → 256
        self.up3 = nn.Linear(128, 256)
        self.res3 = ResidualBlock(256, activation)
        
        # ===== 下降路径（有dropout） =====
        # 256 → 128
        self.down1 = nn.Linear(256, 128)
        self.dropout1 = nn.Dropout(dropout)
        self.res4 = ResidualBlock(128, activation)
        
        # 128 → 64
        self.down2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(dropout)
        self.res5 = ResidualBlock(64, activation)
        
        # 64 → n_features
        self.down3 = nn.Linear(64, n_features)
        self.dropout3 = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (batch, seq_len, n_features)
        
        Returns:
            输出张量，形状为 (batch, seq_len, n_features)
        """
        # x: (batch, seq_len, n_features)
        # 直接在特征维度上应用MLP（最后一个维度）
        
        # ===== 上升路径（无dropout） =====
        # n_features → 64
        x = self.up1(x)  # (batch, seq_len, 64)
        x = self.activation(x)
        x = self.res1(x)  # ResidualBlock(64)
        
        # 64 → 128
        x = self.up2(x)  # (batch, seq_len, 128)
        x = self.activation(x)
        x = self.res2(x)  # ResidualBlock(128)
        
        # 128 → 256
        x = self.up3(x)  # (batch, seq_len, 256)
        x = self.activation(x)
        x = self.res3(x)  # ResidualBlock(256)
        
        # ===== 下降路径（有dropout） =====
        # 256 → 128
        x = self.down1(x)  # (batch, seq_len, 128)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.res4(x)  # ResidualBlock(128)
        
        # 128 → 64
        x = self.down2(x)  # (batch, seq_len, 64)
        x = self.activation(x)
        x = self.dropout2(x)
        x = self.res5(x)  # ResidualBlock(64)
        
        # 64 → n_features
        x = self.down3(x)  # (batch, seq_len, n_features)
        x = self.activation(x)
        x = self.dropout3(x)
        
        return x


class LightweightTSMixerBlock(nn.Module):
    """
    精简TSMixer块
    
    组合时间混合和特征混合，支持可选的LayerNorm和残差连接
    结构：
    1. LayerNorm（可选） -> Time Mixing -> Residual（可选）
    2. LayerNorm（可选） -> Feature Mixing -> Residual（可选）
    """
    
    def __init__(
        self,
        seq_len: int,
        n_features: int,
        shared_time_mixing: Optional[nn.Module],
        dropout: float = 0.1,
        activation: str = "gelu",
        use_layernorm: bool = True,
        use_residual: bool = True,
        time_mixing_type: str = "mlp",  # "mlp" 或 "attention"
        time_attn_n_heads: int = 8,  # 当time_mixing_type="attention"时使用
        use_rope: bool = True  # 当time_mixing_type="attention"时使用
    ):
        """
        初始化精简TSMixer块
        
        Args:
            seq_len: 序列长度
            n_features: 特征数量
            shared_time_mixing: 共享的时间混合块（如果为None则创建独立的）
            dropout: Dropout比率
            activation: 激活函数
            use_layernorm: 是否使用LayerNorm（默认True）
            use_residual: 是否使用残差连接（默认True）
            time_mixing_type: 时间混合类型，"mlp"（默认）或"attention"
            time_attn_n_heads: 时间注意力头数（当time_mixing_type="attention"时使用）
            use_rope: 是否使用RoPE位置编码（当time_mixing_type="attention"时使用）
        """
        super(LightweightTSMixerBlock, self).__init__()
        
        self.use_layernorm = use_layernorm
        self.use_residual = use_residual
        self.time_mixing_type = time_mixing_type
        
        # LayerNorm（如果启用）
        if use_layernorm:
            self.norm1 = nn.LayerNorm(n_features)  # 时间混合前的归一化
            self.norm2 = nn.LayerNorm(n_features)  # 特征混合前的归一化
        
        # 时间混合（共享或独立）
        if shared_time_mixing is not None:
            self.time_mixing = shared_time_mixing
        else:
            if time_mixing_type == "attention":
                # 使用注意力机制（带RoPE）
                self.time_mixing = TimeAttentionBlock(
                    seq_len=seq_len,
                    n_features=n_features,
                    n_heads=time_attn_n_heads,
                    dropout=dropout,
                    activation=activation,
                    use_rope=use_rope
                )
            else:
                # 使用MLP（默认）
                self.time_mixing = LightweightTimeMixingBlock(
                    seq_len=seq_len,
                    dropout=dropout,
                    activation=activation
                )
        
        # 特征混合
        self.feature_mixing = LightweightFeatureMixingBlock(
            n_features=n_features,
            dropout=dropout,
            activation=activation
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (batch, seq_len, n_features)
        
        Returns:
            输出张量，形状为 (batch, seq_len, n_features)
        """
        # 时间混合
        if self.use_layernorm:
            residual = x if self.use_residual else None
            x = self.norm1(x)
            x = self.time_mixing(x)
            if self.use_residual:
                x = x + residual
        else:
            if self.use_residual:
                residual = x
                x = self.time_mixing(x)
                x = x + residual
            else:
                x = self.time_mixing(x)
        
        # 特征混合
        if self.use_layernorm:
            residual = x if self.use_residual else None
            x = self.norm2(x)
            x = self.feature_mixing(x)
            if self.use_residual:
                x = x + residual
        else:
            if self.use_residual:
                residual = x
                x = self.feature_mixing(x)
                x = x + residual
            else:
                x = self.feature_mixing(x)
        
        return x


class FFN(nn.Module):
    """
    前馈网络（带残差模块）
    
    结构：
    d_model → Linear → 激活 → 5层残差(d_model) → Linear(→ff_dim) → 激活 → Linear(→d_model) → 激活 → 5层残差(d_model)
    
    注意：ff_dim(512)层没有残差模块，只有d_model层有残差模块
    """
    
    def __init__(
        self,
        d_model: int,
        ff_dim: int = 512,
        dropout: float = 0.1,
        activation: str = "gelu"
    ):
        """
        初始化FFN
        
        Args:
            d_model: 模型维度
            ff_dim: 前馈网络隐藏维度（默认512）
            dropout: Dropout比率
            activation: 激活函数
        """
        super(FFN, self).__init__()
        
        # 激活函数
        if activation.lower() == "gelu":
            self.activation = nn.GELU()
        elif activation.lower() == "relu":
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # d_model → d_model (第一层投影，保持维度)
        self.proj1 = nn.Linear(d_model, d_model)
        self.res1 = ResidualBlock(d_model, activation)
        
        # d_model → ff_dim
        self.proj2 = nn.Linear(d_model, ff_dim)
        # ff_dim层没有残差模块
        
        # ff_dim → d_model
        self.proj3 = nn.Linear(ff_dim, d_model)
        self.res2 = ResidualBlock(d_model, activation)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (batch, ..., d_model)
        
        Returns:
            输出张量，形状为 (batch, ..., d_model)
        """
        # d_model → d_model → 激活 → 5层残差
        x = self.proj1(x)
        x = self.activation(x)
        x = self.res1(x)
        x = self.dropout(x)
        
        # d_model → ff_dim → 激活（没有残差）
        x = self.proj2(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        # ff_dim → d_model → 激活 → 5层残差
        x = self.proj3(x)
        x = self.activation(x)
        x = self.res2(x)
        x = self.dropout(x)
        
        return x


class CrossAttentionLayer(nn.Module):
    """
    交叉注意力层（带FFN）
    
    使用Query从一组特征中关注Key-Value对
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        ff_dim: int = 512,
        dropout: float = 0.1,
        activation: str = "gelu",
        use_layernorm: bool = True
    ):
        """
        初始化交叉注意力层
        
        Args:
            d_model: 模型维度
            n_heads: 注意力头数
            ff_dim: FFN隐藏维度
            dropout: Dropout比率
            activation: 激活函数
            use_layernorm: 是否使用LayerNorm
        """
        super(CrossAttentionLayer, self).__init__()
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.use_layernorm = use_layernorm
        
        # Q, K, V投影
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # 输出投影
        self.W_o = nn.Linear(d_model, d_model)
        
        self.attn_dropout = nn.Dropout(dropout)
        
        # LayerNorm
        if use_layernorm:
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
        
        # FFN
        self.ffn = FFN(d_model, ff_dim, dropout, activation)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            query: Query张量，形状为 (batch, len_q, d_model)
            key: Key张量，形状为 (batch, len_k, d_model)
            value: Value张量，形状为 (batch, len_v, d_model)
            mask: 掩码（可选）
        
        Returns:
            注意力输出，形状为 (batch, len_q, d_model)
        """
        batch_size, len_q, d_model = query.shape
        
        # Pre-Norm（如果启用）
        if self.use_layernorm:
            query_norm = self.norm1(query)
        else:
            query_norm = query
        
        # 线性投影并分离多头
        Q = self.W_q(query_norm).view(batch_size, len_q, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        # Q, K, V: (batch, n_heads, len, d_k)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        # scores: (batch, n_heads, len_q, len_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax归一化
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # 计算注意力输出
        attn_output = torch.matmul(attn_weights, V)
        # attn_output: (batch, n_heads, len_q, d_k)
        
        # 合并多头
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, len_q, d_model
        )
        
        # 输出投影
        output = self.W_o(attn_output)
        output = self.attn_dropout(output)
        
        # 残差连接
        output = output + query
        
        # FFN + 残差
        if self.use_layernorm:
            output = output + self.ffn(self.norm2(output))
        else:
            output = output + self.ffn(output)
        
        return output


class SelfAttentionLayer(nn.Module):
    """
    自注意力层（带FFN）
    
    对输入特征做自注意力
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        ff_dim: int = 512,
        dropout: float = 0.1,
        activation: str = "gelu",
        use_layernorm: bool = True
    ):
        """
        初始化自注意力层
        
        Args:
            d_model: 模型维度
            n_heads: 注意力头数
            ff_dim: FFN隐藏维度
            dropout: Dropout比率
            activation: 激活函数
            use_layernorm: 是否使用LayerNorm
        """
        super(SelfAttentionLayer, self).__init__()
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.use_layernorm = use_layernorm
        
        # Q, K, V投影
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # 输出投影
        self.W_o = nn.Linear(d_model, d_model)
        
        self.attn_dropout = nn.Dropout(dropout)
        
        # LayerNorm
        if use_layernorm:
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
        
        # FFN
        self.ffn = FFN(d_model, ff_dim, dropout, activation)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (batch, len, d_model)
            mask: 掩码（可选）
        
        Returns:
            注意力输出，形状为 (batch, len, d_model)
        """
        batch_size, len_seq, d_model = x.shape
        
        # Pre-Norm（如果启用）
        if self.use_layernorm:
            x_norm = self.norm1(x)
        else:
            x_norm = x
        
        # 线性投影并分离多头
        Q = self.W_q(x_norm).view(batch_size, len_seq, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x_norm).view(batch_size, len_seq, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x_norm).view(batch_size, len_seq, self.n_heads, self.d_k).transpose(1, 2)
        # Q, K, V: (batch, n_heads, len, d_k)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        # scores: (batch, n_heads, len, len)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax归一化
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # 计算注意力输出
        attn_output = torch.matmul(attn_weights, V)
        # attn_output: (batch, n_heads, len, d_k)
        
        # 合并多头
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, len_seq, d_model
        )
        
        # 输出投影
        output = self.W_o(attn_output)
        output = self.attn_dropout(output)
        
        # 残差连接
        output = output + x
        
        # FFN + 残差
        if self.use_layernorm:
            output = output + self.ffn(self.norm2(output))
        else:
            output = output + self.ffn(output)
        
        return output
