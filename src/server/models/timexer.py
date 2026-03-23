#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TimeXer 主模型（学习型Missing Embedding + Instance Normalization）
版本: v0.45
日期: 20260207

实现TimeXer模型，用于多变量时间序列预测
采用双分支架构：内生分支 + 宏观分支，通过交叉注意力融合

v0.45新增：Instance Normalization + 反归一化（基于v0.43）
关键改进：
1. 学习型Missing Embedding（继承自v0.43）
2. Instance Normalization：模型内部对输入进行归一化（每个样本独立）
3. 反归一化：输出前将预测值还原到原始尺度
4. 损失计算在原始尺度上进行，更符合业务含义
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List
from pathlib import Path
import importlib.util


# 动态导入TimeXerBlock
def _load_module(module_path: Path, module_name: str):
    """动态加载模块"""
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# 导入TimeXerBlock和相关组件
models_path = Path(__file__).parent
try:
    blocks_module = _load_module(models_path / "timexer_blocks.py", "timexer_blocks")
    LightweightTSMixerBlock = blocks_module.LightweightTSMixerBlock
    LightweightTimeMixingBlock = blocks_module.LightweightTimeMixingBlock
    TimeAttentionBlock = blocks_module.TimeAttentionBlock
    ResidualBlock = blocks_module.ResidualBlock
    CrossAttentionLayer = blocks_module.CrossAttentionLayer
    SelfAttentionLayer = blocks_module.SelfAttentionLayer
except Exception as e:
    raise ImportError(f"Failed to import TimeXer blocks: {e}") from e


class TimeXer(nn.Module):
    """
    TimeXer模型（学习型Missing Embedding + Instance Normalization）
    
    用于多变量时间序列预测的深度神经网络
    采用双分支架构：内生分支处理公司内部数据，宏观分支处理宏观指标
    
    模型结构：
    1. 输入: (batch, seq_len, n_features) = (batch, 500, 64)，可能包含-1000标记的缺失值
    2. **立即替换缺失值为可学习embedding**（v0.43）
    3. **Instance Normalization**（v0.45新增）：对输入进行归一化
    4. 分离: 内生 [batch, 500, 44] + 宏观 [batch, 500, 20]
    5. 共享时间混合层（处理500时间步）
    6. 内生分支: 3个LightweightTSMixerBlock → 时间聚合 → [batch, 256]
    7. 宏观分支: 2个LightweightTSMixerBlock → 时间聚合 → [batch, 256]
    8. 交叉注意力融合层（3层）
    9. 输出投影: 多层残差降维 → [batch, prediction_len]
    10. **反归一化**（v0.45新增）：将输出还原到原始尺度
    
    v0.45新增：Instance Normalization + 反归一化（方案B：masked 位置不参与）
    - 在处理缺失值后对输入进行Instance Normalization
    - 均值和标准差仅基于有效位置（mask=True）计算，归一化仅作用于有效位置，masked 位置保持 missing_emb
    - 保存均值和标准差用于反归一化
    - 输出前使用索引2（"收盘"）的统计量进行反归一化
    - 损失计算在原始尺度上进行
    """
    
    def __init__(
        self,
        seq_len: int,
        n_features: int,
        endogenous_features: int = 44,
        exogenous_features: int = 20,
        prediction_len: int = 1,
        # 特征分离配置（使用索引列表，优先级高于数量）
        endogenous_indices: Optional[List[int]] = None,
        exogenous_indices: Optional[List[int]] = None,
        # 内生分支配置
        endogenous_blocks: int = 3,
        endogenous_hidden_dim: int = 256,
        # 宏观分支配置
        exogenous_blocks: int = 2,
        exogenous_hidden_dim: int = 256,
        # 共享时间混合配置
        shared_time_mixing: bool = True,
        time_mixing_type: str = "attention",  # "mlp" 或 "attention"
        time_attn_n_heads: int = 8,  # 当time_mixing_type="attention"时使用
        use_rope: bool = True,  # 当time_mixing_type="attention"时使用
        # 交叉注意力配置
        cross_attn_n_heads: int = 8,
        cross_attn_ff_dim: int = 512,
        # 通用配置
        dropout: float = 0.1,
        activation: str = "gelu",
        use_layernorm: bool = True,  # 全局开关，如果为False则全部禁用
        use_residual: bool = True,
        # v0.41细粒度LayerNorm控制
        use_layernorm_in_tsmixer: bool = None,  # TSMixerBlock中是否使用（None则使用use_layernorm）
        use_layernorm_in_attention: bool = None,  # 注意力层中是否使用（None则使用use_layernorm）
        use_layernorm_before_pooling: bool = None,  # 时间聚合前是否使用（None则使用use_layernorm）
        # 为了兼容训练脚本的配置加载，接受额外的配置参数
        n_blocks: int = None,  # 保留以兼容，但不使用
        ff_dim: int = None,  # 保留以兼容，但不使用
        norm_type: str = "layer",  # 保留以兼容
        temporal_aggregation_config: Optional[Dict] = None,
        output_projection_config: Optional[Dict] = None,
        # v0.45新增：Instance Normalization参数
        use_norm: bool = True,  # 是否使用Instance Normalization
        norm_feature_indices: Optional[List[int]] = None,  # 需要归一化的特征索引（None=全部）
        output_feature_index: int = 2  # 输出对应的特征索引（用于反归一化）
    ):
        """
        初始化TimeXer模型
        
        Args:
            seq_len: 输入序列长度
            n_features: 总特征数量（内生+宏观）
            endogenous_features: 内生特征数量（默认44，如果提供了endogenous_indices则忽略）
            exogenous_features: 宏观特征数量（默认20，如果提供了exogenous_indices则忽略）
            prediction_len: 预测长度（输出维度）
            endogenous_indices: 内生特征位置索引列表（可选，优先级高于endogenous_features）
            exogenous_indices: 宏观特征位置索引列表（可选，优先级高于exogenous_features）
            endogenous_blocks: 内生分支TSMixer块数量（默认3）
            endogenous_hidden_dim: 内生分支隐藏维度（默认256）
            exogenous_blocks: 宏观分支TSMixer块数量（默认2）
            exogenous_hidden_dim: 宏观分支隐藏维度（默认256）
            shared_time_mixing: 是否共享时间混合层（默认True）
            time_mixing_type: 时间混合类型，"mlp"或"attention"
            time_attn_n_heads: 时间注意力头数（当time_mixing_type="attention"时使用，默认8）
            use_rope: 是否使用RoPE位置编码（当time_mixing_type="attention"时使用，默认True）
            cross_attn_n_heads: 交叉注意力头数（默认8）
            cross_attn_ff_dim: 交叉注意力FFN维度（默认512）
            dropout: Dropout比率
            activation: 激活函数，"gelu"或"relu"
            use_layernorm: 全局LayerNorm开关（默认True），如果为False则全部禁用
            use_residual: 是否启用残差连接（默认True）
            use_layernorm_in_tsmixer: TSMixerBlock中是否使用LayerNorm（None则使用use_layernorm）
            use_layernorm_in_attention: 注意力层中是否使用LayerNorm（None则使用use_layernorm）
            use_layernorm_before_pooling: 时间聚合前是否使用LayerNorm（None则使用use_layernorm）
            n_blocks: TSMixer块数量（保留以兼容，不使用）
            ff_dim: 前馈网络隐藏维度（保留以兼容，不使用）
            norm_type: 归一化类型（保留以兼容）
            temporal_aggregation_config: 时间聚合配置（可选）
            output_projection_config: 输出投影配置（可选）
            use_norm: 是否使用Instance Normalization（v0.45新增，默认True）
            norm_feature_indices: 需要归一化的特征索引列表（None=全部特征，v0.45新增）
            output_feature_index: 输出对应的特征索引，用于反归一化（默认2="收盘"，v0.45新增）
        """
        super(TimeXer, self).__init__()
        
        # 如果提供了索引列表，使用索引列表；否则使用数量
        if endogenous_indices is not None and exogenous_indices is not None:
            # 使用索引列表方式
            # 使用register_buffer注册索引，PyTorch会自动处理设备移动
            self.register_buffer('endogenous_indices', torch.tensor(endogenous_indices, dtype=torch.long))
            self.register_buffer('exogenous_indices', torch.tensor(exogenous_indices, dtype=torch.long))
            self.use_indices = True
            
            # 从索引列表计算特征数量
            endogenous_features = len(endogenous_indices)
            exogenous_features = len(exogenous_indices)
            
            # 验证索引范围
            all_indices = set(endogenous_indices) | set(exogenous_indices)
            if len(all_indices) != len(endogenous_indices) + len(exogenous_indices):
                raise ValueError(f"内生索引和宏观索引有重叠")
            if max(all_indices) >= n_features:
                raise ValueError(f"索引超出范围: 最大索引={max(all_indices)}, 总特征数={n_features}")
            if min(all_indices) < 0:
                raise ValueError(f"索引不能为负数: 最小索引={min(all_indices)}")
            
            # 验证索引数量
            if len(all_indices) != n_features:
                raise ValueError(f"索引数量({len(all_indices)}) != 总特征数({n_features})")
        else:
            # 使用数量方式（向后兼容）
            self.endogenous_indices = None
            self.exogenous_indices = None
            self.use_indices = False
            
            # 验证特征数量
            assert endogenous_features + exogenous_features == n_features, \
                f"内生特征({endogenous_features}) + 宏观特征({exogenous_features}) != 总特征({n_features})"
        
        self.seq_len = seq_len
        self.n_features = n_features
        self.endogenous_features = endogenous_features
        self.exogenous_features = exogenous_features
        self.prediction_len = prediction_len
        self.endogenous_blocks = endogenous_blocks
        self.exogenous_blocks = exogenous_blocks
        self.endogenous_hidden_dim = endogenous_hidden_dim
        self.exogenous_hidden_dim = exogenous_hidden_dim
        self.dropout = dropout
        self.activation_name = activation
        self.use_layernorm = use_layernorm
        self.use_residual = use_residual
        self.norm_type = norm_type
        
        # v0.41新增：细粒度LayerNorm控制
        # 如果全局use_layernorm=False，则全部禁用
        # 如果全局use_layernorm=True，则根据细粒度设置决定
        if not use_layernorm:
            self.use_layernorm_in_tsmixer = False
            self.use_layernorm_in_attention = False
            self.use_layernorm_before_pooling = False
        else:
            # 如果细粒度参数为None，使用全局设置
            self.use_layernorm_in_tsmixer = use_layernorm if use_layernorm_in_tsmixer is None else use_layernorm_in_tsmixer
            self.use_layernorm_in_attention = use_layernorm if use_layernorm_in_attention is None else use_layernorm_in_attention
            self.use_layernorm_before_pooling = use_layernorm if use_layernorm_before_pooling is None else use_layernorm_before_pooling
        
        # 激活函数
        if activation.lower() == "gelu":
            self.activation = nn.GELU()
        elif activation.lower() == "relu":
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # ========== v0.43：学习型Missing Embedding ==========
        # 可学习的缺失值表示，初始化为小值（0.01量级）
        # 形状：(1, 1, n_features)，所有batch和时间步共享
        self.missing_embedding = nn.Parameter(
            torch.randn(1, 1, n_features) * 0.01
        )
        # 缺失值标记（用于检测）
        self.missing_value_flag = -1000.0
        
        # ========== v0.45新增：Instance Normalization 配置 ==========
        self.use_norm = use_norm
        self.output_feature_index = output_feature_index
        
        # 验证输出特征索引
        assert 0 <= output_feature_index < n_features, \
            f"output_feature_index ({output_feature_index}) 超出范围 [0, {n_features})"
        
        # 创建归一化特征掩码
        if norm_feature_indices is not None:
            # 只对指定索引的特征进行归一化
            norm_mask = torch.zeros(n_features, dtype=torch.bool)
            for idx in norm_feature_indices:
                assert 0 <= idx < n_features, \
                    f"norm_feature_indices中的索引 {idx} 超出范围 [0, {n_features})"
                norm_mask[idx] = True
            self.register_buffer('norm_mask', norm_mask)
        else:
            # 默认对所有特征归一化
            self.register_buffer('norm_mask', torch.ones(n_features, dtype=torch.bool))
        
        # ========== 共享时间混合层 ==========
        # 统一使用48维（能被8整除，且d_k=6是偶数，符合RoPE要求）
        self.time_mixing_dim = 48
        self.shared_time_mixing = None
        if shared_time_mixing:
            if time_mixing_type == "attention":
                # 使用注意力机制（带RoPE），统一使用48维
                self.shared_time_mixing = TimeAttentionBlock(
                    seq_len=seq_len,
                    n_features=self.time_mixing_dim,  # 统一使用48维
                    n_heads=time_attn_n_heads,
                    dropout=dropout,
                    activation=activation,
                    use_rope=use_rope
                )
            else:
                # 使用MLP（默认）
                self.shared_time_mixing = LightweightTimeMixingBlock(
                    seq_len=seq_len,
                    dropout=dropout,
                    activation=activation
                )
        
        # ========== 时间混合投影层（统一投影到48维） ==========
        # 内生分支投影：44 → 48 → 44
        self.endogenous_time_proj_up = nn.Linear(endogenous_features, self.time_mixing_dim)
        self.endogenous_time_proj_down = nn.Linear(self.time_mixing_dim, endogenous_features)
        
        # 宏观分支投影：20 → 48 → 20
        self.exogenous_time_proj_up = nn.Linear(exogenous_features, self.time_mixing_dim)
        self.exogenous_time_proj_down = nn.Linear(self.time_mixing_dim, exogenous_features)
        
        # ========== 内生分支 ==========
        # 如果使用共享时间混合层，特征混合层也使用统一维度（48）
        # 否则使用原始特征维度（44）
        endogenous_mixer_n_features = self.time_mixing_dim if shared_time_mixing else endogenous_features
        self.endogenous_mixer_blocks = nn.ModuleList([
            LightweightTSMixerBlock(
                seq_len=seq_len,
                n_features=endogenous_mixer_n_features,
                shared_time_mixing=self.shared_time_mixing,
                dropout=dropout,
                activation=activation,
                use_layernorm=self.use_layernorm_in_tsmixer,  # 使用细粒度控制
                use_residual=use_residual,
                time_mixing_type=time_mixing_type,
                time_attn_n_heads=time_attn_n_heads,
                use_rope=use_rope
            )
            for _ in range(endogenous_blocks)
        ])
        
        # 内生分支时间聚合前的LayerNorm（根据细粒度设置决定是否创建）
        self.endogenous_norm = None
        if self.use_layernorm_before_pooling:
            self.endogenous_norm = nn.LayerNorm(endogenous_features)
        
        # 内生分支投影到hidden_dim
        self.endogenous_proj = nn.Linear(endogenous_features, endogenous_hidden_dim)
        
        # ========== 宏观分支 ==========
        # 如果使用共享时间混合层，特征混合层也使用统一维度（48）
        # 否则使用原始特征维度（20）
        exogenous_mixer_n_features = self.time_mixing_dim if shared_time_mixing else exogenous_features
        self.exogenous_mixer_blocks = nn.ModuleList([
            LightweightTSMixerBlock(
                seq_len=seq_len,
                n_features=exogenous_mixer_n_features,
                shared_time_mixing=self.shared_time_mixing,
                dropout=dropout,
                activation=activation,
                use_layernorm=self.use_layernorm_in_tsmixer,  # 使用细粒度控制
                use_residual=use_residual,
                time_mixing_type=time_mixing_type,
                time_attn_n_heads=time_attn_n_heads,
                use_rope=use_rope
            )
            for _ in range(exogenous_blocks)
        ])
        
        # 宏观分支时间聚合前的LayerNorm（根据细粒度设置决定是否创建）
        self.exogenous_norm = None
        if self.use_layernorm_before_pooling:
            self.exogenous_norm = nn.LayerNorm(exogenous_features)
        
        # 宏观分支投影到hidden_dim
        self.exogenous_proj = nn.Linear(exogenous_features, exogenous_hidden_dim)
        
        # ========== 交叉注意力融合层 ==========
        # 第一层：内生(Query) × 宏观(Key/Value)
        self.cross_attn1 = CrossAttentionLayer(
            d_model=endogenous_hidden_dim,
            n_heads=cross_attn_n_heads,
            ff_dim=cross_attn_ff_dim,
            dropout=dropout,
            activation=activation,
            use_layernorm=self.use_layernorm_in_attention  # 使用细粒度控制
        )
        
        # 需要将宏观特征投影到内生维度（带激活和残差）
        self.exogenous_to_endogenous_proj = nn.Linear(exogenous_hidden_dim, endogenous_hidden_dim)
        self.exogenous_to_endogenous_res = ResidualBlock(endogenous_hidden_dim, activation)
        
        # 第二层：自注意力（增强后的内生特征）
        self.self_attn = SelfAttentionLayer(
            d_model=endogenous_hidden_dim,
            n_heads=cross_attn_n_heads,
            ff_dim=cross_attn_ff_dim,
            dropout=dropout,
            activation=activation,
            use_layernorm=self.use_layernorm_in_attention  # 使用细粒度控制
        )
        
        # 第三层：宏观(Query) × 增强内生(Key/Value)
        self.cross_attn2 = CrossAttentionLayer(
            d_model=exogenous_hidden_dim,
            n_heads=cross_attn_n_heads,
            ff_dim=cross_attn_ff_dim,
            dropout=dropout,
            activation=activation,
            use_layernorm=self.use_layernorm_in_attention  # 使用细粒度控制
        )
        
        # 需要将增强内生特征投影到宏观维度（带激活和残差）
        self.endogenous_to_exogenous_proj = nn.Linear(endogenous_hidden_dim, exogenous_hidden_dim)
        self.endogenous_to_exogenous_res = ResidualBlock(exogenous_hidden_dim, activation)
        
        # ========== 融合策略 ==========
        # 拼接两个增强特征
        fusion_dim = endogenous_hidden_dim + exogenous_hidden_dim
        self.fusion_proj = nn.Linear(fusion_dim, endogenous_hidden_dim)  # 融合到内生维度
        
        # ========== 输出投影（多层残差降维） ==========
        # 结构：endogenous_hidden_dim → 64维残差 → 32维 → prediction_len（直接线性输出）
        
        # endogenous_hidden_dim维度的5层残差模块
        self.output_res1 = ResidualBlock(endogenous_hidden_dim, activation)
        
        # endogenous_hidden_dim → 64
        self.output_proj1 = nn.Linear(endogenous_hidden_dim, 64)
        self.output_dropout1 = nn.Dropout(dropout)
        
        # 64维度的5层残差模块
        self.output_res2 = ResidualBlock(64, activation)
        
        # 64 → 32
        self.output_proj2 = nn.Linear(64, 32)
        self.output_dropout2 = nn.Dropout(dropout)
        
        # 32 → prediction_len（直接线性变换，无激活函数，无残差模块）
        self.output_proj3 = nn.Linear(32, prediction_len)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播（学习型Missing Embedding + Instance Normalization）
        
        Args:
            x: 输入张量，形状为 (batch, seq_len, n_features)
               可能包含-1000标记的缺失值
            mask: 掩码张量（可选，保留以兼容）
                  如果为None，自动从x中检测-1000
        
        Returns:
            输出张量，形状为 (batch, prediction_len)，已反归一化到原始尺度
            
        注意：为了兼容训练脚本，当prediction_len=1时，输出形状为(batch, 1)
        
        **重要**：缺失值处理和归一化必须在第一时间完成，在任何线性层之前！
        """
        # x: (batch, seq_len, n_features) = (batch, 500, 64)
        
        # ========== Step 1: 处理缺失值（必须在第一时间完成）==========
        # 自动检测-1000标记的缺失值（精确匹配）
        if mask is None:
            mask = (x != self.missing_value_flag)  # True=有效数据, False=缺失数据(-1000)
        
        # 将缺失值替换为可学习的embedding
        # missing_emb: (1, 1, 64) → 广播到 (batch, 500, 64)
        missing_emb = self.missing_embedding.expand_as(x)
        
        # torch.where可导，保持梯度传播
        # 如果mask=True使用原始值x，否则使用missing_emb
        x = torch.where(mask, x, missing_emb)
        
        # 现在x中不再有-1000，全是正常值范围的数据
        
        # ========== Step 2: v0.45新增 - Instance Normalization（可选，方案B：仅有效位置参与统计与归一化）==========
        if self.use_norm:
            B, T, F = x.shape
            # 初始化：先复制原始数据
            x_normed = x.clone()
            # 初始化均值和标准差（所有特征）
            means = torch.zeros(B, 1, F, device=x.device)
            stdev = torch.ones(B, 1, F, device=x.device)
            
            # 只对指定特征进行归一化
            norm_indices = self.norm_mask.nonzero(as_tuple=True)[0]
            if len(norm_indices) > 0:
                # 提取需要归一化的特征子集及对应 mask
                x_norm_subset = x[:, :, norm_indices]  # [B, T, num_norm_features]
                mask_subset = mask[:, :, norm_indices].float()  # [B, T, num_norm_features]，True=有效
                
                # 仅对有效位置（mask=True）计算均值和标准差（masked statistics）
                count = mask_subset.sum(dim=1, keepdim=True).clamp(min=1)  # [B, 1, num_norm_features]
                sum_x = (x_norm_subset * mask_subset).sum(dim=1, keepdim=True)
                means_subset = (sum_x / count).detach()  # [B, 1, num_norm_features]
                x_centered = x_norm_subset - means_subset
                sum_sq = ((x_centered ** 2) * mask_subset).sum(dim=1, keepdim=True)
                var_subset = sum_sq / count
                stdev_subset = torch.sqrt(var_subset + 1e-5).detach()  # [B, 1, num_norm_features]
                
                # 方案B：仅对有效位置做归一化，masked 位置保持原值（missing_emb）
                x_normed_subset = (x_norm_subset - means_subset) / stdev_subset
                x_normed_subset = torch.where(
                    mask_subset.to(torch.bool), x_normed_subset, x_norm_subset
                )
                
                # 将归一化后的值写回
                x_normed[:, :, norm_indices] = x_normed_subset
                
                # 保存均值和标准差（用于反归一化）
                means[:, :, norm_indices] = means_subset
                stdev[:, :, norm_indices] = stdev_subset
            
            x = x_normed
        else:
            means = None
            stdev = None
        
        # ========== Step 3: 分离内生和宏观特征 ==========
        if self.use_indices:
            # 使用索引列表方式分离特征
            # register_buffer会自动处理设备移动，无需手动处理
            endogenous_x = x[:, :, self.endogenous_indices]  # (batch, 500, endogenous_features)
            exogenous_x = x[:, :, self.exogenous_indices]  # (batch, 500, exogenous_features)
        else:
            # 使用数量方式分离特征（向后兼容）
            endogenous_x = x[:, :, :self.endogenous_features]  # (batch, 500, 44)
            exogenous_x = x[:, :, self.endogenous_features:]  # (batch, 500, 20)
        
        # ========== 内生分支 ==========
        # 投影到统一时间混合维度（44 → 48）
        endogenous_x = self.endogenous_time_proj_up(endogenous_x)  # (batch, 500, 48)
        
        for mixer_block in self.endogenous_mixer_blocks:
            endogenous_x = mixer_block(endogenous_x)  # (batch, 500, 48)
        
        # 投影回原始维度（48 → 44）
        endogenous_x = self.endogenous_time_proj_down(endogenous_x)  # (batch, 500, 44)
        
        # 时间聚合前的LayerNorm（如果存在）
        if self.endogenous_norm is not None:
            endogenous_x = self.endogenous_norm(endogenous_x)
        
        # 时间聚合（平均池化）
        endogenous_x = endogenous_x.mean(dim=1)  # (batch, 44)
        
        # 投影到hidden_dim
        endogenous_x = self.endogenous_proj(endogenous_x)  # (batch, 256)
        endogenous_x = self.activation(endogenous_x)
        
        # ========== 宏观分支 ==========
        # 投影到统一时间混合维度（20 → 48）
        exogenous_x = self.exogenous_time_proj_up(exogenous_x)  # (batch, 500, 48)
        
        for mixer_block in self.exogenous_mixer_blocks:
            exogenous_x = mixer_block(exogenous_x)  # (batch, 500, 48)
        
        # 投影回原始维度（48 → 20）
        exogenous_x = self.exogenous_time_proj_down(exogenous_x)  # (batch, 500, 20)
        
        # 时间聚合前的LayerNorm（如果存在）
        if self.exogenous_norm is not None:
            exogenous_x = self.exogenous_norm(exogenous_x)
        
        # 时间聚合（平均池化）
        exogenous_x = exogenous_x.mean(dim=1)  # (batch, 20)
        
        # 投影到hidden_dim
        exogenous_x = self.exogenous_proj(exogenous_x)  # (batch, 256)
        exogenous_x = self.activation(exogenous_x)
        
        # ========== 交叉注意力融合层 ==========
        # 第一层：内生(Query) × 宏观(Key/Value)
        # 统一处理：两个特征都经过投影+激活+残差
        endogenous_for_attn1 = self.exogenous_to_endogenous_proj(endogenous_x)  # (batch, 256)
        endogenous_for_attn1 = self.activation(endogenous_for_attn1)  # 激活
        endogenous_for_attn1 = self.exogenous_to_endogenous_res(endogenous_for_attn1)  # 残差
        endogenous_for_attn1 = endogenous_for_attn1.unsqueeze(1)  # (batch, 1, 256)
        
        exogenous_for_attn1 = self.exogenous_to_endogenous_proj(exogenous_x)  # (batch, 256)
        exogenous_for_attn1 = self.activation(exogenous_for_attn1)  # 激活
        exogenous_for_attn1 = self.exogenous_to_endogenous_res(exogenous_for_attn1)  # 残差
        exogenous_for_attn1 = exogenous_for_attn1.unsqueeze(1)  # (batch, 1, 256)
        
        enhanced_endogenous = self.cross_attn1(
            query=endogenous_for_attn1,
            key=exogenous_for_attn1,
            value=exogenous_for_attn1
        )  # (batch, 1, 256)
        enhanced_endogenous = enhanced_endogenous.squeeze(1)  # (batch, 256)
        
        # 第二层：自注意力（增强后的内生特征）
        enhanced_endogenous = enhanced_endogenous.unsqueeze(1)  # (batch, 1, 256)
        enhanced_endogenous = self.self_attn(enhanced_endogenous)  # (batch, 1, 256)
        enhanced_endogenous = enhanced_endogenous.squeeze(1)  # (batch, 256)
        
        # 第三层：宏观(Query) × 增强内生(Key/Value)
        # 统一处理：两个特征都经过投影+激活+残差
        exogenous_for_attn2 = self.endogenous_to_exogenous_proj(exogenous_x)  # (batch, 256)
        exogenous_for_attn2 = self.activation(exogenous_for_attn2)  # 激活
        exogenous_for_attn2 = self.endogenous_to_exogenous_res(exogenous_for_attn2)  # 残差
        exogenous_for_attn2 = exogenous_for_attn2.unsqueeze(1)  # (batch, 1, 256)
        
        endogenous_for_attn2 = self.endogenous_to_exogenous_proj(enhanced_endogenous)  # (batch, 256)
        endogenous_for_attn2 = self.activation(endogenous_for_attn2)  # 激活
        endogenous_for_attn2 = self.endogenous_to_exogenous_res(endogenous_for_attn2)  # 残差
        endogenous_for_attn2 = endogenous_for_attn2.unsqueeze(1)  # (batch, 1, 256)
        
        enhanced_exogenous = self.cross_attn2(
            query=exogenous_for_attn2,
            key=endogenous_for_attn2,
            value=endogenous_for_attn2
        )  # (batch, 1, 256)
        enhanced_exogenous = enhanced_exogenous.squeeze(1)  # (batch, 256)
        
        # ========== 融合策略 ==========
        # 拼接两个增强特征
        fused_features = torch.cat([enhanced_endogenous, enhanced_exogenous], dim=1)  # (batch, 512)
        
        # 投影到融合维度
        fused_features = self.fusion_proj(fused_features)  # (batch, 256)
        fused_features = self.activation(fused_features)
        
        # ========== 输出投影（多层残差降维） ==========
        # endogenous_hidden_dim维度的残差模块
        x = self.output_res1(fused_features)  # (batch, 256)
        
        # endogenous_hidden_dim → 64
        x = self.output_proj1(x)  # (batch, 64)
        x = self.activation(x)
        x = self.output_dropout1(x)
        
        # 64维度的残差模块
        x = self.output_res2(x)  # (batch, 64)
        
        # 64 → 32
        x = self.output_proj2(x)  # (batch, 32)
        x = self.activation(x)
        x = self.output_dropout2(x)
        
        # 32 → prediction_len（直接线性变换，无激活函数，无残差模块）
        x = self.output_proj3(x)  # (batch, prediction_len)
        
        # ========== Step 9: v0.45新增 - 反归一化（将输出还原到原始尺度）==========
        if self.use_norm and means is not None:
            # 使用输出特征索引对应的均值和标准差
            output_mean = means[:, :, self.output_feature_index]  # [B, 1]
            output_std = stdev[:, :, self.output_feature_index]   # [B, 1]
            
            # 反归一化: x_original = x_normalized * std + mean
            # 广播: [B, 1] → [B, prediction_len]
            x = x * output_std + output_mean
        
        return x
    
    def get_num_parameters(self) -> int:
        """
        返回模型的可训练参数数量
        
        此方法是为了兼容训练脚本
        
        Returns:
            可训练参数数量
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_info(self) -> Dict:
        """
        返回模型的详细信息（用于保存到配置文件）
        
        Returns:
            包含模型详细信息的字典
        """
        # 获取归一化特征索引
        norm_indices = self.norm_mask.nonzero(as_tuple=True)[0].cpu().tolist()
        
        return {
            'seq_len': self.seq_len,
            'n_features': self.n_features,
            'endogenous_features': self.endogenous_features,
            'exogenous_features': self.exogenous_features,
            'prediction_len': self.prediction_len,
            'endogenous_blocks': self.endogenous_blocks,
            'exogenous_blocks': self.exogenous_blocks,
            'endogenous_hidden_dim': self.endogenous_hidden_dim,
            'exogenous_hidden_dim': self.exogenous_hidden_dim,
            'dropout': self.dropout,
            'activation': self.activation_name,
            'norm_type': self.norm_type,
            'use_layernorm': self.use_layernorm,
            'use_layernorm_in_tsmixer': self.use_layernorm_in_tsmixer,
            'use_layernorm_in_attention': self.use_layernorm_in_attention,
            'use_layernorm_before_pooling': self.use_layernorm_before_pooling,
            'use_residual': self.use_residual,
            'num_parameters': self.get_num_parameters(),
            'missing_embedding_enabled': True,  # v0.43
            'missing_value_flag': self.missing_value_flag,  # v0.43
            # v0.45新增信息
            'use_norm': self.use_norm,
            'norm_feature_indices': norm_indices,
            'num_norm_features': len(norm_indices),
            'output_feature_index': self.output_feature_index,
            'architecture': {
                'shared_time_mixing': self.shared_time_mixing is not None,
                'endogenous_branch': {
                    'n_blocks': self.endogenous_blocks,
                    'hidden_dim': self.endogenous_hidden_dim
                },
                'exogenous_branch': {
                    'n_blocks': self.exogenous_blocks,
                    'hidden_dim': self.exogenous_hidden_dim
                },
                'fusion': {
                    'cross_attention_layers': 2,
                    'self_attention_layers': 1
                }
            }
        }
