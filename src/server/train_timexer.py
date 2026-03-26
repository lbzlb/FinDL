#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
主训练脚本（集成TensorBoard）- TimeXer版本（双分支架构）
版本: v0.45
日期: 20260207

整合所有模块，作为训练入口，支持TensorBoard可视化
使用TimeXer双分支模型进行时间序列预测（内生+宏观，交叉注意力融合）

v0.45新增：Instance Normalization + 反归一化（基于v0.43）
- 学习型Missing Embedding（继承自v0.43）
- Instance Normalization：模型内部对输入进行归一化
- 反归一化：输出前将预测值还原到原始尺度
- 损失计算在原始尺度上进行
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import yaml
import argparse
import sys
import importlib.util
from datetime import datetime
import pandas as pd
import json
import os
import re


# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


# ===== 默认训练数据路径（如需调整请修改此处） =====
DEFAULT_PREPROCESSED_DIR = project_root / "data" / "processed" / "/data/project_20251211/data/processed/preprocess_data_NaNto-1000_20260214115803"

# ===== 数据设备配置（如需调整请修改此处） =====
# 选项: 'cpu' 或 'cuda'
# - 'cpu': 数据存储在CPU内存中，训练时通过pin_memory传输到GPU（推荐，支持多进程加载）
# - 'cuda': 数据直接存储在GPU显存中，训练时无需传输（更快，但需要显存充足，且num_workers必须为0）
# 注意: 使用内存映射模式（mmap_mode=True）时，数据自动使用CPU，此配置被忽略
DATA_DEVICE = 'cpu'  # 修改此处: 'cpu' 或 'cuda'


# 动态导入模块
def _load_module(module_path: Path, module_name: str):
    """动态加载模块"""
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# 导入所需模块（使用v0.45_20260207版本的模型，v0.3_20260118版本的trainer，v0.3_20260119版本的数据(支持mmap)，v0.1版本的utils）
models_path = project_root / "src" / "models" / "v0.45_20260207"
data_path = project_root / "src" / "data" / "v0.3_20260119"
training_path = project_root / "src" / "training" / "v0.3_20260118"
utils_path = project_root / "src" / "utils" / "v0.1_20251212"

# 导入模型
timexer_module = _load_module(models_path / "timexer.py", "timexer")
TimeXer = timexer_module.TimeXer

# 导入数据集
stock_dataset_module = _load_module(data_path / "stock_dataset.py", "stock_dataset")
StockDataset = stock_dataset_module.StockDataset

# 导入预处理数据集
preprocessed_dataset_module = _load_module(data_path / "preprocessed_dataset.py", "preprocessed_dataset")
PreprocessedStockDataset = preprocessed_dataset_module.PreprocessedStockDataset

# 导入训练器
trainer_module = _load_module(training_path / "trainer.py", "trainer")
Trainer = trainer_module.Trainer
EarlyStopping = trainer_module.EarlyStopping

# 导入工具函数
feature_utils_module = _load_module(utils_path / "feature_utils.py", "feature_utils")
compute_feature_stats = feature_utils_module.compute_feature_stats
save_feature_stats = feature_utils_module.save_feature_stats
get_feature_columns = feature_utils_module.get_feature_columns

data_utils_module = _load_module(utils_path / "data_utils.py", "data_utils")
load_parquet_file = data_utils_module.load_parquet_file
get_data_by_rows = data_utils_module.get_data_by_rows

loss_utils_module = _load_module(utils_path / "loss_utils.py", "loss_utils")
MAPELoss = loss_utils_module.MAPELoss
SMAPELoss = loss_utils_module.SMAPELoss


def load_config(config_path: str) -> dict:
    """加载YAML配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def create_optimizer(model: nn.Module, optimizer_config: dict) -> torch.optim.Optimizer:
    """创建优化器"""
    opt_type = optimizer_config['type'].lower()
    lr = float(optimizer_config['lr'])
    weight_decay = float(optimizer_config.get('weight_decay', 0.0))
    betas = optimizer_config.get('betas', [0.9, 0.999])
    # 确保 betas 是浮点数列表
    if isinstance(betas, list):
        betas = tuple(float(b) for b in betas)
    
    if opt_type == 'adam':
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas)
    elif opt_type == 'adamw':
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas)
    elif opt_type == 'sgd':
        momentum = float(optimizer_config.get('momentum', 0.9))
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
    else:
        raise ValueError(f"Unknown optimizer type: {opt_type}")


def create_scheduler(optimizer: torch.optim.Optimizer, scheduler_config: dict, num_epochs: int):
    """创建学习率调度器"""
    sched_type = scheduler_config['type'].lower()
    
    if sched_type == 'none':
        return None
    elif sched_type == 'cosine':
        T_max = int(scheduler_config.get('T_max', num_epochs))
        eta_min = float(scheduler_config.get('eta_min', 0))
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
    elif sched_type == 'step':
        step_size = int(scheduler_config.get('step_size', 30))
        gamma = float(scheduler_config.get('gamma', 0.1))
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif sched_type == 'plateau':
        factor = float(scheduler_config.get('factor', 0.5))
        patience = int(scheduler_config.get('patience', 10))
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=factor, patience=patience)
    else:
        raise ValueError(f"Unknown scheduler type: {sched_type}")


def get_feature_split_indices(preset: str, n_features: int) -> tuple:
    """
    根据预设方案返回内生和外生索引
    
    Args:
        preset: 预设方案名称 ("default" 或 "subset10")
        n_features: 总特征数量
        
    Returns:
        (endogenous_indices, exogenous_indices): 内生和外生索引列表
        
    Raises:
        ValueError: 如果预设方案未知
    """
    if preset == "default":
        # 方案1: 内生[0-43]，外生[44-63]
        endogenous_indices = list(range(0, 44))
        exogenous_indices = list(range(44, 64))
    elif preset == "subset10":
        # 方案2: 内生[1-10]，外生[0,11-63]
        endogenous_indices = list(range(1, 11))  # [1,2,3,4,5,6,7,8,9,10]
        exogenous_indices = [0] + list(range(11, 64))  # [0,11,12,...,63]
    else:
        raise ValueError(f"Unknown feature_split_preset: {preset}. Available: 'default', 'subset10'")
    
    print(f"\n应用特征分离预设方案: '{preset}'")
    print(f"  内生索引: {endogenous_indices[:5]}...{endogenous_indices[-5:]} (共{len(endogenous_indices)}个)")
    print(f"  外生索引: {exogenous_indices[:5]}...{exogenous_indices[-5:]} (共{len(exogenous_indices)}个)")
    
    return endogenous_indices, exogenous_indices


def create_criterion(loss_config: dict) -> nn.Module:
    """创建损失函数"""
    loss_type = loss_config['type'].lower()
    reduction = loss_config.get('reduction', 'mean')
    
    if loss_type == 'mse':
        return nn.MSELoss(reduction=reduction)
    elif loss_type == 'mae':
        return nn.L1Loss(reduction=reduction)
    elif loss_type == 'huber':
        delta = float(loss_config.get('delta', 1.0))
        return nn.HuberLoss(reduction=reduction, delta=delta)
    elif loss_type == 'mape':
        epsilon = float(loss_config.get('epsilon', 1e-8))
        max_relative_error = loss_config.get('max_relative_error', None)
        if max_relative_error is not None:
            max_relative_error = float(max_relative_error)
        return MAPELoss(reduction=reduction, epsilon=epsilon, max_relative_error=max_relative_error)
    elif loss_type == 'smape':
        epsilon = float(loss_config.get('epsilon', 1e-8))
        max_relative_error = loss_config.get('max_relative_error', None)
        if max_relative_error is not None:
            max_relative_error = float(max_relative_error)
        return SMAPELoss(reduction=reduction, epsilon=epsilon, max_relative_error=max_relative_error)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}. Available: 'mse', 'mae', 'huber', 'mape', 'smape'")


def compute_train_stats(train_dataset: StockDataset, dataset_config: dict) -> dict:
    """计算训练集的特征统计量"""
    print("计算训练集特征统计量...")
    
    # 获取特征列
    exclude_columns = dataset_config['features']['exclude_columns']
    
    # 从训练集中采样一些数据来计算统计量
    # 为了避免内存问题，只使用部分数据
    sample_size = min(1000, len(train_dataset))
    indices = torch.randperm(len(train_dataset))[:sample_size].tolist()  # 转换为Python列表
    
    all_features = []
    for idx in indices:
        sample = train_dataset.index_df.iloc[idx]
        source_file = sample['source_file']
        
        # 读取数据文件
        df = load_parquet_file(source_file, use_cache=True)
        
        # 提取输入数据
        input_start = sample['input_row_start']
        input_end = sample['input_row_end']
        input_df = get_data_by_rows(df, input_start, input_end)
        
        # 获取特征列
        feature_columns = get_feature_columns(input_df, exclude_columns)
        features = input_df[feature_columns]
        all_features.append(features)
    
    # 合并所有特征
    all_features_df = pd.concat(all_features, ignore_index=True)
    
    # 计算统计量
    stats = compute_feature_stats(all_features_df, feature_columns, method="standard")
    
    print(f"特征统计量计算完成，共 {len(stats)} 个特征")
    return stats


def get_latest_code_modify_time(config_version: str) -> str:
    """
    获取训练代码及相关文件的最新修改时间
    
    Args:
        config_version: 配置版本号
        
    Returns:
        时间戳字符串，格式：YYYYMMDDHHMMSS
    """
    # 需要检查的文件列表
    files_to_check = []
    
    # 1. 训练脚本本身
    train_script = Path(__file__)
    files_to_check.append(train_script)
    
    # 2. 模型文件（使用v0.45_20260207版本）
    models_path = project_root / "src" / "models" / "v0.45_20260207"
    files_to_check.append(models_path / "timexer.py")
    files_to_check.append(models_path / "timexer_blocks.py")
    # v0.45: 不需要masked_layernorm.py
    
    # 3. 数据集文件（使用v0.3_20260119版本）
    data_path = project_root / "src" / "data" / "v0.3_20260119"
    files_to_check.append(data_path / "stock_dataset.py")
    files_to_check.append(data_path / "preprocessed_dataset.py")
    
    # 4. 训练器文件（使用v0.3_20260118版本）
    training_path = project_root / "src" / "training" / "v0.3_20260118"
    files_to_check.append(training_path / "trainer.py")
    
    # 5. 工具文件（使用v0.1版本）
    utils_path = project_root / "src" / "utils" / "v0.1_20251212"
    files_to_check.append(utils_path / "feature_utils.py")
    files_to_check.append(utils_path / "data_utils.py")
    files_to_check.append(utils_path / "loss_utils.py")
    
    # 6. 配置文件
    config_dir = project_root / "configs" / config_version
    config_files = [
        config_dir / "training_config.yaml"
    ]
    # 尝试添加其他配置文件（如果存在）
    for config_name in ["dataset_config.yaml", "timexer_config.yaml"]:
        config_file = config_dir / config_name
        if config_file.exists():
            config_files.append(config_file)
        else:
            # 如果当前版本不存在，尝试使用v0.41版本
            old_config_file = project_root / "configs" / "v0.41_20260116" / config_name
            if old_config_file.exists():
                config_files.append(old_config_file)
    
    files_to_check.extend(config_files)
    
    # 获取所有文件的最新修改时间
    latest_mtime = 0
    for file_path in files_to_check:
        if file_path.exists():
            mtime = os.path.getmtime(file_path)
            latest_mtime = max(latest_mtime, mtime)
    
    # 转换为时间戳字符串 YYYYMMDDHHMMSS
    dt = datetime.fromtimestamp(latest_mtime)
    timestamp = dt.strftime('%Y%m%d%H%M%S')
    
    return timestamp


def find_preprocessed_files(preprocessed_dir: Path) -> tuple[Path, Path]:
    """
    自动检测预处理目录中的训练集和验证集文件
    
    Args:
        preprocessed_dir: 预处理数据目录
        
    Returns:
        (train_pt_file, val_pt_file): 训练集和验证集文件路径
        
    Raises:
        FileNotFoundError: 如果找不到对应的文件
    """
    # 查找 train_*.pt 文件
    train_files = list(preprocessed_dir.glob("train_*.pt"))
    if not train_files:
        available_files = list(preprocessed_dir.glob("*.pt"))
        raise FileNotFoundError(
            f"在预处理目录中未找到训练集文件 (train_*.pt)\n"
            f"预处理目录: {preprocessed_dir}\n"
            f"目录中的 .pt 文件: {[f.name for f in available_files]}\n"
            f"请先运行预处理脚本生成训练集文件"
        )
    if len(train_files) > 1:
        print(f"警告: 找到多个训练集文件，使用最新的: {train_files[0].name}")
        train_files = sorted(train_files, key=lambda x: x.stat().st_mtime, reverse=True)
    
    # 查找 val_*.pt 文件
    val_files = list(preprocessed_dir.glob("val_*.pt"))
    if not val_files:
        available_files = list(preprocessed_dir.glob("*.pt"))
        raise FileNotFoundError(
            f"在预处理目录中未找到验证集文件 (val_*.pt)\n"
            f"预处理目录: {preprocessed_dir}\n"
            f"目录中的 .pt 文件: {[f.name for f in available_files]}\n"
            f"请先运行预处理脚本生成验证集文件"
        )
    if len(val_files) > 1:
        print(f"警告: 找到多个验证集文件，使用最新的: {val_files[0].name}")
        val_files = sorted(val_files, key=lambda x: x.stat().st_mtime, reverse=True)
    
    train_pt_file = train_files[0]
    val_pt_file = val_files[0]
    
    print(f"检测到训练集文件: {train_pt_file.name}")
    print(f"检测到验证集文件: {val_pt_file.name}")
    
    return train_pt_file, val_pt_file


def extract_version_number(config_version: str) -> str:
    """
    从config_version中提取版本号部分（去掉日期后缀）
    
    Args:
        config_version: 配置版本号，如 'v0.42_20260118'
        
    Returns:
        版本号，如 'v0.42'
    """
    # 使用split分割，取第一部分（下划线前的版本号）
    # 例如: 'v0.42_20260118' -> 'v0.42'
    #      'v0.42_test_20260118' -> 'v0.42_test'
    parts = config_version.split('_')
    
    # 如果包含数字日期后缀，去掉最后一个部分
    # 检查最后一部分是否是纯数字（日期）
    if len(parts) > 1 and parts[-1].isdigit():
        # 去掉日期部分，重新组合
        return '_'.join(parts[:-1])
    
    # 如果没有日期后缀，返回原值
    return config_version


def get_data_generation_time(preprocessed_dir: Path = None, dataset_config: dict = None) -> str:
    """
    获取数据生成时间戳及后缀（从时间戳开始往后的所有内容）
    
    Args:
        preprocessed_dir: 预处理数据目录（使用预处理数据时）
        dataset_config: 数据集配置（使用原始数据时）
        
    Returns:
        时间戳及后缀字符串，例如：20260109111857_500120 或 20251216172734
    """
    # 情况1: 使用预处理数据
    if preprocessed_dir is not None:
        # 从目录名提取时间戳及之后的所有内容
        # 例如：preprocess_data_v0.9_20260109111857_500120 -> 20260109111857_500120
        dir_name = preprocessed_dir.name
        # 匹配格式：preprocess_data_v0.9_后面所有内容
        match = re.search(r'preprocess_data_v\d+\.\d+_(.+)', dir_name)
        if match:
            return match.group(1)
        
        # 如果目录名不符合格式，尝试从 preprocess_summary.json 读取
        summary_files = list(preprocessed_dir.glob("preprocess_summary_*.json"))
        if summary_files:
            try:
                with open(summary_files[0], 'r', encoding='utf-8') as f:
                    summary = json.load(f)
                    created_at = summary.get('created_at', '')
                    if created_at:
                        # 解析格式：2025-12-16 13:20:23 -> 20251216132023
                        dt = datetime.strptime(created_at, '%Y-%m-%d %H:%M:%S')
                        return dt.strftime('%Y%m%d%H%M%S')
            except Exception:
                pass
    
    # 情况2: 使用原始数据，从 dataset_config.yaml 中的 index_dir 提取
    if dataset_config is not None:
        index_dir = dataset_config.get('index_dir', '')
        if index_dir:
            index_path = Path(index_dir)
            dir_name = index_path.name
            # 尝试匹配 YYYYMMDD_HHMMSS 格式，提取时间戳及之后的所有内容
            match = re.search(r'(\d{8})_(\d{6})(.*)', dir_name)
            if match:
                date_part = match.group(1)  # YYYYMMDD
                time_part = match.group(2)  # HHMMSS
                suffix = match.group(3)  # 之后的所有内容（可能为空）
                return f"{date_part}{time_part}{suffix}"
            
            # 尝试匹配 YYYYMMDDHHMMSS 格式（14位连续数字），提取时间戳及之后的所有内容
            match = re.search(r'(\d{14})(.*)', dir_name)
            if match:
                timestamp = match.group(1)
                suffix = match.group(2)  # 之后的所有内容（可能为空）
                return f"{timestamp}{suffix}"
    
    # 如果都无法提取，返回当前时间
    return datetime.now().strftime('%Y%m%d%H%M%S')


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='训练TimeXer模型（双分支架构，交叉注意力融合，集成TensorBoard，学习型Missing Embedding）')
    parser.add_argument('--config_version', type=str, default='v0.45_20260207',
                        help='配置版本号')
    parser.add_argument('--model_config', type=str, default=None,
                        help='模型配置文件路径（可选，默认使用config_version）')
    parser.add_argument('--training_config', type=str, default=None,
                        help='训练配置文件路径（可选，默认使用config_version）')
    parser.add_argument('--dataset_config', type=str, default=None,
                        help='数据集配置文件路径（可选，默认使用config_version）')
    parser.add_argument('--use_preprocessed', action='store_true', default=True,
                        help='使用预处理的数据（默认开启，速度更快）')
    parser.add_argument('--no_preprocessed', action='store_true',
                        help='不使用预处理数据，使用原始数据加载方式')
    parser.add_argument('--preprocessed_dir', type=str, default=None,
                        help='预处理数据子目录路径（如data/processed/preprocess_data_20251216121500），留空则自动查找最新的')
    
    args = parser.parse_args()
    
    # 如果指定了 --no_preprocessed，则关闭预处理模式
    if args.no_preprocessed:
        args.use_preprocessed = False
    
    # 确定配置文件路径
    config_dir = project_root / "configs" / args.config_version
    
    # 优先使用命令行指定的配置，否则使用config_version目录下的配置
    # 如果config_version目录下不存在，则回退到v0.1_20251212（包含完整的配置文件）
    fallback_config_dir = project_root / "configs" / "v0.1_20251212"
    
    # 模型配置
    if args.model_config:
        model_config_path = args.model_config
    elif (config_dir / "timexer_config.yaml").exists():
        model_config_path = str(config_dir / "timexer_config.yaml")
    else:
        model_config_path = str(fallback_config_dir / "timexer_config.yaml")
    
    # 训练配置
    if args.training_config:
        training_config_path = args.training_config
    elif (config_dir / "training_config.yaml").exists():
        training_config_path = str(config_dir / "training_config.yaml")
    else:
        training_config_path = str(fallback_config_dir / "training_config.yaml")
    
    # 数据集配置
    if args.dataset_config:
        dataset_config_path = args.dataset_config
    elif (config_dir / "dataset_config.yaml").exists():
        dataset_config_path = str(config_dir / "dataset_config.yaml")
    else:
        dataset_config_path = str(fallback_config_dir / "dataset_config.yaml")
    
    print("=" * 80)
    print("TimeXer 训练脚本 v0.45 (双分支 + 交叉注意力 + TensorBoard + 学习型Missing Embedding)")
    print("=" * 80)
    print(f"模型配置: {model_config_path}")
    print(f"训练配置: {training_config_path}")
    print(f"数据集配置: {dataset_config_path}")
    print("=" * 80)
    
    # 加载配置文件
    print("\n加载配置文件...")
    model_config = load_config(model_config_path)
    training_config = load_config(training_config_path)
    dataset_config = load_config(dataset_config_path)
    
    model_cfg = model_config['model']
    train_cfg = training_config['training']
    dataset_cfg = dataset_config['dataset']
    
    # 读取TensorBoard配置
    tb_config = training_config.get('tensorboard', {})
    tb_enabled = tb_config.get('enabled', True)
    
    # 获取代码运行时间（用于目录命名，确保每次运行都是独立的）
    config_version = args.config_version
    print("\n获取代码运行时间...")
    run_timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    print(f"代码运行时间: {run_timestamp}")
    
    # 同时获取代码最新修改时间（用于记录到元数据）
    code_modify_timestamp = get_latest_code_modify_time(config_version)
    print(f"代码最新修改时间: {code_modify_timestamp}")
    
    # 初始化数据集
    print("\n初始化数据集...")
    
    # 判断是否使用预处理数据
    preprocessed_dir = DEFAULT_PREPROCESSED_DIR
    if args.use_preprocessed:
        # ===== 使用预处理数据（快速模式，支持mask + 内存映射） =====
        print("使用预处理数据（快速模式，支持mask + 内存映射优化）")
        
        # 确定预处理数据目录
        if args.preprocessed_dir:
            # 用户通过命令行指定了目录
            preprocessed_dir = project_root / args.preprocessed_dir
            print(f"使用命令行指定的预处理目录: {preprocessed_dir}")
        else:
            # 使用代码中设置的默认路径
            print(f"使用代码中设置的默认预处理目录: {preprocessed_dir}")
        
        # 验证路径是否存在
        if not preprocessed_dir.exists():
            raise FileNotFoundError(
                f"预处理数据目录不存在: {preprocessed_dir}\n"
                f"请检查路径是否正确，或先运行预处理脚本生成数据"
            )
        
        # 获取数据生成时间
        print("\n获取数据生成时间...")
        data_timestamp = get_data_generation_time(preprocessed_dir=preprocessed_dir, dataset_config=None)
        print(f"数据生成时间: {data_timestamp}")
        
        # 自动检测预处理文件
        print("\n自动检测预处理文件...")
        train_pt_file, val_pt_file = find_preprocessed_files(preprocessed_dir)
        
        # 加载预处理数据集（v0.3支持mask + 内存映射）
        # 使用内存映射模式避免OOM，预计算mask充分利用CPU内存
        print("  使用内存映射模式加载（数据保留在磁盘，按需加载到内存）")
        print("  启用预计算mask（一次性生成所有mask，充分利用CPU内存）")
        train_dataset = PreprocessedStockDataset(
            pt_file_path=str(train_pt_file),
            load_to_memory=False,  # mmap模式下忽略
            device=None,  # 内存映射模式必须使用CPU
            blank_value=-1000.0,  # 空白数据标记值
            return_mask=True,  # 启用mask
            mmap_mode=True,  # 启用内存映射（关键！）
            precompute_mask=True  # 预计算mask，提升训练速度，充分利用CPU内存
        )
        
        val_dataset = PreprocessedStockDataset(
            pt_file_path=str(val_pt_file),
            load_to_memory=False,
            device=None,
            blank_value=-1000.0,
            return_mask=True,
            mmap_mode=True,  # 启用内存映射
            precompute_mask=True  # 预计算mask
        )
        
        # 从元数据中获取特征统计量
        feature_stats = train_dataset.get_feature_stats()
    
    else:
        # ===== 使用原始数据（常规模式，支持mask） =====
        print("使用原始数据（常规模式，支持mask）")
        
        # 获取数据生成时间（从索引文件目录）
        print("\n获取数据生成时间...")
        data_timestamp = get_data_generation_time(preprocessed_dir=None, dataset_config=dataset_cfg)
        print(f"数据生成时间: {data_timestamp}")
        
        index_dir = Path(dataset_cfg['index_dir'])
        train_index_file = index_dir / dataset_cfg['train_index_file']
        val_index_file = index_dir / dataset_cfg['val_index_file']
        
        exclude_columns = dataset_cfg['features']['exclude_columns']
        target_column = dataset_cfg['features']['target_column']
        normalize = dataset_cfg['features']['normalize']
        normalize_method = dataset_cfg['features']['normalize_method']
        cache_enabled = dataset_cfg['cache_enabled']
        cache_size = dataset_cfg['cache_size']
        
        # 先创建训练集（用于计算统计量）
        train_dataset = StockDataset(
            index_file=str(train_index_file),
            exclude_columns=exclude_columns,
            target_column=target_column,
            normalize=False,  # 先不标准化，等计算完统计量后再标准化
            normalize_method=normalize_method,
            feature_stats=None,
            stats_file=None,
            cache_enabled=cache_enabled,
            cache_size=cache_size,
            blank_value=-1000.0,  # v0.42新增
            return_mask=True  # v0.42新增
        )
        
        # 计算特征统计量（如果需要标准化）
        feature_stats = None
        stats_file = None
        if normalize:
            feature_stats = compute_train_stats(train_dataset, dataset_cfg)
            
            # 重新创建训练集（使用统计量）
            train_dataset = StockDataset(
                index_file=str(train_index_file),
                exclude_columns=exclude_columns,
                target_column=target_column,
                normalize=normalize,
                normalize_method=normalize_method,
                feature_stats=feature_stats,
                stats_file=None,
                cache_enabled=cache_enabled,
                cache_size=cache_size,
                blank_value=-1000.0,
                return_mask=True
            )
        
        # 创建验证集
        val_dataset = StockDataset(
            index_file=str(val_index_file),
            exclude_columns=exclude_columns,
            target_column=target_column,
            normalize=normalize,
            normalize_method=normalize_method,
            feature_stats=feature_stats,
            stats_file=stats_file,
            cache_enabled=cache_enabled,
            cache_size=cache_size,
            blank_value=-1000.0,
            return_mask=True
        )
    
    # 创建输出目录（使用代码运行时间和数据生成时间）
    # 从config_version中提取版本号部分（去掉日期后缀）
    version_number = extract_version_number(config_version)
    output_dir = project_root / "experiments" / f"timexer_{version_number}_{run_timestamp}_{data_timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n输出目录: {output_dir}")
    print(f"  运行时间戳: {run_timestamp}")
    print(f"  数据时间戳: {data_timestamp}")
    print(f"  代码修改时间戳: {code_modify_timestamp}")
    
    # 保存特征统计量（如果需要）
    if feature_stats is not None:
        stats_file = output_dir / "feature_stats.json"
        save_feature_stats(feature_stats, stats_file)
        print(f"特征统计量已保存到: {stats_file}")
    
    # 获取实际的特征数量和序列长度
    num_features = train_dataset.get_num_features()
    seq_len = train_dataset.get_seq_len()
    
    print(f"训练集样本数: {len(train_dataset)}")
    print(f"验证集样本数: {len(val_dataset)}")
    print(f"特征数量: {num_features}")
    print(f"序列长度: {seq_len}")
    
    # 更新模型配置中的特征数量和序列长度（如果不同）
    if model_cfg.get('n_features', 0) != num_features:
        print(f"\n警告: 配置中的特征数量 ({model_cfg.get('n_features', 0)}) 与实际特征数量 ({num_features}) 不同")
        print(f"更新模型配置中的特征数量为: {num_features}")
        model_cfg['n_features'] = num_features
    
    if model_cfg.get('seq_len', 0) != seq_len:
        print(f"\n警告: 配置中的序列长度 ({model_cfg.get('seq_len', 0)}) 与实际序列长度 ({seq_len}) 不同")
        print(f"更新模型配置中的序列长度为: {seq_len}")
        model_cfg['seq_len'] = seq_len
    
    # 创建数据加载器
    print("\n创建数据加载器...")
    # 如果数据在GPU上，必须设置num_workers=0（多进程无法访问GPU数据）
    num_workers = 0 if DATA_DEVICE == 'cuda' else train_cfg['num_workers']
    # 如果数据在GPU上，pin_memory无效（数据已在GPU）
    pin_memory = False if DATA_DEVICE == 'cuda' else train_cfg['pin_memory']
    
    if DATA_DEVICE == 'cuda':
        print(f"  数据设备: GPU (num_workers=0, pin_memory=False)")
    else:
        print(f"  数据设备: CPU (num_workers={num_workers}, pin_memory={pin_memory})")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg['batch_size'],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=train_cfg.get('prefetch_factor', 2) if num_workers > 0 else None,  # 多进程时才需要prefetch
        persistent_workers=train_cfg.get('persistent_workers', True) if num_workers > 0 else False  # 多进程时才需要persistent
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg['batch_size'],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=train_cfg.get('prefetch_factor', 2) if num_workers > 0 else None,
        persistent_workers=train_cfg.get('persistent_workers', True) if num_workers > 0 else False
    )
    
    # 初始化模型（使用更新后的配置）
    print("\n初始化TimeXer模型（v0.45，学习型Missing Embedding）...")
    
    # 应用特征分离预设方案（如果配置中指定了）
    preset = model_cfg.get('feature_split_preset', None)
    if preset is not None:
        # 根据预设方案生成索引列表，覆盖配置中的索引
        endogenous_indices, exogenous_indices = get_feature_split_indices(preset, num_features)
        # 更新配置
        model_cfg['endogenous_indices'] = endogenous_indices
        model_cfg['exogenous_indices'] = exogenous_indices
    else:
        # 读取配置中的索引列表
        endogenous_indices = model_cfg.get('endogenous_indices', None)
        exogenous_indices = model_cfg.get('exogenous_indices', None)
        
        # 如果提供了索引列表，打印信息
        if endogenous_indices is not None and exogenous_indices is not None:
            print(f"\n使用索引列表方式分离特征:")
            print(f"  内生数据位置索引: {endogenous_indices[:5]}...{endogenous_indices[-5:]} (共{len(endogenous_indices)}个)")
            print(f"  宏观数据位置索引: {exogenous_indices[:5]}...{exogenous_indices[-5:]} (共{len(exogenous_indices)}个)")
        else:
            print(f"\n使用特征数量方式分离特征:")
            print(f"  内生特征数量: {model_cfg.get('endogenous_features', 44)}")
            print(f"  宏观特征数量: {model_cfg.get('exogenous_features', 20)}")
    
    model = TimeXer(
        seq_len=model_cfg['seq_len'],  # 使用自动检测的序列长度
        n_features=model_cfg['n_features'],  # 使用自动检测的特征数量
        endogenous_features=model_cfg.get('endogenous_features', 44),
        exogenous_features=model_cfg.get('exogenous_features', 20),
        prediction_len=model_cfg.get('prediction_len', 1),
        endogenous_indices=endogenous_indices,  # 索引列表（如果提供）
        exogenous_indices=exogenous_indices,  # 索引列表（如果提供）
        endogenous_blocks=model_cfg.get('endogenous_blocks', 3),
        endogenous_hidden_dim=model_cfg.get('endogenous_hidden_dim', 256),
        exogenous_blocks=model_cfg.get('exogenous_blocks', 2),
        exogenous_hidden_dim=model_cfg.get('exogenous_hidden_dim', 256),
        shared_time_mixing=model_cfg.get('shared_time_mixing', True),
        cross_attn_n_heads=model_cfg.get('cross_attn_n_heads', 8),
        cross_attn_ff_dim=model_cfg.get('cross_attn_ff_dim', 1024),
        dropout=model_cfg.get('dropout', 0.1),
        activation=model_cfg.get('activation', 'gelu'),
        use_layernorm=model_cfg.get('use_layernorm', True),
        use_residual=model_cfg.get('use_residual', True),
        norm_type=model_cfg.get('norm_type', 'layer'),
        n_blocks=model_cfg.get('n_blocks', None),  # 保留以兼容，不使用
        ff_dim=model_cfg.get('ff_dim', None),  # 保留以兼容，不使用
        temporal_aggregation_config=model_cfg.get('temporal_aggregation', {}),
        output_projection_config=model_cfg.get('output_projection', {}),
        # v0.45新增：Instance Normalization参数
        use_norm=model_cfg.get('use_norm', True),
        norm_feature_indices=model_cfg.get('norm_feature_indices', None),
        output_feature_index=model_cfg.get('output_feature_index', 2)
    )
    print(f"模型参数数量: {model.get_num_parameters():,}")
    print(f"LayerNorm: {'启用' if model.use_layernorm else '禁用'}")
    print(f"残差连接: {'启用' if model.use_residual else '禁用'}")
    print(f"学习型Missing Embedding: 启用 (自动处理-1000缺失值)")
    
    # v0.45新增：显示归一化配置
    print(f"Instance Normalization: {'启用' if model_cfg.get('use_norm', True) else '禁用'}")
    if model_cfg.get('use_norm', True):
        norm_indices = model_cfg.get('norm_feature_indices', None)
        if norm_indices is None:
            print(f"  归一化特征: 全部 (0-{num_features-1})")
        else:
            print(f"  归一化特征: {len(norm_indices)}个特征")
            print(f"  归一化索引: {norm_indices[:10]}{'...' if len(norm_indices) > 10 else ''}")
        print(f"  输出特征索引: {model_cfg.get('output_feature_index', 2)} (用于反归一化)")
        print(f"  反归一化: 启用 (输出还原到原始尺度)")
        print(f"  损失计算: 原始尺度 (更符合业务含义)")
    
    # 创建优化器
    print("\n创建优化器...")
    optimizer = create_optimizer(model, train_cfg['optimizer'])
    
    # 创建学习率调度器
    print("创建学习率调度器...")
    scheduler = create_scheduler(optimizer, train_cfg['scheduler'], train_cfg['num_epochs'])
    
    # 创建损失函数
    print("创建损失函数...")
    criterion = create_criterion(train_cfg['loss'])
    
    # 创建早停机制
    early_stopping = None
    if train_cfg['early_stopping']['enabled']:
        # 早停使用的指标和模式（如果没配置，使用best_metric的配置）
        early_stop_metric = train_cfg['early_stopping'].get('metric', train_cfg.get('best_metric', 'loss'))
        early_stop_mode = train_cfg['early_stopping'].get('mode', train_cfg.get('best_metric_mode', 'min'))
        early_stopping = EarlyStopping(
            patience=train_cfg['early_stopping']['patience'],
            min_delta=train_cfg['early_stopping']['min_delta'],
            mode=early_stop_mode
        )
    
    # 创建子目录
    configs_dir = output_dir / "configs"
    logs_dir = output_dir / "logs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # TensorBoard目录
    tensorboard_dir = None
    if tb_enabled:
        tb_log_dir = tb_config.get('log_dir', 'tensorboard')
        # 使用完整的模型文件夹名称作为运行名称
        run_name = output_dir.name
        tensorboard_dir = output_dir / tb_log_dir / run_name
        print(f"\nTensorBoard已启用")
        print(f"  日志目录: {tensorboard_dir}")
        print(f"  运行名称: {run_name}")
        print(f"  记录间隔: 每 {tb_config.get('log_interval', 50)} batch")
        print(f"  直方图: {'启用' if tb_config.get('log_histograms', True) else '禁用'}")
    
    # 创建训练器（带TensorBoard支持，v0.3支持mask + 内存映射数据）
    print("\n创建训练器（v0.3，支持mask + 内存映射数据）...")
    
    # 获取最佳模型选择配置
    best_metric = train_cfg.get('best_metric', 'loss')
    best_metric_mode = train_cfg.get('best_metric_mode', 'min')
    print(f"最佳模型选择指标: {best_metric} ({best_metric_mode})")
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        device=train_cfg['device'],
        save_dir=str(output_dir / "checkpoints"),
        save_best=train_cfg['save_best'],
        best_metric=best_metric,
        best_metric_mode=best_metric_mode,
        early_stopping=early_stopping,
        mixed_precision=train_cfg['mixed_precision'],
        val_interval=train_cfg['val_interval'],
        save_interval=train_cfg['save_interval'],
        log_file=str(logs_dir / "training_log.jsonl"),
        history_file=str(logs_dir / "training_history.json"),
        # TensorBoard参数
        tensorboard_enabled=tb_enabled,
        tensorboard_dir=str(tensorboard_dir) if tensorboard_dir else None,
        tb_log_interval=tb_config.get('log_interval', 50),
        tb_histogram_interval=tb_config.get('histogram_interval', 500),
        tb_log_histograms=tb_config.get('log_histograms', True)
    )
    
    # 保存配置到configs子目录
    print("\n保存配置...")
    
    # 获取模型的详细信息
    model_info = model.get_model_info()
    
    # 合并模型配置和详细信息，并添加运行时元数据
    detailed_model_config = {
        'metadata': {
            **model_config.get('metadata', {}),
            'run_timestamp': run_timestamp,  # 代码运行时间
            'code_modify_timestamp': code_modify_timestamp,  # 代码最后修改时间
            'data_timestamp': data_timestamp,  # 数据生成时间
            'output_directory': str(output_dir)  # 输出目录
        },
        'model': {
            **model_config.get('model', {}),
            **model_info  # 添加详细的模型信息
        }
    }
    
    # 保存详细的模型配置
    with open(configs_dir / "model_config.yaml", 'w', encoding='utf-8') as f:
        yaml.dump(detailed_model_config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
    
    with open(configs_dir / "training_config.yaml", 'w', encoding='utf-8') as f:
        yaml.dump(training_config, f, allow_unicode=True)
    with open(configs_dir / "dataset_config.yaml", 'w', encoding='utf-8') as f:
        yaml.dump(dataset_config, f, allow_unicode=True)
    
    # 开始训练
    print("\n" + "=" * 80)
    print("开始训练")
    print("=" * 80)
    if tb_enabled:
        # 提示使用父目录启动，这样可以对比多个实验
        tb_parent_dir = output_dir / tb_config.get('log_dir', 'tensorboard')
        print(f"\n启动TensorBoard查看训练:")
        print(f"  单次运行: tensorboard --logdir={tensorboard_dir}")
        print(f"  对比多次: tensorboard --logdir={tb_parent_dir.parent.parent / 'tensorboard'}")
        print("=" * 80)
    
    history = trainer.train(train_cfg['num_epochs'])
    
    # 最终保存训练历史（作为备份，实际上每个epoch后已经自动保存）
    with open(logs_dir / "training_history.json", 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    
    print(f"\n训练完成！所有结果已保存到: {output_dir}")
    if tb_enabled:
        tb_parent_dir = output_dir / tb_config.get('log_dir', 'tensorboard')
        print(f"\n查看TensorBoard:")
        print(f"  单次运行: tensorboard --logdir={tensorboard_dir}")
        print(f"  对比多次: tensorboard --logdir={tb_parent_dir.parent.parent / 'tensorboard'}")


if __name__ == '__main__':
    main()
