#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
基于预测数据的模型推理脚本
版本: v0.1
日期: 20260120133643

功能：
1. 加载训练好的模型
2. 加载预测数据（所有公司最近500天的数据）
3. 对每个公司进行预测
4. 计算预测收益率
5. 整合历史误差指标
6. 生成Excel和Parquet输出文件

使用方法：
    修改代码配置区域，设置模型目录、预测数据文件和误差排名文件路径
    然后运行: python predict_with_predata_v0.1_20260120133643.py
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import importlib.util
from datetime import datetime
from tqdm import tqdm
import yaml
import json
import re

# 添加项目路径（自动向上查找包含src目录的项目根目录）
current_path = Path(__file__).resolve()
project_root = current_path
while project_root.name and project_root != project_root.parent:
    if (project_root / 'src').exists():
        break
    project_root = project_root.parent

if not (project_root / 'src').exists():
    raise RuntimeError(f"无法找到项目根目录（应包含src文件夹），当前脚本位置: {Path(__file__)}")

sys.path.insert(0, str(project_root))


# ===== 配置区域（修改此处） =====
# 模型目录
MODEL_DIR = "/data/project_20251211/experiments/timexer_v0.43_20260127120833_20260119170929_500120"

# 预测数据文件（.pt文件路径）
PREDATA_FILE = "/data/project_20251211/data/pre_data/processed_data_20260118/data.pt"

# 误差排名文件路径
ERROR_RANKING_FILE = "/data/project_20251211/tests/inference_results/timexer_v0.43_20260127120833_20260119170929_500120_20260121134938/不同误差下的公司排名.parquet"

# 输出目录
OUTPUT_DIR = "/data/project_20251211/tests/pre_data"

# 计算设备
DEVICE = 'cuda'  # 或 'cpu'

# 推理批次大小
BATCH_SIZE = 256


# ============================================================================
# 工具函数
# ============================================================================

def _load_module(module_path: Path, module_name: str):
    """动态加载模块"""
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def detect_model_type(model_dir: Path) -> tuple:
    """
    检测模型类型和版本
    
    Returns:
        (model_type, model_version): 模型类型和版本
    """
    config_path = model_dir / "configs" / "model_config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"模型配置文件不存在: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    model_name = config.get('model', {}).get('name', '').lower()
    dir_name = model_dir.name.lower()
    
    # 优先从目录名精确识别TimeXer版本
    if 'timexer' in model_name or 'timexer' in dir_name:
        version_pattern = r'v0\.\d+_\d{8}'
        match = re.search(version_pattern, dir_name)
        if match:
            version_str = match.group(0)
            if version_str.startswith('v0.43_'):
                return 'timexer', 'v0.43_20260119'
            elif version_str.startswith('v0.42_'):
                return 'timexer', 'v0.42_20260118'
            elif version_str.startswith('v0.41_'):
                return 'timexer', 'v0.41_20260116'
            elif version_str.startswith('v0.5_'):
                return 'timexer_mlp', 'v0.5_20260107'
            elif version_str.startswith('v0.4_'):
                return 'timexer', 'v0.4_20260106'
        
        if 'timexer_mlp' in model_name:
            return 'timexer_mlp', 'v0.5_20260107'
        else:
            return 'timexer', 'v0.4_20260106'
    
    # 从模型名称识别其他类型
    if 'itransformer' in model_name or 'itransformer_decoder' in model_name:
        return 'itransformer', 'v0.1_20251212'
    elif 'crossformer' in model_name:
        return 'crossformer', 'v0.3_20251230'
    elif 'tsmixer' in model_name:
        return 'tsmixer', 'v0.2_20251226'
    else:
        if 'itransformer' in dir_name:
            return 'itransformer', 'v0.1_20251212'
        elif 'crossformer' in dir_name:
            return 'crossformer', 'v0.3_20251230'
        elif 'tsmixer' in dir_name:
            return 'tsmixer', 'v0.2_20251226'
        else:
            raise ValueError(f"无法识别模型类型: {model_name} (目录: {dir_name})")


def load_model_dynamically(model_dir: Path, device: str = 'cuda'):
    """
    动态加载模型
    
    Args:
        model_dir: 模型目录路径
        device: 计算设备
        
    Returns:
        加载好的模型
    """
    print(f"\n加载模型: {model_dir}")
    
    # 检测模型类型
    model_type, model_version = detect_model_type(model_dir)
    print(f"检测到模型类型: {model_type}, 版本: {model_version}")
    
    # 加载模型配置
    config_path = model_dir / "configs" / "model_config.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    model_config = config['model']
    
    # 加载checkpoint
    checkpoint_path = model_dir / "checkpoints" / "best_model.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"模型检查点不存在: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # 根据模型类型加载对应的模型类
    models_path = project_root / "src" / "models" / model_version
    
    if model_type == 'itransformer':
        itransformer_module = _load_module(models_path / "itransformer_decoder.py", "itransformer_decoder")
        ModelClass = itransformer_module.iTransformerDecoder
        
        model = ModelClass(
            input_features=model_config['input_features'],
            seq_len=model_config['seq_len'],
            d_model=model_config['d_model'],
            n_layers=model_config['n_layers'],
            n_heads=model_config['n_heads'],
            d_ff=model_config['d_ff'],
            dropout=model_config['dropout'],
            activation=model_config['activation'],
            decoder_config=model_config.get('decoder', {}),
            input_resnet_config=model_config.get('input_resnet', {}),
            output_resnet_config=model_config.get('output_resnet', {}),
            final_output_config=model_config.get('final_output', {})
        )
    
    elif model_type == 'crossformer':
        crossformer_module = _load_module(models_path / "crossformer.py", "crossformer")
        ModelClass = crossformer_module.Crossformer
        
        model = ModelClass(
            seq_len=model_config['seq_len'],
            n_features=model_config['n_features'],
            d_model=model_config['d_model'],
            n_blocks=model_config['n_blocks'],
            n_heads=model_config['n_heads'],
            n_segments=model_config['n_segments'],
            n_feature_groups=model_config['n_feature_groups'],
            dropout=model_config['dropout'],
            activation=model_config['activation'],
            prediction_len=model_config.get('prediction_len', 1),
            router_topk_ratio=model_config.get('router_topk_ratio', 0.5),
            positional_encoding=model_config.get('positional_encoding', {}),
            temporal_aggregation=model_config.get('temporal_aggregation', {}),
            output_projection=model_config.get('output_projection', {})
        )
    
    elif model_type == 'timexer_mlp':
        timexer_mlp_module = _load_module(models_path / "timexer_mlp.py", "timexer_mlp")
        ModelClass = timexer_mlp_module.TimeXerMLP
        
        model = ModelClass(
            seq_len=model_config['seq_len'],
            n_features=model_config['n_features'],
            endogenous_features=model_config.get('endogenous_features', 44),
            exogenous_features=model_config.get('exogenous_features', 20),
            prediction_len=model_config.get('prediction_len', 1),
            endogenous_indices=model_config.get('endogenous_indices'),
            exogenous_indices=model_config.get('exogenous_indices'),
            endogenous_blocks=model_config.get('endogenous_blocks', 3),
            endogenous_hidden_dim=model_config.get('endogenous_hidden_dim', 256),
            exogenous_blocks=model_config.get('exogenous_blocks', 2),
            exogenous_hidden_dim=model_config.get('exogenous_hidden_dim', 256),
            shared_time_mixing=model_config.get('shared_time_mixing', True),
            mlp_fusion_ff_dim=model_config.get('mlp_fusion_ff_dim', 512),
            dropout=model_config.get('dropout', 0.1),
            activation=model_config.get('activation', 'gelu'),
            use_layernorm=model_config.get('use_layernorm', True),
            use_residual=model_config.get('use_residual', True),
            norm_type=model_config.get('norm_type', 'layer'),
            n_blocks=model_config.get('n_blocks', None),
            ff_dim=model_config.get('ff_dim', None),
            temporal_aggregation_config=model_config.get('temporal_aggregation', {}),
            output_projection_config=model_config.get('output_projection', {})
        )
    
    elif model_type == 'timexer':
        timexer_module = _load_module(models_path / "timexer.py", "timexer")
        ModelClass = timexer_module.TimeXer
        
        # 基础参数
        model_params = {
            'seq_len': model_config['seq_len'],
            'n_features': model_config['n_features'],
            'prediction_len': model_config.get('prediction_len', 1),
            'endogenous_features': model_config.get('endogenous_features', 44),
            'exogenous_features': model_config.get('exogenous_features', 20),
            'endogenous_indices': model_config.get('endogenous_indices'),
            'exogenous_indices': model_config.get('exogenous_indices'),
            'endogenous_blocks': model_config.get('endogenous_blocks', 3),
            'endogenous_hidden_dim': model_config.get('endogenous_hidden_dim', 256),
            'exogenous_blocks': model_config.get('exogenous_blocks', 2),
            'exogenous_hidden_dim': model_config.get('exogenous_hidden_dim', 256),
            'shared_time_mixing': model_config.get('shared_time_mixing', True),
            'time_mixing_type': model_config.get('time_mixing_type', 'attention'),
            'time_attn_n_heads': model_config.get('time_attn_n_heads', 8),
            'use_rope': model_config.get('use_rope', True),
            'cross_attn_n_heads': model_config.get('cross_attn_n_heads', 8),
            'cross_attn_ff_dim': model_config.get('cross_attn_ff_dim', 1024),
            'dropout': model_config.get('dropout', 0.1),
            'activation': model_config.get('activation', 'gelu'),
            'use_layernorm': model_config.get('use_layernorm', True),
            'use_residual': model_config.get('use_residual', True),
            'temporal_aggregation_config': model_config.get('temporal_aggregation', {}),
            'output_projection_config': model_config.get('output_projection', {})
        }
        
        # v0.41及以上版本特有参数
        if model_version in ['v0.41_20260116', 'v0.42_20260118', 'v0.43_20260119']:
            model_params['use_layernorm_in_tsmixer'] = model_config.get('use_layernorm_in_tsmixer')
            model_params['use_layernorm_in_attention'] = model_config.get('use_layernorm_in_attention')
            model_params['use_layernorm_before_pooling'] = model_config.get('use_layernorm_before_pooling')
        
        model = ModelClass(**model_params)
    
    elif model_type == 'tsmixer':
        tsmixer_module = _load_module(models_path / "tsmixer.py", "tsmixer")
        ModelClass = tsmixer_module.TSMixer
        
        model = ModelClass(
            seq_len=model_config['seq_len'],
            n_features=model_config['n_features'],
            prediction_len=model_config.get('prediction_len', 1),
            n_blocks=model_config.get('n_blocks', 4),
            ff_dim=model_config.get('ff_dim', 2048),
            dropout=model_config.get('dropout', 0.1),
            activation=model_config.get('activation', 'gelu'),
            norm_type=model_config.get('norm_type', 'layer'),
            use_residual=model_config.get('use_residual', True),
            temporal_aggregation_config=model_config.get('temporal_aggregation', {}),
            output_projection_config=model_config.get('output_projection', {})
        )
    
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"模型参数数量: {model.get_num_parameters():,}")
    print(f"模型加载完成，使用设备: {device}")
    
    return model


def load_predata(predata_file: Path):
    """
    加载预测数据文件
    
    Args:
        predata_file: 预测数据文件路径
    
    Returns:
        data_dict: 包含X tensor和metadata的字典
    """
    print(f"\n加载预测数据: {predata_file}")
    
    if not predata_file.exists():
        raise FileNotFoundError(f"预测数据文件不存在: {predata_file}")
    
    data = torch.load(predata_file, map_location='cpu', weights_only=False)
    
    print(f"数据形状: {data['X'].shape}")
    print(f"  - 样本数（公司数）: {data['X'].shape[0]}")
    print(f"  - 序列长度: {data['X'].shape[1]}")
    print(f"  - 特征数: {data['X'].shape[2]}")
    print(f"公司数: {data['metadata']['num_companies']}")
    
    return data


def load_error_ranking(error_ranking_file: Path):
    """
    加载误差排名文件
    
    Args:
        error_ranking_file: 误差排名文件路径
    
    Returns:
        error_df: 误差数据DataFrame（透视后）
    """
    print(f"\n加载误差排名文件: {error_ranking_file}")
    
    if not error_ranking_file.exists():
        print(f"警告: 误差排名文件不存在: {error_ranking_file}")
        return None
    
    df = pd.read_parquet(error_ranking_file)
    print(f"误差排名记录数: {len(df)}")
    
    # 数据透视：将排名方式转换为列
    pivot_df = df.pivot_table(
        index=['company_id', 'company_name', 'stock_code'],
        columns='排名方式',
        values='指标值',
        aggfunc='first'
    ).reset_index()
    
    # 统一数据类型：将company_id转换为int（与results_df保持一致）
    try:
        pivot_df['company_id'] = pivot_df['company_id'].astype(int)
        print(f"已将company_id转换为int类型")
    except (ValueError, TypeError) as e:
        print(f"警告: 无法将company_id转换为整数: {e}，保持原类型")
    
    print(f"透视后的公司数: {len(pivot_df)}")
    print(f"误差指标列: {[col for col in pivot_df.columns if col not in ['company_id', 'company_name', 'stock_code']]}")
    
    return pivot_df


def predict_and_merge(
    model: nn.Module,
    predata: dict,
    error_ranking_df: pd.DataFrame = None,
    device: str = 'cuda',
    batch_size: int = 256
):
    """
    对预测数据进行推理并合并误差信息
    
    Args:
        model: 模型
        predata: 预测数据字典
        error_ranking_df: 误差排名DataFrame
        device: 计算设备
        batch_size: 批次大小
    
    Returns:
        results_df: 结果DataFrame
    """
    print("\n" + "=" * 80)
    print("开始预测")
    print("=" * 80)
    
    # 提取数据
    X = predata['X']
    metadata = predata['metadata']
    company_info_list = metadata['company_info']
    feature_columns = metadata['feature_columns_example']
    
    # 查找"收盘"列的索引
    close_idx = None
    for idx, col in enumerate(feature_columns):
        if '收盘' in col:
            close_idx = idx
            break
    
    if close_idx is None:
        raise ValueError("在特征列中未找到'收盘'列")
    
    print(f"'收盘'列索引: {close_idx}")
    
    # 批量推理
    model.eval()
    predictions = []
    
    print("\n执行批量预测...")
    with torch.no_grad():
        for i in tqdm(range(0, len(X), batch_size), desc="预测进度"):
            batch_X = X[i:i+batch_size].to(device)
            batch_pred = model(batch_X)
            predictions.extend(batch_pred.cpu().numpy().flatten().tolist())
    
    print(f"预测完成，共 {len(predictions)} 个样本")
    
    # 构建结果DataFrame
    results = []
    
    for idx, company_info in enumerate(company_info_list):
        if idx >= len(predictions):
            break
        
        # 获取基本信息
        company_id = company_info.get('sequence_id', idx)
        company_name = company_info.get('company_name', 'Unknown')
        stock_code = company_info.get('stock_code', 'Unknown')
        
        # 获取日期列表
        dates = company_info.get('dates', [])
        last_trade_date = dates[-1] if dates else None
        
        # 获取最后一天的收盘价（从原始数据）
        last_close_price = X[idx, -1, close_idx].item()
        
        # 预测收盘价
        predicted_close_price = predictions[idx]
        
        # 计算收益率
        if last_close_price != 0 and not np.isnan(last_close_price) and last_close_price != -1000:
            return_rate = (predicted_close_price - last_close_price) / last_close_price * 100
        else:
            return_rate = np.nan
        
        result_row = {
            'company_id': company_id,
            'company_name': company_name,
            'stock_code': stock_code,
            'last_trade_date': last_trade_date,
            'last_close_price': last_close_price,
            'predicted_close_price': predicted_close_price,
            'return_rate': return_rate
        }
        
        results.append(result_row)
    
    results_df = pd.DataFrame(results)
    
    # 合并误差信息
    if error_ranking_df is not None:
        print("\n合并误差指标...")
        print(f"results_df中的公司数: {len(results_df)}")
        print(f"error_ranking_df中的公司数: {len(error_ranking_df)}")
        
        # 调试信息：检查前5个公司的键值
        print("\nresults_df前5个公司的键值:")
        for i in range(min(5, len(results_df))):
            row = results_df.iloc[i]
            print(f"  {i}: company_id={row['company_id']} ({type(row['company_id']).__name__}), "
                  f"company_name='{row['company_name']}', stock_code='{row['stock_code']}'")
        
        print("\nerror_ranking_df前5个公司的键值:")
        for i in range(min(5, len(error_ranking_df))):
            row = error_ranking_df.iloc[i]
            print(f"  {i}: company_id={row['company_id']} ({type(row['company_id']).__name__}), "
                  f"company_name='{row['company_name']}', stock_code='{row['stock_code']}'")
        
        # 检查有多少company_id能匹配上
        common_ids = set(results_df['company_id']) & set(error_ranking_df['company_id'])
        print(f"\n可以通过company_id匹配的公司数: {len(common_ids)}")
        
        # 尝试只通过company_id进行匹配
        results_df = results_df.merge(
            error_ranking_df,
            on='company_id',
            how='left',
            suffixes=('', '_error')
        )
        
        # 如果有重复的列名（company_name和stock_code），需要处理
        if 'company_name_error' in results_df.columns:
            # 用error_ranking_df中的值更新（如果results_df中的值不存在或为Unknown）
            mask = (results_df['company_name'] == 'Unknown') | results_df['company_name'].isna()
            results_df.loc[mask, 'company_name'] = results_df.loc[mask, 'company_name_error']
            results_df = results_df.drop('company_name_error', axis=1)
        
        if 'stock_code_error' in results_df.columns:
            mask = (results_df['stock_code'] == 'Unknown') | results_df['stock_code'].isna()
            results_df.loc[mask, 'stock_code'] = results_df.loc[mask, 'stock_code_error']
            results_df = results_df.drop('stock_code_error', axis=1)
        
        print(f"合并后的记录数: {len(results_df)}")
        
        # 检查有多少记录成功匹配到了误差数据
        error_cols = [col for col in results_df.columns if '训练集' in col or '验证集' in col]
        if error_cols:
            matched_count = results_df[error_cols[0]].notna().sum()
            print(f"成功匹配到误差数据的公司数: {matched_count}/{len(results_df)}")
    
    return results_df


def save_results(results_df: pd.DataFrame, output_dir: Path, model_name: str, predata_name: str):
    """
    保存结果到Excel和Parquet文件
    
    Args:
        results_df: 结果DataFrame
        output_dir: 输出目录
        model_name: 模型名称
        predata_name: 预测数据名称
    """
    print("\n" + "=" * 80)
    print("保存结果")
    print("=" * 80)
    
    # 创建输出子目录
    output_subdir = output_dir / f"{model_name}_{predata_name}_predictions"
    output_subdir.mkdir(parents=True, exist_ok=True)
    
    # 保存Excel
    excel_path = output_subdir / "predata.xlsx"
    print(f"\n保存Excel文件: {excel_path}")
    results_df.to_excel(excel_path, index=False, engine='openpyxl')
    
    # 保存Parquet
    parquet_path = output_subdir / "predata.parquet"
    print(f"保存Parquet文件: {parquet_path}")
    results_df.to_parquet(parquet_path, index=False)
    
    print(f"\n结果已保存到: {output_subdir}")
    print(f"  - Excel: predata.xlsx")
    print(f"  - Parquet: predata.parquet")
    print(f"  - 记录数: {len(results_df)}")


def main():
    """主函数"""
    # 路径处理
    model_dir = Path(MODEL_DIR)
    predata_file = Path(PREDATA_FILE)
    error_ranking_file = Path(ERROR_RANKING_FILE) if ERROR_RANKING_FILE else None
    output_dir = Path(OUTPUT_DIR)
    
    # 检查路径
    if not model_dir.exists():
        raise FileNotFoundError(f"模型目录不存在: {model_dir}")
    if not predata_file.exists():
        raise FileNotFoundError(f"预测数据文件不存在: {predata_file}")
    
    # 检查设备
    if DEVICE == 'cuda' and not torch.cuda.is_available():
        print("警告: CUDA不可用，使用CPU")
        device = 'cpu'
    else:
        device = DEVICE
    
    # 提取模型名称和预测数据名称
    model_name = model_dir.name
    predata_name = predata_file.parent.name  # 使用文件夹名称
    
    print("=" * 80)
    print("基于预测数据的模型推理脚本 v0.1")
    print("=" * 80)
    print(f"模型目录: {model_dir}")
    print(f"预测数据文件: {predata_file}")
    print(f"误差排名文件: {error_ranking_file}")
    print(f"输出目录: {output_dir}")
    print(f"计算设备: {device}")
    print(f"批次大小: {BATCH_SIZE}")
    print("=" * 80)
    
    # 加载模型
    model = load_model_dynamically(model_dir, device=device)
    
    # 加载预测数据
    predata = load_predata(predata_file)
    
    # 加载误差排名文件
    error_ranking_df = None
    if error_ranking_file and error_ranking_file.exists():
        error_ranking_df = load_error_ranking(error_ranking_file)
    
    # 预测并合并结果
    results_df = predict_and_merge(
        model=model,
        predata=predata,
        error_ranking_df=error_ranking_df,
        device=device,
        batch_size=BATCH_SIZE
    )
    
    # 保存结果
    save_results(results_df, output_dir, model_name, predata_name)
    
    print("\n" + "=" * 80)
    print("任务完成！")
    print("=" * 80)


if __name__ == '__main__':
    main()
