#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据预处理脚本 - 保持原始数据不变，仅将NaN和0替换为-1000
版本: NaNto-1000
日期: 20260213

功能:
1. 读取索引文件和原始数据
2. 按公司分组处理所有样本
3. 对所有特征列：将NaN填充为0，再将所有0值替换为-1000
4. 不做任何归一化处理，保留原始数据的真实值
5. 保存为.pt文件（包含tensor和元数据）

核心特点:
- 不做任何归一化，保持原始值不变
- NaN和0值统一标记为-1000（缺失值标记）
- 继承v1.0的内存优化策略（torch.stack + gc.collect + malloc_trim）
- 按公司分组处理，避免一次性加载所有数据

使用方法:
    # 最简单的方式：直接运行（使用代码中的所有默认设置）
    python src/data/parquet_to_tensor/process_data_NaNto-1000_20260213.py
    
    # 或者通过命令行参数覆盖默认设置（可选）
    python src/data/parquet_to_tensor/process_data_NaNto-1000_20260213.py --config_version v0.1_20251212

注意:
    - 所有默认配置都在代码顶部定义（DEFAULT_CONFIG_VERSION等）
    - 如需修改，直接编辑代码中的默认值即可，无需命令行参数
    - 命令行参数仅用于临时覆盖默认设置
"""

import torch
import pandas as pd
import yaml
import argparse
import sys
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime
import importlib.util
import numpy as np
import gc
from typing import List, Optional, Dict
from collections import Counter


# 添加项目路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))


# 动态导入模块
def _load_module(module_path: Path, module_name: str):
    """动态加载模块"""
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# 导入所需模块
utils_path = project_root / "src" / "utils" / "v0.1_20251212"
data_path = project_root / "src" / "data" / "v0.1_20251212"

data_utils_module = _load_module(utils_path / "data_utils.py", "data_utils")
load_parquet_file = data_utils_module.load_parquet_file
get_data_by_rows = data_utils_module.get_data_by_rows
get_data_by_single_row = data_utils_module.get_data_by_single_row

feature_selector_module = _load_module(data_path / "feature_selector.py", "feature_selector")
FeatureSelector = feature_selector_module.FeatureSelector


# ============================================================================
# 默认配置：可在代码中直接修改，无需命令行参数
# ============================================================================
# 脚本版本号（用于生成文件命名）
SCRIPT_VERSION = 'NaNto-1000'

# 配置版本号（用于加载配置文件）
# 配置文件路径: configs/{DEFAULT_CONFIG_VERSION}/dataset_config.yaml
DEFAULT_CONFIG_VERSION = 'v0.1_20251212'

# 默认索引文件路径配置（优先使用，如果设置为None则从config文件读取）
# 设置为具体路径时，直接使用该路径；设置为None时，从config文件中读取
DEFAULT_INDEX_DIR = '/data/project_20251211/data/processed/roll_generate_index_v0.7_20260325_165322'
DEFAULT_TRAIN_INDEX_FILE = 'train_samples_index.parquet'
DEFAULT_VAL_INDEX_FILE = 'val_samples_index.parquet'
# 如果希望从config文件读取索引路径，可以设置为None：
# DEFAULT_INDEX_DIR = None
# DEFAULT_TRAIN_INDEX_FILE = None
# DEFAULT_VAL_INDEX_FILE = None

# 默认输出目录（相对于项目根目录）
DEFAULT_OUTPUT_DIR = 'data/processed'
# ============================================================================


def load_config(config_path: str) -> dict:
    """加载YAML配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def process_company_data(
    company_file: Path,
    company_samples: pd.DataFrame,
    feature_selector: FeatureSelector,
    warning_handler: Optional[callable] = None
) -> List[Dict]:
    """
    处理单个公司的数据：保持原始值不变，仅将NaN和0替换为-1000
    
    Args:
        company_file: 公司数据文件路径
        company_samples: 该公司的所有样本索引（训练集+验证集）
        feature_selector: 特征选择器
        warning_handler: 警告信息处理函数（用于在tqdm进度条中正确显示警告）
    
    Returns:
        样本数据列表
    """
    # 读取整个公司的数据文件
    df = load_parquet_file(str(company_file), use_cache=False)
    
    # 重置特征选择器缓存，确保每个公司使用自己的特征列
    feature_selector.reset()
    
    # 获取特征列（排除exclude_columns）
    feature_columns = feature_selector.get_feature_columns(df)
    
    # 提取样本数据
    samples_data = []
    for _, sample in company_samples.iterrows():
        # 提取输入数据
        input_start = sample['input_row_start']
        input_end = sample['input_row_end']
        input_df = get_data_by_rows(df, input_start, input_end)
        
        # 提取目标数据
        target_row = sample['target_row']
        target_series = get_data_by_single_row(df, target_row)
        target_value = target_series[feature_selector.get_target_column()]
        
        # 提取特征列
        X = input_df[feature_columns].copy()
        
        # 步骤1：将NaN填充为0
        if X.isna().sum().sum() > 0:
            X = X.fillna(0.0)
        
        # 强制转换所有列为数值类型
        for col in X.columns:
            if X[col].dtype == 'object' or not pd.api.types.is_numeric_dtype(X[col]):
                X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0.0)
        
        # 确保所有数据都是float32类型
        try:
            X = X.astype('float32')
        except (ValueError, TypeError) as e:
            warning_msg = f"警告: 样本 {sample['sample_id']} 转换为float32失败: {str(e)}"
            if warning_handler:
                warning_handler(warning_msg)
            else:
                print(f"\n{warning_msg}")
            for col in X.columns:
                try:
                    X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0.0).astype('float32')
                except (ValueError, TypeError):
                    X[col] = 0.0
        
        # 步骤2：将所有特征列中的0值替换为-1000（NaN已在步骤1中变为0）
        for col in feature_columns:
            if col in X.columns:
                col_data = X[col].values.copy()
                col_data[col_data == 0] = -1000.0
                X[col] = col_data
        
        # 转换为tensor
        try:
            X_values = X.values
            if X_values.dtype == 'object':
                X_numeric = X.select_dtypes(include=[np.number])
                X_values = X_numeric.values
            
            X_tensor = torch.FloatTensor(X_values)
        except (TypeError, ValueError) as e:
            error_msg = f"错误: 样本 {sample['sample_id']} 转换失败: {str(e)}"
            if warning_handler:
                warning_handler(error_msg)
            else:
                print(f"\n{error_msg}")
            X_tensor = torch.zeros((len(X), len(feature_columns)), dtype=torch.float32)
        
        samples_data.append({
            'sample_id': sample['sample_id'],
            'X': X_tensor,
            'y': float(target_value),
            'feature_columns': feature_columns
        })
    
    return samples_data


def preprocess_dataset_by_company(
    train_index_file: Path,
    val_index_file: Path,
    feature_selector: FeatureSelector,
    output_dir: Path,
) -> Dict:
    """
    按公司分组预处理数据集（仅将NaN和0替换为-1000，不做归一化）
    
    Args:
        train_index_file: 训练集索引文件路径
        val_index_file: 验证集索引文件路径
        feature_selector: 特征选择器
        output_dir: 输出目录
    
    Returns:
        元数据字典
    """
    print("\n" + "=" * 80)
    print("按公司分组预处理数据集（仅NaN/0→-1000，不做归一化）")
    print("=" * 80)
    
    # 读取索引文件
    train_index_df = pd.read_parquet(train_index_file)
    val_index_df = pd.read_parquet(val_index_file)
    
    print(f"训练集样本数: {len(train_index_df)}")
    print(f"验证集样本数: {len(val_index_df)}")
    
    # 合并索引文件（用于按公司分组）
    all_index_df = pd.concat([train_index_df, val_index_df], ignore_index=True)
    all_index_df['split'] = ['train'] * len(train_index_df) + ['val'] * len(val_index_df)
    
    # 按公司分组（使用source_file作为分组键）
    company_groups = all_index_df.groupby('source_file')
    num_companies = len(company_groups)
    
    # 按文件名中的数字ID对公司分组进行排序
    def extract_numeric_id_from_path(file_path_str):
        """从文件路径中提取数字ID，例如: /path/123_公司名_xxx.parquet -> 123"""
        try:
            file_path = Path(file_path_str)
            filename = file_path.stem
            id_str = filename.split('_')[0]
            return int(id_str)
        except (ValueError, IndexError):
            return 999999
    
    company_groups_sorted = sorted(company_groups, key=lambda x: extract_numeric_id_from_path(x[0]))
    
    print(f"\n总公司数: {num_companies}")
    print("开始按公司处理...")
    
    # 获取序列长度（从第一个样本）
    if len(all_index_df) == 0:
        raise ValueError("索引文件为空")
    
    first_sample = all_index_df.iloc[0]
    seq_len = first_sample['input_row_end'] - first_sample['input_row_start'] + 1
    
    print(f"序列长度: {seq_len}")
    print("特征数量: 每个公司独立确定")
    
    # 存储所有样本数据（按sample_id排序）
    all_samples_dict = {}
    
    # 收集警告信息（避免打断进度条）
    warnings_collected = []
    
    # 处理每个公司（按数字ID排序）
    pbar = tqdm(company_groups_sorted, desc="处理公司数据", total=num_companies)
    for company_file, company_samples in pbar:
        company_file_path = Path(company_file)
        
        if not company_file_path.exists():
            warnings_collected.append(f"警告: 公司文件不存在: {company_file}")
            continue
        
        try:
            samples_data = process_company_data(
                company_file_path,
                company_samples,
                feature_selector,
                warning_handler=lambda msg: warnings_collected.append(msg)
            )
            
            for sample_data in samples_data:
                all_samples_dict[sample_data['sample_id']] = sample_data
        
        except (ValueError, KeyError, IndexError, FileNotFoundError, OSError) as e:
            warnings_collected.append(f"错误: 处理公司 {company_file} 时出错: {str(e)}")
            import traceback
            error_trace = traceback.format_exc()
            warnings_collected.append(error_trace)
            continue
    
    pbar.close()
    gc.collect()
    
    # 统一打印警告信息
    if warnings_collected:
        print("\n" + "=" * 80)
        print(f"处理完成，共收集到 {len(warnings_collected)} 条警告/错误信息:")
        print("=" * 80)
        warning_counts = Counter(warnings_collected)
        unique_warnings = list(warning_counts.keys())
        
        for warning in unique_warnings[:50]:
            count = warning_counts[warning]
            if count > 1:
                print(f"[{count}次] {warning}")
            else:
                print(warning)
        
        if len(unique_warnings) > 50:
            print(f"\n... 还有 {len(unique_warnings) - 50} 条警告未显示")
        print("=" * 80 + "\n")
    
    # 创建训练集和验证集的sample_id集合
    train_sample_ids = set(train_index_df['sample_id'].tolist())
    val_sample_ids = set(val_index_df['sample_id'].tolist())
    
    # 分离训练集和验证集
    train_samples = []
    val_samples = []
    
    for sample_id, sample_data in all_samples_dict.items():
        if sample_id in train_sample_ids:
            train_samples.append(sample_data)
        elif sample_id in val_sample_ids:
            val_samples.append(sample_data)
        else:
            warnings_collected.append(f"警告: 样本 {sample_id} 既不在训练集也不在验证集中")
    
    # 按sample_id排序
    train_samples.sort(key=lambda x: x['sample_id'])
    val_samples.sort(key=lambda x: x['sample_id'])
    
    # 统计所有样本的特征数量分布
    feature_count_distribution = {}
    for sample_data in train_samples + val_samples:
        num_features = len(sample_data['feature_columns'])
        feature_count_distribution[num_features] = feature_count_distribution.get(num_features, 0) + 1
    
    # 打印特征数量分布统计
    print("\n特征数量分布统计：")
    print("=" * 80)
    total_samples_count = len(train_samples) + len(val_samples)
    for num_features in sorted(feature_count_distribution.keys()):
        count = feature_count_distribution[num_features]
        percentage = count / total_samples_count * 100
        print(f"  {num_features}列: {count:,}个样本 ({percentage:.2f}%)")
    print(f"总计: {total_samples_count:,}个样本")
    print("=" * 80)
    
    # 使用第一个样本的特征列作为示例（仅用于metadata）
    if train_samples:
        feature_columns_example = train_samples[0]['feature_columns']
    elif val_samples:
        feature_columns_example = val_samples[0]['feature_columns']
    else:
        raise ValueError("没有有效的样本数据")
    
    # 转换为tensor（采用v1.0/v0.9的torch.stack方式，避免内存碎片化）
    print("\n开始创建训练集tensor...")
    
    del all_samples_dict
    gc.collect()
    
    train_X = torch.stack([s['X'] for s in train_samples])
    train_y = torch.tensor([[s['y']] for s in train_samples], dtype=torch.float32)
    
    for s in train_samples:
        del s['X']
    gc.collect()
    
    print(f"训练集tensor创建完成: {train_X.shape}")
    
    # 在创建验证集之前，强制释放内存
    print("\n释放内存，准备创建验证集...")
    gc.collect()
    gc.collect()
    
    import ctypes
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
        print("已执行 malloc_trim，释放内存给操作系统")
    except Exception:
        pass
    
    print("开始创建验证集tensor...")
    
    val_X = torch.stack([s['X'] for s in val_samples])
    val_y = torch.tensor([[s['y']] for s in val_samples], dtype=torch.float32)
    
    for s in val_samples:
        del s['X']
    gc.collect()
    
    print(f"验证集tensor创建完成: {val_X.shape}")
    
    print(f"\n训练集: {len(train_samples)} 个样本")
    print(f"验证集: {len(val_samples)} 个样本")
    
    # 收集每个样本的特征列名
    train_sample_feature_columns = [list(s['feature_columns']) for s in train_samples]
    val_sample_feature_columns = [list(s['feature_columns']) for s in val_samples]
    
    print(f"已收集所有样本的特征列名映射（训练集: {len(train_sample_feature_columns)}, 验证集: {len(val_sample_feature_columns)}）")
    
    # 保存训练集
    train_output_file = output_dir / f"train_{SCRIPT_VERSION}.pt"
    train_data_to_save = {
        'X': train_X,
        'y': train_y,
        'metadata': {
            'num_samples': int(len(train_samples)),
            'num_features': int(len(feature_columns_example)),
            'seq_len': int(seq_len),
            'feature_columns_example': list(feature_columns_example),
            'sample_feature_columns': train_sample_feature_columns,
            'feature_columns_note': '注意：不同市场的公司具有相同数量但不同名称的特征列（如A股用沪深300，美股用纳斯达克）。sample_feature_columns包含每个样本的实际列名。',
            'feature_count_distribution': {str(k): int(v) for k, v in feature_count_distribution.items()},
            'target_column': str(feature_selector.get_target_column()),
            'exclude_columns': list(feature_selector.exclude_columns),
            'processing_note': '本版本不做任何归一化处理，保留原始数据的真实值。仅将NaN和0值统一替换为-1000（缺失值标记）。',
            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'index_file': str(train_index_file)
        }
    }
    
    print(f"\n保存训练集到: {train_output_file}")
    torch.save(train_data_to_save, train_output_file)
    train_size_mb = train_output_file.stat().st_size / (1024 * 1024)
    print(f"训练集文件大小: {train_size_mb:.2f} MB")
    
    train_metadata = train_data_to_save['metadata'].copy()
    
    del train_data_to_save, train_X, train_y
    gc.collect()
    
    # 保存验证集
    val_output_file = output_dir / f"val_{SCRIPT_VERSION}.pt"
    val_data_to_save = {
        'X': val_X,
        'y': val_y,
        'metadata': {
            'num_samples': int(len(val_samples)),
            'num_features': int(len(feature_columns_example)),
            'seq_len': int(seq_len),
            'feature_columns_example': list(feature_columns_example),
            'sample_feature_columns': val_sample_feature_columns,
            'feature_columns_note': '注意：不同市场的公司具有相同数量但不同名称的特征列（如A股用沪深300，美股用纳斯达克）。sample_feature_columns包含每个样本的实际列名。',
            'feature_count_distribution': {str(k): int(v) for k, v in feature_count_distribution.items()},
            'target_column': str(feature_selector.get_target_column()),
            'exclude_columns': list(feature_selector.exclude_columns),
            'processing_note': '本版本不做任何归一化处理，保留原始数据的真实值。仅将NaN和0值统一替换为-1000（缺失值标记）。',
            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'index_file': str(val_index_file)
        }
    }
    
    print(f"\n保存验证集到: {val_output_file}")
    torch.save(val_data_to_save, val_output_file)
    val_size_mb = val_output_file.stat().st_size / (1024 * 1024)
    print(f"验证集文件大小: {val_size_mb:.2f} MB")
    
    val_metadata = val_data_to_save['metadata'].copy()
    
    del val_data_to_save, val_X, val_y
    gc.collect()
    
    return {
        'train': train_metadata,
        'val': val_metadata,
        'num_companies': num_companies
    }


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='预处理股票数据集（仅NaN/0→-1000，不做归一化）')
    parser.add_argument('--config_version', type=str, default=DEFAULT_CONFIG_VERSION,
                        help=f'配置版本号（默认: {DEFAULT_CONFIG_VERSION}，可在代码中修改 DEFAULT_CONFIG_VERSION）')
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f'输出目录（默认: {DEFAULT_OUTPUT_DIR}，可在代码中修改 DEFAULT_OUTPUT_DIR）')
    
    args = parser.parse_args()
    
    config_version = args.config_version
    output_dir = args.output_dir
    
    print("=" * 80)
    print("数据预处理脚本 NaNto-1000（保持原始数据，仅将NaN和0替换为-1000）")
    print("=" * 80)
    print(f"配置版本: {config_version}")
    print(f"输出目录: {output_dir}")
    print("处理策略: 不做任何归一化，仅将NaN和0值替换为-1000")
    print("=" * 80)
    
    # 加载配置
    config_dir = project_root / "configs" / config_version
    dataset_config_path = config_dir / "dataset_config.yaml"
    
    print(f"\n加载配置: {dataset_config_path}")
    dataset_config = load_config(dataset_config_path)
    dataset_cfg = dataset_config['dataset']
    
    # 获取索引文件路径（优先使用代码中定义的默认路径）
    if DEFAULT_INDEX_DIR is not None:
        index_dir = Path(DEFAULT_INDEX_DIR)
        train_index_file = index_dir / DEFAULT_TRAIN_INDEX_FILE
        val_index_file = index_dir / DEFAULT_VAL_INDEX_FILE
        print(f"使用代码中定义的索引路径:")
        print(f"  索引目录: {index_dir}")
        print(f"  训练集索引: {train_index_file}")
        print(f"  验证集索引: {val_index_file}")
    else:
        index_dir = Path(dataset_cfg['index_dir'])
        train_index_file = index_dir / dataset_cfg['train_index_file']
        val_index_file = index_dir / dataset_cfg['val_index_file']
        print(f"从配置文件读取索引路径:")
        print(f"  索引目录: {index_dir}")
        print(f"  训练集索引: {train_index_file}")
        print(f"  验证集索引: {val_index_file}")
    
    # 获取其他配置参数
    exclude_columns = dataset_cfg['features']['exclude_columns']
    target_column = dataset_cfg['features']['target_column']
    
    # 创建特征选择器
    feature_selector = FeatureSelector(
        exclude_columns=exclude_columns,
        target_column=target_column
    )
    
    # 创建输出目录
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    output_dir_path = project_root / output_dir / f"preprocess_data_{SCRIPT_VERSION}_{timestamp}"
    output_dir_path.mkdir(parents=True, exist_ok=True)
    print(f"\n输出目录: {output_dir_path}")
    
    # 预处理数据集（按公司分组）
    result = preprocess_dataset_by_company(
        train_index_file=train_index_file,
        val_index_file=val_index_file,
        feature_selector=feature_selector,
        output_dir=output_dir_path,
    )
    
    # 保存元数据汇总
    summary = {
        'config_version': config_version,
        'version': SCRIPT_VERSION,
        'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'processing_config': {
            'exclude_columns': exclude_columns,
            'processing_note': '不做任何归一化处理，保留原始数据的真实值。仅将NaN和0值统一替换为-1000（缺失值标记）。',
        },
        'memory_optimization': '继承v1.0/v0.9的内存优化：使用torch.stack()快速创建tensor避免内存碎片化，并在创建验证集前强制释放内存（gc.collect + malloc_trim）',
        'train': result['train'],
        'val': result['val'],
        'num_companies': result['num_companies'],
        'files': {
            'train': str(output_dir_path / f"train_{SCRIPT_VERSION}.pt"),
            'val': str(output_dir_path / f"val_{SCRIPT_VERSION}.pt")
        }
    }
    
    summary_file = output_dir_path / f"preprocess_summary_{SCRIPT_VERSION}.json"
    
    def convert_to_json_serializable(obj):
        """递归转换对象为JSON可序列化的类型"""
        if isinstance(obj, dict):
            return {k: convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj
    
    summary_serializable = convert_to_json_serializable(summary)
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_serializable, f, ensure_ascii=False, indent=2)
    
    print("\n" + "=" * 80)
    print("预处理完成！")
    print("=" * 80)
    print(f"训练集: {output_dir_path / f'train_{SCRIPT_VERSION}.pt'}")
    print(f"验证集: {output_dir_path / f'val_{SCRIPT_VERSION}.pt'}")
    print(f"汇总信息: {summary_file}")
    print("=" * 80)
    
    # 磁盘占用
    train_size_mb = (output_dir_path / f"train_{SCRIPT_VERSION}.pt").stat().st_size / (1024 * 1024)
    val_size_mb = (output_dir_path / f"val_{SCRIPT_VERSION}.pt").stat().st_size / (1024 * 1024)
    total_size_mb = train_size_mb + val_size_mb
    
    print("\n磁盘占用:")
    print(f"  训练集: {train_size_mb:.2f} MB")
    print(f"  验证集: {val_size_mb:.2f} MB")
    print(f"  总计: {total_size_mb:.2f} MB")
    print(f"\n内存占用估算: ~{total_size_mb * 1.2:.2f} MB (训练时)")
    
    print(f"\n处理的公司数: {result['num_companies']}")
    print("\n本版本说明:")
    print("  - 不做任何归一化处理，保留原始数据的真实值")
    print("  - 仅将NaN和0值统一替换为-1000（缺失值标记）")
    print("  - 继承v1.0/v0.9的内存优化特性")


if __name__ == '__main__':
    main()
