#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Parquet数据转换为预测数据脚本（无归一化版本，NaN和0替换为-1000）
版本: v0.01
日期: 20260214

功能：
1. 读取原始parquet文件夹中的所有公司数据
2. 提取每个公司最后500个交易日的数据
3. 不进行归一化处理，保持原始数值
4. 生成单个.pt文件用于模型预测
5. 生成详细的JSON元数据（包含每个样本的日期序列）

处理方式：
- 排除列：日期、company_name、sequence_id、stock_code
- NaN填充为-1000
- 0值替换为-1000
- 所有特征列保持原始数值，不做归一化处理

输出：
- {源文件夹名}.pt: 预测数据tensor
- {源文件夹名}_summary.json: 详细元数据（包含日期信息）

使用方法：
    python parquet_to_predata_v0.01_20260214.py
"""

import torch
import pandas as pd
import argparse
import sys
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime
import numpy as np
import gc
from typing import List, Dict, Tuple


# ============================================================================
# 默认配置
# ============================================================================
SCRIPT_VERSION = 'v0.01'

# 默认输入目录（原始parquet数据文件夹）
DEFAULT_SOURCE_DIR = '/data/project_20251211/data/raw/processed_data_20260308'

# 默认输出目录
DEFAULT_OUTPUT_DIR = '/data/project_20251211/data/pre_data'

# 序列长度（最近N个交易日）
DEFAULT_SEQ_LEN = 500

# 列分类配置
EXCLUDE_COLUMNS = [
    "日期",
    "company_name",
    "sequence_id",
    "stock_code"
]

# ============================================================================


def get_feature_columns(df: pd.DataFrame, exclude_columns: List[str]) -> List[str]:
    """
    获取特征列（排除指定列）
    
    Args:
        df: 数据DataFrame
        exclude_columns: 要排除的列列表
    
    Returns:
        特征列列表
    """
    all_columns = df.columns.tolist()
    feature_columns = [col for col in all_columns if col not in exclude_columns]
    return feature_columns


def process_company_file(
    file_path: Path,
    seq_len: int,
    exclude_columns: List[str]
) -> Tuple[torch.Tensor, Dict, List[str]]:
    """
    处理单个公司的parquet文件（v0.01：无归一化版本，NaN和0替换为-1000）
    
    Args:
        file_path: parquet文件路径
        seq_len: 序列长度
        exclude_columns: 排除列列表
    
    Returns:
        (数据tensor, 公司信息字典, 日期列表) 或 (None, None, None) 如果数据不足
    """
    try:
        # 读取parquet文件
        df = pd.read_parquet(file_path)
        
        # 检查数据量是否足够
        if len(df) < seq_len:
            return None, None, None
        
        # 按日期排序（升序）
        if '日期' in df.columns:
            df['日期'] = pd.to_datetime(df['日期'], errors='coerce')
            df = df.sort_values('日期', ascending=True).reset_index(drop=True)
        
        # 提取最后seq_len行
        df_last = df.tail(seq_len).copy()
        
        # 提取日期列表（转换为字符串格式）
        dates = []
        if '日期' in df_last.columns:
            dates = [str(d) if pd.notna(d) else None for d in df_last['日期'].tolist()]
        
        # 提取公司信息
        company_info = {}
        if 'company_name' in df_last.columns:
            company_info['company_name'] = str(df_last['company_name'].iloc[0])
        if 'stock_code' in df_last.columns:
            company_info['stock_code'] = str(df_last['stock_code'].iloc[0])
        if 'sequence_id' in df_last.columns:
            company_info['sequence_id'] = int(df_last['sequence_id'].iloc[0])
        
        # 获取特征列
        feature_columns = get_feature_columns(df_last, exclude_columns)
        
        # 提取特征数据
        X = df_last[feature_columns].copy()
        
        # 处理NaN（填充为-1000）
        if X.isna().sum().sum() > 0:
            X = X.fillna(-1000.0)
        
        # 强制转换所有列为数值类型
        for col in X.columns:
            if X[col].dtype == 'object' or not pd.api.types.is_numeric_dtype(X[col]):
                X[col] = pd.to_numeric(X[col], errors='coerce').fillna(-1000.0)
        
        # 确保所有数据都是float32类型
        X = X.astype('float32')
        
        # 将0值替换为-1000
        X = X.replace(0.0, -1000.0)
        
        # 转换为tensor（不做归一化处理，保持原始数值）
        X_values = X.values
        X_tensor = torch.FloatTensor(X_values)
        
        company_info['source_file'] = str(file_path)
        company_info['num_rows'] = len(df)
        company_info['selected_rows'] = seq_len
        
        return X_tensor, company_info, dates
        
    except Exception as e:
        print(f"\n错误: 处理文件 {file_path.name} 失败: {str(e)}")
        return None, None, None


def get_all_parquet_files(source_dir: Path) -> List[Path]:
    """
    获取目录下所有parquet文件并按数字ID排序
    
    Args:
        source_dir: 源目录
    
    Returns:
        排序后的parquet文件列表
    """
    parquet_files = list(source_dir.glob("*.parquet"))
    
    # 按文件名中的数字ID排序
    def extract_numeric_id(file_path: Path) -> int:
        """从文件名中提取数字ID"""
        try:
            filename = file_path.stem
            # 提取第一个下划线之前的数字
            id_str = filename.split('_')[0]
            return int(id_str)
        except (ValueError, IndexError):
            # 如果无法提取数字，返回一个很大的数（排在最后）
            return 999999
    
    parquet_files.sort(key=extract_numeric_id)
    
    return parquet_files


def process_all_files(
    source_dir: Path,
    seq_len: int,
    exclude_columns: List[str]
) -> Tuple[torch.Tensor, Dict]:
    """
    处理所有parquet文件
    
    Args:
        source_dir: 源目录
        seq_len: 序列长度
        exclude_columns: 排除列列表
    
    Returns:
        (合并的tensor, 元数据字典)
    """
    print("\n" + "=" * 80)
    print("开始处理Parquet文件")
    print("=" * 80)
    
    # 获取所有parquet文件
    parquet_files = get_all_parquet_files(source_dir)
    total_files = len(parquet_files)
    
    print(f"找到 {total_files} 个parquet文件")
    print(f"序列长度: {seq_len}")
    print(f"数据不足{seq_len}行的公司将被跳过")
    print(f"处理方式: 保持原始数值，不做归一化，NaN和0替换为-1000")
    
    if total_files == 0:
        raise ValueError(f"在 {source_dir} 中未找到任何parquet文件")
    
    # 存储所有样本数据
    samples_data = []
    company_list = []
    skipped_companies = []
    
    # 处理每个文件
    pbar = tqdm(parquet_files, desc="处理公司数据")
    for file_path in pbar:
        X_tensor, company_info, dates = process_company_file(
            file_path,
            seq_len,
            exclude_columns
        )
        
        if X_tensor is None:
            skipped_companies.append({
                'file': file_path.name,
                'reason': f'数据不足{seq_len}行'
            })
            continue
        
        # 记录样本数据
        sample_idx = len(samples_data)
        samples_data.append(X_tensor)
        
        # 记录公司信息
        company_record = {
            'sample_idx': sample_idx,
            'company_name': company_info.get('company_name', 'Unknown'),
            'stock_code': company_info.get('stock_code', 'Unknown'),
            'sequence_id': company_info.get('sequence_id', -1),
            'source_file': company_info['source_file'],
            'total_rows': company_info['num_rows'],
            'selected_rows': company_info['selected_rows'],
            'dates': dates,  # 添加日期列表
            'date_range': {
                'start': dates[0] if dates else None,
                'end': dates[-1] if dates else None
            }
        }
        company_list.append(company_record)
        
        # 更新进度条描述
        pbar.set_postfix({
            'processed': len(samples_data),
            'skipped': len(skipped_companies)
        })
    
    pbar.close()
    
    # 打印统计信息
    print("\n" + "=" * 80)
    print("处理统计")
    print("=" * 80)
    print(f"成功处理: {len(samples_data)} 个公司")
    print(f"跳过公司: {len(skipped_companies)} 个")
    
    if skipped_companies:
        print("\n跳过的公司列表（前10个）:")
        for skip_info in skipped_companies[:10]:
            print(f"  - {skip_info['file']}: {skip_info['reason']}")
        if len(skipped_companies) > 10:
            print(f"  ... 还有 {len(skipped_companies) - 10} 个未显示")
    
    if len(samples_data) == 0:
        raise ValueError("没有任何公司的数据符合要求（至少需要500行数据）")
    
    # 合并所有样本为单个tensor
    print("\n创建合并tensor...")
    all_X = torch.stack(samples_data)
    
    # 释放内存
    del samples_data
    gc.collect()
    
    print(f"合并tensor形状: {all_X.shape}")
    print(f"  - 样本数: {all_X.shape[0]}")
    print(f"  - 序列长度: {all_X.shape[1]}")
    print(f"  - 特征数: {all_X.shape[2]}")
    
    # 获取特征列示例（从第一个公司的文件读取）
    first_file = parquet_files[0]
    df_example = pd.read_parquet(first_file)
    feature_columns_example = get_feature_columns(df_example, exclude_columns)
    
    # 构建元数据
    metadata = {
        'version': SCRIPT_VERSION,
        'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'source_folder': source_dir.name,
        'source_path': str(source_dir),
        'num_companies': len(company_list),
        'num_samples': len(company_list),
        'num_features': all_X.shape[2],
        'seq_len': seq_len,
        'feature_columns_example': feature_columns_example,
        'company_info': company_list,  # 包含每个公司的详细信息和日期列表
        'skipped_companies': skipped_companies,
        'column_processing_config': {
            'exclude_columns': exclude_columns,
            'normalization': False,
            'processing_note': 'v0.01版本：所有特征列保持原始数值，不做归一化处理。NaN填充为-1000，0值替换为-1000。'
        }
    }
    
    return all_X, metadata


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='将Parquet数据转换为预测数据（.pt格式，无归一化版本，NaN和0替换为-1000）'
    )
    parser.add_argument(
        '--source_dir',
        type=str,
        default=DEFAULT_SOURCE_DIR,
        help=f'源数据目录（默认: {DEFAULT_SOURCE_DIR}）'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f'输出目录（默认: {DEFAULT_OUTPUT_DIR}）'
    )
    parser.add_argument(
        '--seq_len',
        type=int,
        default=DEFAULT_SEQ_LEN,
        help=f'序列长度（默认: {DEFAULT_SEQ_LEN}）'
    )
    
    args = parser.parse_args()
    
    # 路径处理
    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    seq_len = args.seq_len
    
    print("=" * 80)
    print("Parquet数据转换为预测数据脚本（无归一化版本，NaN和0替换为-1000）")
    print(f"版本: {SCRIPT_VERSION}")
    print("=" * 80)
    print(f"源目录: {source_dir}")
    print(f"输出目录: {output_dir}")
    print(f"序列长度: {seq_len}")
    print(f"归一化处理: 否（保持原始数值）")
    print(f"NaN和0值处理: 替换为-1000")
    print("=" * 80)
    
    # 检查源目录
    if not source_dir.exists():
        raise FileNotFoundError(f"源目录不存在: {source_dir}")
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 处理所有文件
    all_X, metadata = process_all_files(
        source_dir,
        seq_len,
        EXCLUDE_COLUMNS
    )
    
    # 创建以源文件夹名命名的输出子目录（添加版本号后缀）
    folder_name = f"{source_dir.name}_{SCRIPT_VERSION}"
    output_subdir = output_dir / folder_name
    output_subdir.mkdir(parents=True, exist_ok=True)
    
    # 文件统一命名为 data.pt 和 data_summary.json
    output_pt_file = output_subdir / "data.pt"
    output_json_file = output_subdir / "data_summary.json"
    
    # 保存.pt文件
    print("\n" + "=" * 80)
    print("保存数据")
    print("=" * 80)
    
    data_to_save = {
        'X': all_X,
        'metadata': metadata
    }
    
    print(f"保存tensor到: {output_pt_file}")
    torch.save(data_to_save, output_pt_file)
    
    file_size_mb = output_pt_file.stat().st_size / (1024 * 1024)
    print(f"文件大小: {file_size_mb:.2f} MB")
    
    # 保存JSON元数据
    print(f"保存元数据到: {output_json_file}")
    
    # 将numpy类型转换为Python原生类型
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
    
    metadata_serializable = convert_to_json_serializable(metadata)
    
    with open(output_json_file, 'w', encoding='utf-8') as f:
        json.dump(metadata_serializable, f, ensure_ascii=False, indent=2)
    
    print("\n" + "=" * 80)
    print("转换完成！")
    print("=" * 80)
    print(f"输出文件:")
    print(f"  - Tensor数据: {output_pt_file}")
    print(f"  - 元数据: {output_json_file}")
    print(f"\n数据统计:")
    print(f"  - 公司数: {metadata['num_companies']}")
    print(f"  - 样本数: {metadata['num_samples']}")
    print(f"  - 特征数: {metadata['num_features']}")
    print(f"  - 序列长度: {metadata['seq_len']}")
    print(f"  - 跳过公司数: {len(metadata['skipped_companies'])}")
    print(f"\n数据处理:")
    print(f"  - 归一化: 否")
    print(f"  - 0值替换: 是（替换为-1000）")
    print(f"  - NaN填充: 是（填充为-1000）")
    print("=" * 80)


if __name__ == '__main__':
    main()
