#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
公司数据提取脚本
版本: v0.3
日期: 20260126155425

功能:
1. 从预处理数据中提取指定公司的训练集和验证集数据
2. 为每家公司生成独立的Excel文件
3. 每个样本（输入输出对）生成一个独立的sheet
4. 支持灵活配置数据源和目标公司
5. 【v0.2新增】使用原始特征名作为列名，而不是编号
6. 【v0.3新增】从源文件提取真实日期，替代timestep编号

使用方法:
    1. 修改下面的配置区域
    2. 运行: conda activate /data/project_20251211/.conda_env
    3. 运行: python extract_company_data_v0.3_20260126155425.py
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import re
import json
from collections import OrderedDict

# ===== 配置区域（修改此处） =====
# 数据文件夹路径
PREPROCESSED_DATA_DIR = "/data/project_20251211/data/processed/preprocess_data_NaNto-1000_20260325165454"

# 需要提取的公司名称列表
TARGET_COMPANIES = [
    "贵州茅台",
    "寒武纪-U",
    "TENCENT騰訊控股"
]

# 是否使用模糊匹配（如果公司名称可能有细微差异）
USE_FUZZY_MATCH = False

# 输出文件夹（如果为None，则保存到数据源文件夹）
OUTPUT_DIR = None  # 默认保存到PREPROCESSED_DATA_DIR

# 每个Excel文件最大sheet数量（防止文件过大）
MAX_SHEETS_PER_FILE = 500

# 【v0.3新增】是否从源文件提取真实日期（如果为False，使用timestep编号）
EXTRACT_REAL_DATES = True

# 【v0.3新增】源文件缓存大小（同时缓存多少个源文件，避免内存溢出）
SOURCE_FILE_CACHE_SIZE = 10
# ================================


def load_dates_from_source(source_file_path, start_row, end_row, cache_dict):
    """
    从源parquet文件中提取指定行范围的日期列表
    
    Args:
        source_file_path: 源文件路径
        start_row: 起始行号（包含）
        end_row: 结束行号（包含）
        cache_dict: 缓存字典，用于存储已读取的源文件
    
    Returns:
        dates_list: 日期列表，如 ['2010-01-04', '2010-01-05', ...]
        如果出错返回None
    """
    try:
        source_path = Path(source_file_path)
        
        # 检查文件是否存在
        if not source_path.exists():
            print(f"  警告: 源文件不存在: {source_path}")
            return None
        
        # 检查缓存
        cache_key = str(source_path)
        if cache_key in cache_dict:
            df = cache_dict[cache_key]
        else:
            # 读取源文件（只读取日期列以节省内存）
            df = pd.read_parquet(source_path, columns=['日期'])
            
            # 缓存管理：如果缓存已满，删除最早的条目
            if len(cache_dict) >= SOURCE_FILE_CACHE_SIZE:
                # 删除第一个键（最早添加的）
                oldest_key = next(iter(cache_dict))
                del cache_dict[oldest_key]
            
            cache_dict[cache_key] = df
        
        # 提取指定行范围的日期
        # end_row是包含的，所以需要+1
        dates = df.iloc[start_row:end_row+1]['日期']
        
        # 格式化日期为字符串列表
        dates_list = [pd.to_datetime(d).strftime('%Y-%m-%d') for d in dates]
        
        return dates_list
        
    except Exception as e:
        print(f"  警告: 从源文件提取日期时出错: {e}")
        return None


def load_data_and_index(data_dir: Path):
    """
    加载预处理数据和索引文件
    
    Returns:
        train_data, val_data, train_index_df, val_index_df, feature_info
    """
    print("=" * 80)
    print("加载数据")
    print("=" * 80)
    
    data_dir = Path(data_dir)
    
    # 查找训练集文件
    train_files = sorted(data_dir.glob("train_*.pt"))
    val_files = sorted(data_dir.glob("val_*.pt"))
    
    if not train_files and not val_files:
        raise FileNotFoundError(f"在 {data_dir} 中未找到任何数据文件 (train_*.pt 或 val_*.pt)")
    
    train_data = None
    val_data = None
    train_index_df = None
    val_index_df = None
    feature_info = {}
    
    # 加载训练集
    if train_files:
        train_pt = train_files[-1]  # 使用最新的文件
        print(f"\n加载训练集: {train_pt.name}")
        train_data = torch.load(train_pt, map_location='cpu', weights_only=False)
        print(f"训练集样本数: {train_data['metadata']['num_samples']}")
        
        # 提取特征名称信息
        if 'feature_columns_example' in train_data['metadata']:
            feature_info['feature_columns'] = train_data['metadata']['feature_columns_example']
            print(f"特征列数: {len(feature_info['feature_columns'])}")
        
        if 'sample_feature_columns' in train_data['metadata']:
            feature_info['sample_feature_columns'] = train_data['metadata']['sample_feature_columns']
            print(f"样本特征列映射数: {len(feature_info['sample_feature_columns'])}")
        
        # 加载训练集索引
        train_index_path = Path(train_data['metadata']['index_file'])
        if train_index_path.exists():
            print(f"加载训练集索引: {train_index_path.name}")
            train_index_df = pd.read_parquet(train_index_path)
            print(f"训练集索引记录数: {len(train_index_df)}")
            
            # 【v0.3新增】检查索引文件是否包含必需的日期字段
            required_fields = ['source_file', 'input_row_start', 'input_row_end']
            missing_fields = [f for f in required_fields if f not in train_index_df.columns]
            if missing_fields and EXTRACT_REAL_DATES:
                print(f"警告: 训练集索引缺少字段 {missing_fields}，将无法提取真实日期")
        else:
            print(f"警告: 训练集索引文件不存在: {train_index_path}")
    
    # 加载验证集
    if val_files:
        val_pt = val_files[-1]  # 使用最新的文件
        print(f"\n加载验证集: {val_pt.name}")
        val_data = torch.load(val_pt, map_location='cpu', weights_only=False)
        print(f"验证集样本数: {val_data['metadata']['num_samples']}")
        
        # 如果训练集没有加载特征信息，从验证集加载
        if not feature_info:
            if 'feature_columns_example' in val_data['metadata']:
                feature_info['feature_columns'] = val_data['metadata']['feature_columns_example']
                print(f"特征列数: {len(feature_info['feature_columns'])}")
            
            if 'sample_feature_columns' in val_data['metadata']:
                feature_info['sample_feature_columns'] = val_data['metadata']['sample_feature_columns']
                print(f"样本特征列映射数: {len(feature_info['sample_feature_columns'])}")
        
        # 加载验证集索引
        val_index_path = Path(val_data['metadata']['index_file'])
        if val_index_path.exists():
            print(f"加载验证集索引: {val_index_path.name}")
            val_index_df = pd.read_parquet(val_index_path)
            print(f"验证集索引记录数: {len(val_index_df)}")
            
            # 【v0.3新增】检查索引文件是否包含必需的日期字段
            required_fields = ['source_file', 'input_row_start', 'input_row_end']
            missing_fields = [f for f in required_fields if f not in val_index_df.columns]
            if missing_fields and EXTRACT_REAL_DATES:
                print(f"警告: 验证集索引缺少字段 {missing_fields}，将无法提取真实日期")
        else:
            print(f"警告: 验证集索引文件不存在: {val_index_path}")
    
    return train_data, val_data, train_index_df, val_index_df, feature_info


def show_available_companies(train_index_df, val_index_df, num_to_show=30):
    """显示数据中可用的公司名称"""
    print("\n" + "=" * 80)
    print("数据中可用的公司（前{}个）".format(num_to_show))
    print("=" * 80)
    
    all_companies = set()
    
    if train_index_df is not None:
        all_companies.update(train_index_df['company_name'].unique())
    
    if val_index_df is not None:
        all_companies.update(val_index_df['company_name'].unique())
    
    all_companies = sorted(list(all_companies))
    print(f"\n总公司数: {len(all_companies)}")
    print("\n公司名称列表:")
    for i, company in enumerate(all_companies[:num_to_show], 1):
        print(f"  {i:3d}. {company}")
    
    if len(all_companies) > num_to_show:
        print(f"  ... 还有 {len(all_companies) - num_to_show} 家公司")
    
    return all_companies


def match_companies(target_companies, available_companies, use_fuzzy=False):
    """
    匹配目标公司
    
    Returns:
        matched_companies: 匹配成功的公司名称列表
        unmatched_companies: 未匹配的公司名称列表
    """
    matched = []
    unmatched = []
    
    for target in target_companies:
        if use_fuzzy:
            # 模糊匹配：查找包含目标字符串的公司
            found = False
            for available in available_companies:
                if target.lower() in available.lower() or available.lower() in target.lower():
                    matched.append(available)
                    found = True
                    print(f"✓ 模糊匹配成功: '{target}' -> '{available}'")
                    break
            if not found:
                unmatched.append(target)
                print(f"✗ 未找到匹配: '{target}'")
        else:
            # 精确匹配
            if target in available_companies:
                matched.append(target)
                print(f"✓ 精确匹配成功: '{target}'")
            else:
                unmatched.append(target)
                print(f"✗ 未找到精确匹配: '{target}'")
    
    return matched, unmatched


def extract_company_data(company_name, train_data, val_data, train_index_df, val_index_df, feature_info):
    """
    提取指定公司的所有样本
    
    Returns:
        train_samples: 训练集样本列表
        val_samples: 验证集样本列表
    """
    train_samples = []
    val_samples = []
    
    sample_feature_columns = feature_info.get('sample_feature_columns', [])
    default_feature_columns = feature_info.get('feature_columns', [])
    
    # 【v0.3新增】创建源文件缓存字典（使用OrderedDict保持插入顺序）
    source_file_cache = OrderedDict()
    
    # 【v0.3新增】统计日期提取情况
    dates_extracted_count = 0
    dates_failed_count = 0
    
    # 提取训练集样本
    if train_data is not None and train_index_df is not None:
        company_train_indices = train_index_df[train_index_df['company_name'] == company_name].index.tolist()
        
        print(f"正在提取训练集数据...")
        for idx in tqdm(company_train_indices, desc="  处理训练集样本"):
            sample_info = train_index_df.iloc[idx]
            X = train_data['X'][idx].numpy()  # (seq_len, n_features)
            y = train_data['y'][idx].item()
            
            # 获取该样本的特征列名
            if idx < len(sample_feature_columns):
                feature_names = sample_feature_columns[idx]
            else:
                # 如果没有该样本的特征列信息，使用默认特征列
                feature_names = default_feature_columns[:X.shape[1]] if default_feature_columns else [f'feature_{i}' for i in range(X.shape[1])]
            
            # 【v0.3新增】提取输入序列的真实日期
            input_dates = None
            if EXTRACT_REAL_DATES:
                # 检查必需字段是否存在
                if all(field in sample_info.index for field in ['source_file', 'input_row_start', 'input_row_end']):
                    source_file = sample_info['source_file']
                    start_row = sample_info['input_row_start']
                    end_row = sample_info['input_row_end']
                    
                    # 从源文件提取日期
                    input_dates = load_dates_from_source(source_file, start_row, end_row, source_file_cache)
                    
                    # 验证日期列表长度
                    if input_dates is not None:
                        if len(input_dates) == X.shape[0]:
                            dates_extracted_count += 1
                        else:
                            print(f"  警告: 样本 {idx} 日期数量({len(input_dates)})与序列长度({X.shape[0]})不匹配")
                            input_dates = None
                            dates_failed_count += 1
                    else:
                        dates_failed_count += 1
            
            train_samples.append({
                'index': idx,
                'sample_id': sample_info['sample_id'],
                'company_name': sample_info['company_name'],
                'stock_code': sample_info.get('stock_code', ''),
                'target_date': sample_info.get('target_date', ''),
                'X': X,
                'y': y,
                'feature_names': feature_names,
                'input_dates': input_dates  # 【v0.3新增】
            })
    
    # 提取验证集样本
    if val_data is not None and val_index_df is not None:
        company_val_indices = val_index_df[val_index_df['company_name'] == company_name].index.tolist()
        
        # 验证集的索引需要加上训练集的长度来访问sample_feature_columns
        train_samples_count = len(train_data['X']) if train_data is not None else 0
        
        print(f"正在提取验证集数据...")
        for idx in tqdm(company_val_indices, desc="  处理验证集样本"):
            sample_info = val_index_df.iloc[idx]
            X = val_data['X'][idx].numpy()  # (seq_len, n_features)
            y = val_data['y'][idx].item()
            
            # 获取该样本的特征列名（验证集需要加上训练集的偏移量）
            global_idx = train_samples_count + idx
            if global_idx < len(sample_feature_columns):
                feature_names = sample_feature_columns[global_idx]
            else:
                # 如果没有该样本的特征列信息，使用默认特征列
                feature_names = default_feature_columns[:X.shape[1]] if default_feature_columns else [f'feature_{i}' for i in range(X.shape[1])]
            
            # 【v0.3新增】提取输入序列的真实日期
            input_dates = None
            if EXTRACT_REAL_DATES:
                # 检查必需字段是否存在
                if all(field in sample_info.index for field in ['source_file', 'input_row_start', 'input_row_end']):
                    source_file = sample_info['source_file']
                    start_row = sample_info['input_row_start']
                    end_row = sample_info['input_row_end']
                    
                    # 从源文件提取日期
                    input_dates = load_dates_from_source(source_file, start_row, end_row, source_file_cache)
                    
                    # 验证日期列表长度
                    if input_dates is not None:
                        if len(input_dates) == X.shape[0]:
                            dates_extracted_count += 1
                        else:
                            print(f"  警告: 样本 {idx} 日期数量({len(input_dates)})与序列长度({X.shape[0]})不匹配")
                            input_dates = None
                            dates_failed_count += 1
                    else:
                        dates_failed_count += 1
            
            val_samples.append({
                'index': idx,
                'sample_id': sample_info['sample_id'],
                'company_name': sample_info['company_name'],
                'stock_code': sample_info.get('stock_code', ''),
                'target_date': sample_info.get('target_date', ''),
                'X': X,
                'y': y,
                'feature_names': feature_names,
                'input_dates': input_dates  # 【v0.3新增】
            })
    
    # 【v0.3新增】打印日期提取统计
    if EXTRACT_REAL_DATES:
        total_samples = len(train_samples) + len(val_samples)
        print(f"\n日期提取统计:")
        print(f"  成功提取: {dates_extracted_count}/{total_samples} 个样本")
        if dates_failed_count > 0:
            print(f"  提取失败: {dates_failed_count}/{total_samples} 个样本（将使用timestep编号）")
    
    return train_samples, val_samples


def save_samples_to_excel(company_name, train_samples, val_samples, output_dir, max_sheets=500):
    """
    将样本保存为Excel文件
    每个样本一个sheet
    
    如果样本数超过max_sheets，会分割成多个文件
    """
    # 清理文件名中的特殊字符
    safe_company_name = re.sub(r'[<>:"/\\|?*\s]', '_', company_name)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    
    total_samples = len(train_samples) + len(val_samples)
    
    if total_samples == 0:
        print(f"\n{company_name}: 没有样本数据，跳过")
        return []
    
    # 计算需要生成的文件数量
    num_files = (total_samples + max_sheets - 1) // max_sheets
    
    output_files = []
    
    for file_idx in range(num_files):
        if num_files > 1:
            filename = f"{safe_company_name}_extracted_data_v0.3_{timestamp}_part{file_idx+1}.xlsx"
        else:
            filename = f"{safe_company_name}_extracted_data_v0.3_{timestamp}.xlsx"
        
        output_path = output_dir / filename
        
        # 计算当前文件应包含的样本范围
        start_idx = file_idx * max_sheets
        end_idx = min((file_idx + 1) * max_sheets, total_samples)
        
        print(f"\n生成Excel文件 ({file_idx+1}/{num_files}): {filename}")
        print(f"样本范围: {start_idx+1} - {end_idx} (共 {end_idx - start_idx} 个样本)")
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            current_sample_count = 0
            
            # 写入训练集样本
            for sample in tqdm(train_samples, desc="写入训练集", disable=file_idx > 0):
                if current_sample_count >= start_idx and current_sample_count < end_idx:
                    sheet_name = f"train_{sample['index']}"
                    
                    # 创建元信息
                    meta_info = pd.DataFrame({
                        '字段': ['公司名称', '股票代码', '目标日期', '真实收盘价'],
                        '值': [
                            sample['company_name'],
                            sample['stock_code'],
                            str(sample['target_date']),
                            sample['y']
                        ]
                    })
                    
                    # 创建输入数据DataFrame，使用原始特征名
                    X_df = pd.DataFrame(sample['X'])
                    
                    # 使用实际的特征名作为列名
                    feature_names = sample.get('feature_names', [])
                    if len(feature_names) == X_df.shape[1]:
                        X_df.columns = feature_names
                    else:
                        # 如果特征名数量不匹配，使用编号
                        X_df.columns = [f'feature_{i}' for i in range(X_df.shape[1])]
                    
                    # 【v0.3修改】使用真实日期或timestep编号作为行索引
                    input_dates = sample.get('input_dates')
                    if input_dates is not None and len(input_dates) == X_df.shape[0]:
                        X_df.index = input_dates  # 使用真实日期
                    else:
                        X_df.index = [f'timestep_{i}' for i in range(X_df.shape[0])]  # 回退到timestep
                    
                    # 写入sheet
                    meta_info.to_excel(writer, sheet_name=sheet_name, index=False, startrow=0)
                    X_df.to_excel(writer, sheet_name=sheet_name, startrow=len(meta_info) + 2)
                
                current_sample_count += 1
                if current_sample_count >= end_idx:
                    break
            
            # 写入验证集样本
            for sample in tqdm(val_samples, desc="写入验证集", disable=file_idx > 0):
                if current_sample_count >= start_idx and current_sample_count < end_idx:
                    sheet_name = f"val_{sample['index']}"
                    
                    # 创建元信息
                    meta_info = pd.DataFrame({
                        '字段': ['公司名称', '股票代码', '目标日期', '真实收盘价'],
                        '值': [
                            sample['company_name'],
                            sample['stock_code'],
                            str(sample['target_date']),
                            sample['y']
                        ]
                    })
                    
                    # 创建输入数据DataFrame，使用原始特征名
                    X_df = pd.DataFrame(sample['X'])
                    
                    # 使用实际的特征名作为列名
                    feature_names = sample.get('feature_names', [])
                    if len(feature_names) == X_df.shape[1]:
                        X_df.columns = feature_names
                    else:
                        # 如果特征名数量不匹配，使用编号
                        X_df.columns = [f'feature_{i}' for i in range(X_df.shape[1])]
                    
                    # 【v0.3修改】使用真实日期或timestep编号作为行索引
                    input_dates = sample.get('input_dates')
                    if input_dates is not None and len(input_dates) == X_df.shape[0]:
                        X_df.index = input_dates  # 使用真实日期
                    else:
                        X_df.index = [f'timestep_{i}' for i in range(X_df.shape[0])]  # 回退到timestep
                    
                    # 写入sheet
                    meta_info.to_excel(writer, sheet_name=sheet_name, index=False, startrow=0)
                    X_df.to_excel(writer, sheet_name=sheet_name, startrow=len(meta_info) + 2)
                
                current_sample_count += 1
                if current_sample_count >= end_idx:
                    break
        
        print(f"✓ Excel文件已保存: {output_path}")
        output_files.append(output_path)
    
    return output_files


def generate_summary(matched_companies, extraction_results, output_dir):
    """生成提取摘要"""
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    summary_path = output_dir / f"extraction_summary_{timestamp}.txt"
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("公司数据提取摘要\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"提取时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"数据源: {PREPROCESSED_DATA_DIR}\n")
        f.write(f"输出目录: {output_dir}\n")
        f.write(f"真实日期提取: {'启用' if EXTRACT_REAL_DATES else '禁用'}\n\n")  # 【v0.3新增】
        
        f.write(f"成功提取公司数: {len(matched_companies)}\n\n")
        
        for company_name, result in extraction_results.items():
            f.write("-" * 80 + "\n")
            f.write(f"公司名称: {company_name}\n")
            f.write(f"训练集样本数: {result['train_count']}\n")
            f.write(f"验证集样本数: {result['val_count']}\n")
            f.write(f"总样本数: {result['total_count']}\n")
            f.write(f"生成文件数: {len(result['files'])}\n")
            for file_path in result['files']:
                f.write(f"  - {file_path.name}\n")
            f.write("\n")
    
    print(f"\n✓ 摘要文件已保存: {summary_path}")
    return summary_path


def main():
    """主函数"""
    print("\n" + "=" * 80)
    print("公司数据提取脚本 v0.3_20260126155425")
    print("=" * 80)
    
    # 确定输出目录
    data_dir = Path(PREPROCESSED_DATA_DIR)
    output_dir = Path(OUTPUT_DIR) if OUTPUT_DIR else data_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n数据源: {data_dir}")
    print(f"输出目录: {output_dir}")
    print(f"目标公司数: {len(TARGET_COMPANIES)}")
    print(f"真实日期提取: {'启用' if EXTRACT_REAL_DATES else '禁用'}")  # 【v0.3新增】
    
    # 加载数据
    train_data, val_data, train_index_df, val_index_df, feature_info = load_data_and_index(data_dir)
    
    # 显示可用公司
    available_companies = show_available_companies(train_index_df, val_index_df)
    
    # 匹配目标公司
    print("\n" + "=" * 80)
    print("匹配目标公司")
    print("=" * 80)
    matched_companies, unmatched_companies = match_companies(
        TARGET_COMPANIES, available_companies, USE_FUZZY_MATCH
    )
    
    print(f"\n匹配成功: {len(matched_companies)} 家")
    print(f"未匹配: {len(unmatched_companies)} 家")
    
    if unmatched_companies:
        print("\n未匹配的公司:")
        for company in unmatched_companies:
            print(f"  - {company}")
    
    if not matched_companies:
        print("\n错误: 没有匹配到任何公司，程序退出")
        return
    
    # 提取数据并生成Excel
    print("\n" + "=" * 80)
    print("提取数据并生成Excel文件")
    print("=" * 80)
    
    extraction_results = {}
    
    for company_name in matched_companies:
        print(f"\n处理公司: {company_name}")
        print("-" * 80)
        
        # 提取样本
        train_samples, val_samples = extract_company_data(
            company_name, train_data, val_data, train_index_df, val_index_df, feature_info
        )
        
        print(f"训练集样本数: {len(train_samples)}")
        print(f"验证集样本数: {len(val_samples)}")
        print(f"总样本数: {len(train_samples) + len(val_samples)}")
        
        # 保存Excel
        output_files = save_samples_to_excel(
            company_name, train_samples, val_samples, output_dir, MAX_SHEETS_PER_FILE
        )
        
        extraction_results[company_name] = {
            'train_count': len(train_samples),
            'val_count': len(val_samples),
            'total_count': len(train_samples) + len(val_samples),
            'files': output_files
        }
    
    # 生成摘要
    print("\n" + "=" * 80)
    print("生成摘要")
    print("=" * 80)
    summary_path = generate_summary(matched_companies, extraction_results, output_dir)
    
    # 打印完成信息
    print("\n" + "=" * 80)
    print("提取完成!")
    print("=" * 80)
    print(f"\n成功提取 {len(matched_companies)} 家公司的数据")
    print(f"输出目录: {output_dir}")
    print(f"摘要文件: {summary_path}")


if __name__ == "__main__":
    main()
