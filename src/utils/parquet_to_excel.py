#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Parquet文件转换为Excel文件工具
支持将parquet文件转换为xlsx格式的Excel文件
"""

import pandas as pd
import os
import sys
from pathlib import Path
from datetime import datetime

# ============================================================================
# 配置区域 - 请在此处修改输入和输出目录
# ============================================================================

# 输入文件目录（如果使用目录模式，将转换该目录下所有parquet文件）
INPUT_DIR = r"/data/project_20251211/data/raw/processed_data_20251220/853_CHINA TELECOM中國電信_00728_20251219_202124.parquet"

# 输出文件目录（转换后的Excel文件将保存在此目录）
OUTPUT_DIR = r"/data/project_20251211/data/raw/parquet_to_excel"

# 是否处理子目录（True: 递归处理所有子目录中的parquet文件，False: 只处理指定目录）
PROCESS_SUBDIRS = False

# ============================================================================
# 主程序
# ============================================================================


def convert_parquet_to_excel(parquet_path, output_dir):
    """
    将单个parquet文件转换为Excel文件
    
    参数:
        parquet_path: parquet文件路径（可以是文件路径或目录路径）
        output_dir: 输出目录路径
    """
    parquet_path = Path(parquet_path)
    output_dir = Path(output_dir)
    
    # 确保输出目录存在
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 判断是文件还是目录
    if parquet_path.is_file():
        # 单个文件模式
        if not parquet_path.suffix.lower() == '.parquet':
            print(f"错误: {parquet_path} 不是parquet文件")
            return False
        
        files_to_process = [parquet_path]
    elif parquet_path.is_dir():
        # 目录模式
        if PROCESS_SUBDIRS:
            files_to_process = list(parquet_path.rglob("*.parquet"))
        else:
            files_to_process = list(parquet_path.glob("*.parquet"))
        
        if not files_to_process:
            print(f"错误: 在目录 {parquet_path} 中未找到parquet文件")
            return False
    else:
        print(f"错误: 路径不存在: {parquet_path}")
        return False
    
    # 处理所有文件
    success_count = 0
    fail_count = 0
    
    print("="*80)
    print(f"开始转换，共找到 {len(files_to_process)} 个parquet文件")
    print("="*80)
    
    for i, parquet_file in enumerate(files_to_process, 1):
        try:
            print(f"\n[{i}/{len(files_to_process)}] 正在处理: {parquet_file.name}")
            
            # 读取parquet文件
            print("  正在读取parquet文件...")
            df = pd.read_parquet(parquet_file)
            print(f"  ✓ 读取成功: {df.shape[0]:,} 行 x {df.shape[1]} 列")
            
            # 生成输出文件名（保持原文件名，只改扩展名）
            excel_filename = parquet_file.stem + ".xlsx"
            excel_path = output_dir / excel_filename
            
            # 如果文件已存在，添加时间戳
            if excel_path.exists():
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                excel_filename = f"{parquet_file.stem}_{timestamp}.xlsx"
                excel_path = output_dir / excel_filename
            
            # 转换为Excel
            print(f"  正在转换为Excel: {excel_path.name}")
            df.to_excel(excel_path, index=False, engine='openpyxl')
            
            file_size = excel_path.stat().st_size / (1024 * 1024)  # MB
            print(f"  ✓ 转换成功: {excel_path.name} ({file_size:.2f} MB)")
            
            success_count += 1
            
        except Exception as e:
            print(f"  ✗ 转换失败: {str(e)}")
            fail_count += 1
            import traceback
            traceback.print_exc()
    
    # 输出统计信息
    print("\n" + "="*80)
    print("转换完成！")
    print("="*80)
    print(f"成功: {success_count} 个文件")
    print(f"失败: {fail_count} 个文件")
    print(f"输出目录: {output_dir.absolute()}")
    
    return success_count > 0


def main():
    """主函数"""
    print("="*80)
    print("Parquet转Excel工具")
    print("="*80)
    
    # 检查输入目录是否存在
    input_path = Path(INPUT_DIR)
    if not input_path.exists():
        print(f"\n错误: 输入路径不存在: {INPUT_DIR}")
        print("\n请检查配置区域中的 INPUT_DIR 设置")
        sys.exit(1)
    
    # 执行转换
    success = convert_parquet_to_excel(INPUT_DIR, OUTPUT_DIR)
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()

