#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
财务数据获取测试脚本 v3.21 - NOTICE_DATE混合获取版（使用eastmoney_v0.5）

核心功能：
- ✅ 分批保存：每N家公司生成一个Excel文件（默认20家/批）
- ✅ 断点续传：支持中断后继续运行，不丢失进度
- ✅ 索引文件：生成company_index.xlsx，记录所有公司的处理状态和存储位置
- ✅ 进度追踪：自动保存JSON进度文件
- ✅ 错误隔离：单个公司失败不影响其他公司
- ✅ 延迟控制：每家公司间隔N秒，避免请求过于频繁
- ✅ 时间戳文件夹：每次运行自动创建独立的输出文件夹
- ✅ 详细失败记录：记录每家公司的财务指标和财务报表获取状态
- ✨ 多周期自动重试：默认周期失败时自动尝试其他周期（继承自v3.2）
- ✨ NOTICE_DATE混合获取：API优先+规则补充（v3.21新增）

v3.21 相比 v3.2 的改进：
- 使用 eastmoney_v0.5.py（支持港股NOTICE_DATE API获取）
- NOTICE_DATE混合获取方案：
  * 港股：首先尝试API获取真实披露日期，然后对缺失值进行规则计算补充
  * A股：使用规则计算方法生成NOTICE_DATE（财报季末+60天）
  * 美股：使用规则计算方法生成NOTICE_DATE（财报季末+60天）
- 逐行检查并补充缺失的NOTICE_DATE（不覆盖已有值）
- 支持所有市场类型的NOTICE_DATE处理

继承自v3.2的功能：
- 自动周期重试机制：
  * A股indicator：按单季度 → 按报告期
  * A股statements：无周期（不重试）
  * 港股indicator：报告期 → 年度
  * 港股statements：报告期 → 年度
  * 美股indicator：单季报 → 累计季报 → 年报
  * 美股statements：单季报 → 累计季报 → 年报
- 索引文件中记录实际使用的数据周期
- indicator和statements独立重试，互不影响

数据处理功能：
- 数据映射：根据映射Excel文件进行列筛选和重命名
- 自动添加缺失字段（填充为0）
- 格式化日期列为 YYYY-MM-DD 格式
- 合并重复的REPORT_DATE行（保留最新数据）
- 计算自由现金流 (FCF = 经营活动现金流 - 资本性支出)
- 填充所有缺失值为0
- 自动合并重复列名（XX, XX_1, XX_2等）

使用方法：
1. 准备股票列表文件（CSV或Excel格式）
2. 配置参数（在main()函数中）
3. 运行脚本：python financial_data_mapper_v3.21_batch_period.py

依赖文件：
- 映射文件: 东方财富财务数据API映射最终版-陈俊同-20251030.xlsx
- 股票列表文件: stock_list.csv 或 Excel文件
- 数据获取模块: src/providers/eastmoney_v0.5.py
"""

import sys
import os
import re
import json
import time
from datetime import datetime
import pandas as pd
from collections import defaultdict

# 添加src目录到Python路径
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# 导入改进版数据获取函数
# 注意：Python导入时，文件名中的点(.)需要改为下划线(_)
import importlib.util

# 动态导入 eastmoney_financial.py 模块（支持周期参数 + 港股NOTICE_DATE API获取）
spec = importlib.util.spec_from_file_location(
    "eastmoney_financial",
    os.path.join(os.path.dirname(__file__), 'eastmoney_financial.py')
)
eastmoney_financial = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eastmoney_financial)

# 获取需要的函数（v0.5支持周期参数 + 港股NOTICE_DATE API获取，返回 (DataFrame, dict)）
get_full_financial_data_a = eastmoney_financial.get_full_financial_data_a
get_full_financial_data_hk = eastmoney_financial.get_full_financial_data_hk
get_full_financial_data_us = eastmoney_financial.get_full_financial_data_us
get_hk_financial_report_dates = eastmoney_financial.get_hk_financial_report_dates  # v3.21新增：港股NOTICE_DATE API获取


def calculate_notice_date_for_market(report_date_str, market_type):
    """根据REPORT_DATE和市场类型计算NOTICE_DATE（v3.21扩展：支持所有市场）
    
    规则：
    - A股：
      * 12月报表 → 次年3月第三周周三
      * 非12月报表 → 报表日后第五周周三
    - 港股：
      * 12月报表 → 次年3月第三周周三
      * 非12月报表 → 月份+2第三周周三
    - 美股：
      * 12月报表 → 次年2月第二周周五
      * 非12月报表 → 报表日后第五周周一
    
    Args:
        report_date_str: 报告日期字符串（如 "2024-12-31"）
        market_type: 市场类型（'A', 'HK', 'US'）
        
    Returns:
        NOTICE_DATE字符串，格式如 "2025-03-19"；失败返回 None
    """
    try:
        report_date = pd.to_datetime(report_date_str)
        
        if market_type == 'A':
            # A股规则
            if report_date.month == 12:
                # 12月报表 → 次年3月第三周周三
                notice_year = report_date.year + 1
                notice_month = 3
                # 找到目标月份的第一天
                first_day = pd.Timestamp(year=notice_year, month=notice_month, day=1)
                # 找到第一个周三（周三是weekday=2）
                days_until_wednesday = (2 - first_day.weekday()) % 7
                if days_until_wednesday == 0 and first_day.weekday() != 2:
                    days_until_wednesday = 7
                first_wednesday = first_day + pd.Timedelta(days=days_until_wednesday)
                # 第三个周三 = 第一个周三 + 14天
                third_wednesday = first_wednesday + pd.Timedelta(days=14)
                return third_wednesday.strftime('%Y-%m-%d')
            else:
                # 非12月报表 → 报表日后第五周周三
                # 找到报表日后的第一个周三
                days_after = (2 - report_date.weekday()) % 7
                if days_after == 0:
                    days_after = 7
                first_wednesday_after = report_date + pd.Timedelta(days=days_after)
                # 第五周周三 = 第一个周三 + 28天（4周）
                fifth_wednesday = first_wednesday_after + pd.Timedelta(days=28)
                return fifth_wednesday.strftime('%Y-%m-%d')
        
        elif market_type == 'HK':
            # 港股规则
            if report_date.month == 12:
                # 12月报表 → 次年3月第三周周三
                notice_year = report_date.year + 1
                notice_month = 3
            else:
                # 非12月报表 → 月份+2第三周周三
                notice_month = report_date.month + 2
                notice_year = report_date.year
                # 如果月份超过12，需要进位到下一年
                if notice_month > 12:
                    notice_month -= 12
                    notice_year += 1
            
            # 找到目标月份的第一天
            first_day = pd.Timestamp(year=notice_year, month=notice_month, day=1)
            # 找到第一个周三（周三是weekday=2）
            days_until_wednesday = (2 - first_day.weekday()) % 7
            if days_until_wednesday == 0 and first_day.weekday() != 2:
                days_until_wednesday = 7
            first_wednesday = first_day + pd.Timedelta(days=days_until_wednesday)
            # 第三个周三 = 第一个周三 + 14天
            third_wednesday = first_wednesday + pd.Timedelta(days=14)
            return third_wednesday.strftime('%Y-%m-%d')
        
        elif market_type == 'US':
            # 美股规则
            if report_date.month == 12:
                # 12月报表 → 次年2月第二周周五
                notice_year = report_date.year + 1
                notice_month = 2
                # 找到目标月份的第一天
                first_day = pd.Timestamp(year=notice_year, month=notice_month, day=1)
                # 找到第一个周五（周五是weekday=4）
                days_until_friday = (4 - first_day.weekday()) % 7
                if days_until_friday == 0 and first_day.weekday() != 4:
                    days_until_friday = 7
                first_friday = first_day + pd.Timedelta(days=days_until_friday)
                # 第二周周五 = 第一个周五 + 7天
                second_friday = first_friday + pd.Timedelta(days=7)
                return second_friday.strftime('%Y-%m-%d')
            else:
                # 非12月报表 → 报表日后第五周周一
                # 找到报表日后的第一个周一
                days_after = (0 - report_date.weekday()) % 7
                if days_after == 0:
                    days_after = 7
                first_monday_after = report_date + pd.Timedelta(days=days_after)
                # 第五周周一 = 第一个周一 + 28天（4周）
                fifth_monday = first_monday_after + pd.Timedelta(days=28)
                return fifth_monday.strftime('%Y-%m-%d')
        
        else:
            print(f"  警告：不支持的市场类型 {market_type}")
            return None
    
    except Exception as e:
        print(f"  警告：计算NOTICE_DATE失败 (REPORT_DATE={report_date_str}, market={market_type}): {e}")
        return None


def detect_market_type(stock_code):
    """自动判断股票代码属于哪个市场
    
    Returns: 'A' (A股), 'HK' (港股), 'US' (美股), 或 'UNKNOWN' (未知)
    """
    stock_code = str(stock_code).strip()
    
    if stock_code.isdigit() and len(stock_code) == 5:
        return 'HK'
    
    if stock_code.isdigit() and len(stock_code) == 6:
        a_stock_prefixes = ['000', '001', '002', '003', '300', '600', '601', '603', '605', '688', '689', '830']
        if any(stock_code.startswith(prefix) for prefix in a_stock_prefixes):
            return 'A'
    
    if not stock_code.isdigit():
        return 'US'
    
    return 'UNKNOWN'


def extract_stock_code_from_name(company_name):
    """从公司名称中提取股票代码
    
    公司名称格式可能是：
    - "公司名称_股票代码" (如 "小米集团_01810")
    - "股票代码_市场类型" (如 "01810_港股")
    """
    parts = company_name.split('_')
    if len(parts) < 2:
        return None
    
    # 尝试最后一个部分（可能是股票代码）
    last_part = parts[-1]
    if detect_market_type(last_part) != 'UNKNOWN':
        return last_part
    
    # 尝试第二个部分（通常是股票代码）
    if len(parts) >= 2:
        second_part = parts[1]
        if detect_market_type(second_part) != 'UNKNOWN':
            return second_part
    
    return None


def format_date_columns(df):
    """格式化日期列，去掉时间部分，只保留日期（YYYY-MM-DD格式）
    
    处理所有包含"日期"关键字的列，以及REPORT_DATE和NOTICE_DATE
    """
    if df.empty:
        return df
    
    df = df.copy()
    
    # 查找所有包含"日期"的列名，以及特定的日期列
    date_columns = [col for col in df.columns if '日期' in col or col in ['REPORT_DATE', 'NOTICE_DATE']]
    
    for col in date_columns:
        # 转换为日期时间格式，然后只保留日期部分
        df[col] = pd.to_datetime(df[col], errors='coerce').dt.strftime('%Y-%m-%d')
    
    return df


def merge_duplicate_report_dates(df):
    """合并重复的REPORT_DATE，只保留第一次出现的行（最新数据）
    
    注意：此函数在映射前调用，使用原始列名 REPORT_DATE
    
    因为df已经按日期从新到旧排序，keep='first'会保留最新的数据
    """
    if df.empty:
        return df
    
    date_col = 'REPORT_DATE'
    
    if date_col not in df.columns:
        print("  - 未找到 REPORT_DATE 列，跳过重复日期合并")
        return df
    
    duplicate_count = df[date_col].duplicated().sum()
    
    if duplicate_count == 0:
        print("  - 未发现重复的财报截止日期")
        return df
    
    print(f"  - 发现 {duplicate_count} 行重复日期，保留最新数据...")
    original_count = len(df)
    
    # keep='first' 保留第一次出现（因为已按日期从新到旧排序）
    df = df.drop_duplicates(subset=[date_col], keep='first')
    
    removed_count = original_count - len(df)
    print(f"  - 合并完成：{original_count} 行 → {len(df)} 行（删除 {removed_count} 行）")
    
    return df


def fill_missing_values_with_zero(df):
    """将DataFrame中所有数值列的缺失值填充为0
    
    排除日期列和文本列（如"数据类型"），只对数值列进行填充
    
    Args:
        df: DataFrame
        
    Returns:
        填充后的DataFrame
    """
    if df.empty:
        return df
    
    df = df.copy()
    
    # 识别需要排除的列（日期列和文本列）
    excluded_cols = set()
    for col in df.columns:
        # 排除日期列（包含"日期"关键字）
        if '日期' in col:
            excluded_cols.add(col)
        # 排除文本列
        elif col == '数据类型' or col == '提示':
            excluded_cols.add(col)
    
    # 获取数值列
    numeric_cols = []
    for col in df.columns:
        if col not in excluded_cols:
            try:
                # 确保获取的是Series而不是DataFrame（处理列名重复的情况）
                col_data = df[col]
                if not isinstance(col_data, pd.Series):
                    # 如果返回的是DataFrame（列名重复），跳过
                    continue
                
                # 尝试转换为数值类型，如果成功则认为是数值列
                if col_data.dtype in ['int64', 'float64', 'int32', 'float32']:
                    numeric_cols.append(col)
                else:
                    # 对于 object 类型，尝试判断是否为数值类型（可能是字符串形式的数字）
                    try:
                        # 检查是否有非空值，如果有则尝试转换
                        sample_values = col_data.dropna()
                        if len(sample_values) > 0:
                            # 尝试转换第一个非空值
                            pd.to_numeric(sample_values.iloc[0])
                            numeric_cols.append(col)
                    except (ValueError, TypeError):
                        # 无法转换为数值，跳过
                        pass
            except Exception as e:
                # 如果处理列时出错，跳过该列
                print(f"  ⚠️ 处理列 '{col}' 时出错: {e}，跳过")
                continue
    
    # 对数值列填充缺失值
    if numeric_cols:
        filled_count = 0
        for col in numeric_cols:
            try:
                # 确保获取的是Series而不是DataFrame
                col_data = df[col]
                if not isinstance(col_data, pd.Series):
                    # 如果返回的是DataFrame（列名重复），跳过
                    continue
                
                before_count = col_data.isna().sum()
                if before_count > 0:
                    df[col] = col_data.fillna(0)
                    filled_count += before_count
            except Exception as e:
                # 如果填充列时出错，跳过该列
                print(f"  ⚠️ 填充列 '{col}' 时出错: {e}，跳过")
                continue
        
        if filled_count > 0:
            print(f"  - 已将 {filled_count} 个缺失值填充为 0（涉及 {len(numeric_cols)} 个数值列）")
    
        return df
    

def add_notice_date_column(df, stock_code, market_type):
    """为数据添加 NOTICE_DATE 列（v3.21：混合方案，API优先+规则补充）
    
    混合方案说明：
    1. 对港股：首先尝试API获取真实披露日期，与现有数据合并
    2. 对所有市场：逐行检查NOTICE_DATE，对缺失值使用规则计算补充
    
    注意：此函数在映射前调用，使用原始列名 REPORT_DATE
    
    Args:
        df: DataFrame（必须包含 REPORT_DATE 列）
        stock_code: 股票代码
        market_type: 市场类型（'A', 'HK', 'US'）
        
    Returns:
        添加/补充了NOTICE_DATE列的DataFrame
    """
    if df.empty:
        return df
    
    date_col = 'REPORT_DATE'
    notice_col = 'NOTICE_DATE'
    
    if date_col not in df.columns:
        print(f"  - {stock_code} 的数据中未找到 REPORT_DATE 列，跳过添加 NOTICE_DATE")
        return df
    
    market_name = {'A': 'A股', 'HK': '港股', 'US': '美股'}.get(market_type, market_type)
    print(f"  - 正在为 {market_name} {stock_code} 处理 NOTICE_DATE...")
    df = df.copy()
    
    # 步骤1: 确保NOTICE_DATE列存在
    if notice_col not in df.columns:
        df[notice_col] = None
        print(f"    └─ 创建 NOTICE_DATE 列")
    
    # 步骤2: 对港股，尝试API获取真实披露日期
    if market_type == 'HK':
        try:
            print(f"    └─ 尝试从API获取真实披露日期...")
            api_dates_df = get_hk_financial_report_dates(stock_code)
            
            if api_dates_df is not None and not api_dates_df.empty:
                # 合并API数据（左连接，保留df的所有行）
                df = df.merge(
                    api_dates_df, 
                    on=date_col, 
                    how='left', 
                    suffixes=('', '_api')
                )
                
                # 用API数据填充原来为空的NOTICE_DATE
                if f'{notice_col}_api' in df.columns:
                    mask = df[notice_col].isna() & df[f'{notice_col}_api'].notna()
                    df.loc[mask, notice_col] = df.loc[mask, f'{notice_col}_api']
                    df = df.drop(columns=[f'{notice_col}_api'])
                    
                    api_filled_count = mask.sum()
                    print(f"    └─ ✓ API获取成功，填充了 {api_filled_count} 条真实披露日期")
            else:
                print(f"    └─ ⚠ API未返回数据，将使用规则计算")
        except Exception as e:
            print(f"    └─ ⚠ API获取失败: {e}，将使用规则计算")
    
    # 步骤3: 对所有市场，逐行检查并补充缺失的NOTICE_DATE
    missing_mask = df[notice_col].isna() | (df[notice_col] == '') | (df[notice_col] == 'None')
    missing_count = missing_mask.sum()
    
    if missing_count > 0:
        print(f"    └─ 检测到 {missing_count} 条缺失的NOTICE_DATE，使用规则计算补充...")
        
        # 对缺失的行，使用规则计算
        for idx in df[missing_mask].index:
            report_date = df.loc[idx, date_col]
            if pd.notna(report_date) and report_date != '':
                calculated_notice_date = calculate_notice_date_for_market(report_date, market_type)
                if calculated_notice_date:
                    df.loc[idx, notice_col] = calculated_notice_date
        
        # 统计补充后还有多少缺失
        still_missing = df[notice_col].isna().sum()
        filled_by_rules = missing_count - still_missing
        print(f"    └─ ✓ 规则计算补充了 {filled_by_rules} 条NOTICE_DATE")
        
        if still_missing > 0:
            print(f"    └─ ⚠ 仍有 {still_missing} 条无法计算NOTICE_DATE（REPORT_DATE可能无效）")
    else:
        print(f"    └─ ✓ 所有NOTICE_DATE已完整，无需补充")
    
    # 步骤4: 确保NOTICE_DATE列在REPORT_DATE后面
    if notice_col in df.columns:
        cols = list(df.columns)
        if notice_col != cols[cols.index(date_col) + 1] if cols.index(date_col) + 1 < len(cols) else True:
            cols.remove(notice_col)
            date_idx = cols.index(date_col)
            cols.insert(date_idx + 1, notice_col)
            df = df[cols]
    
    return df


class FinancialDataMapper:
    """财务数据映射器"""
    
    def __init__(self, mapping_file_path):
        """初始化映射器
        
        Args:
            mapping_file_path: 映射Excel文件路径
        """
        self.mapping_file_path = mapping_file_path
        self.mapping_df = None
        self.a_stock_mapping = {}  # A股：原始列名 -> 统一命名
        self.h_stock_mapping = {}  # H股：原始列名 -> 统一命名
        self.usa_stock_mapping = {}  # USA股：原始列名 -> 统一命名
        self.unified_name_order = []  # 保存映射文件中统一命名的顺序
        
        self._load_mapping()
    
    def _load_mapping(self):
        """加载映射文件并构建映射字典"""
        print(f"正在加载映射文件: {self.mapping_file_path}")
        
        try:
            self.mapping_df = pd.read_excel(self.mapping_file_path)
            print(f"✓ 映射文件加载成功，共 {len(self.mapping_df)} 条映射规则")
            
            # 构建映射字典并保存统一命名的顺序
            for _, row in self.mapping_df.iterrows():
                a_col = str(row['A股']).strip() if pd.notna(row['A股']) else None
                h_col = str(row['H股']).strip() if pd.notna(row['H股']) else None
                usa_col = str(row['USA股']).strip() if pd.notna(row['USA股']) else None
                unified_name = str(row['统一命名']).strip() if pd.notna(row['统一命名']) else None
                
                if unified_name:
                    # 保存统一命名的顺序（去重）
                    if unified_name not in self.unified_name_order:
                        self.unified_name_order.append(unified_name)
                    
                    # 将所有字段都作为普通映射处理
                    if a_col:
                        self.a_stock_mapping[a_col] = unified_name
                    if h_col:
                        self.h_stock_mapping[h_col] = unified_name
                    if usa_col:
                        self.usa_stock_mapping[usa_col] = unified_name
            
            print(f"  - A股映射规则: {len(self.a_stock_mapping)} 条")
            print(f"  - H股映射规则: {len(self.h_stock_mapping)} 条")
            print(f"  - USA股映射规则: {len(self.usa_stock_mapping)} 条")
            print(f"  - 统一命名顺序: {len(self.unified_name_order)} 个字段")
            
        except Exception as e:
            print(f"✗ 加载映射文件失败: {e}")
            raise
    
    def get_expected_columns_for_market(self, market_type):
        """获取指定市场类型应该有的所有统一命名字段
        
        Args:
            market_type: 'A' (A股), 'HK' (港股), 'US' (美股)
            
        Returns:
            该市场类型所有字段的统一命名列表（按顺序）
        """
        mapping_dict = self.get_mapping_for_market(market_type)
        expected_names = list(set(mapping_dict.values()))
        # 按照unified_name_order排序
        ordered_names = [name for name in self.unified_name_order if name in expected_names]
        return ordered_names
    
    def get_mapping_for_market(self, market_type):
        """根据市场类型获取对应的映射字典
        
        Args:
            market_type: 'A' (A股), 'HK' (港股), 'US' (美股)
            
        Returns:
            映射字典 {原始列名: 统一命名}
        """
        # 支持两种格式的市场类型标识
        market_map = {
            'A': self.a_stock_mapping,
            'HK': self.h_stock_mapping,
            'US': self.usa_stock_mapping,
            'A股': self.a_stock_mapping,
            '港股': self.h_stock_mapping,
            '美股': self.usa_stock_mapping
        }
        return market_map.get(market_type, {})
    
    def map_dataframe(self, df, market_type, stock_code):
        """对单个DataFrame进行映射（筛选列并重命名）
        
        Args:
            df: 原始DataFrame
            market_type: 市场类型 ('A', 'HK', 'US')
            stock_code: 股票代码
            
        Returns:
            映射后的DataFrame
        """
        if df.empty:
            return df
        
        mapping_dict = self.get_mapping_for_market(market_type)
        
        if not mapping_dict:
            print(f"  ⚠ {stock_code} 未找到对应的映射规则（市场类型: {market_type}）")
            return df
        
        # 找出DataFrame中存在且在映射字典中的列
        available_columns = []
        rename_dict = {}
        
        for col in df.columns:
            if col in mapping_dict:
                # 直接匹配
                unified_name = mapping_dict[col]
                available_columns.append(col)
                rename_dict[col] = unified_name
            else:
                # 智能匹配：检查是否是带后缀的列（如 存货_x, 存货_y, 存货_1, 存货_2）
                # 去除后缀后检查是否在映射字典中
                base_col = self._get_base_column_name(col)
                if base_col != col and base_col in mapping_dict:
                    # 找到了带后缀的匹配列
                    unified_name = mapping_dict[base_col]
                    # 如果统一命名已经被使用，则跳过（优先使用不带后缀的）
                    if unified_name not in rename_dict.values():
                        available_columns.append(col)
                        rename_dict[col] = unified_name
        
        # 筛选列并重命名
        if available_columns:
            filtered_df = df[available_columns].copy()
            mapped_df = filtered_df.rename(columns=rename_dict)
            print(f"  - {stock_code} 映射完成：{len(df.columns)} 列 → {len(mapped_df.columns)} 列")
            print(f"    └─ 成功映射: {len(available_columns)} 列")
        else:
            # 如果没有可映射的列，创建空DataFrame
            mapped_df = pd.DataFrame()
            print(f"  - {stock_code} 映射完成：{len(df.columns)} 列 → 0 列")
            print("    └─ 警告：没有找到可映射的列")
        
        # 按照映射文件中的顺序排列列（使用 unified_name_order）
        final_columns = [col for col in self.unified_name_order if col in mapped_df.columns]
        mapped_df = mapped_df[final_columns]
        
        return mapped_df
    
    def _get_base_column_name(self, col_name):
        """获取列名的基础名称（去掉后缀）
        
        处理以下情况：
        - 去除 _x, _y 后缀（pandas merge自动添加的）
        - 去除 _数字 后缀（如 _1, _2）
        
        Args:
            col_name: 列名
            
        Returns:
            基础列名
        """
        # 去除 _x 或 _y 后缀
        if col_name.endswith('_x') or col_name.endswith('_y'):
            return col_name[:-2]
        
        # 去除 _数字 后缀（如 _1, _2, _3）
        match = re.match(r'^(.+)_(\d+)$', col_name)
        if match:
            return match.group(1)
        
        return col_name


def add_missing_columns_and_sort(df, market_type, mapper):
    """添加缺失的列（填充0）并按映射文件顺序排列
    
    Args:
        df: 映射后的DataFrame
        market_type: 市场类型
        mapper: FinancialDataMapper对象
        
    Returns:
        添加缺失列并排序后的DataFrame
    """
    if df.empty or not mapper:
        return df
    
    df = df.copy()
    
    # 获取该市场类型应该有的所有列
    expected_columns_list = mapper.get_expected_columns_for_market(market_type)
    expected_columns = set(expected_columns_list)
    existing_columns = set(df.columns)
    missing_columns = expected_columns - existing_columns
    
    # 添加缺失列
    if missing_columns:
        for col_name in missing_columns:
            df[col_name] = 0
        print(f"  - 添加了 {len(missing_columns)} 个缺失列（填充值：0）")
    
    # 按映射文件顺序排列列
    final_cols = [col for col in expected_columns_list if col in df.columns]
    df = df[final_cols]
    print(f"  - 已按映射文件顺序排列列（共 {len(final_cols)} 列）")
    
    return df


class BatchProcessor:
    """批量处理器 - 支持分批保存、断点续传、索引生成"""
    
    def __init__(self, mapping_file_path, batch_size=20, output_dir=None, delay_seconds=3):
        """初始化批量处理器
        
        Args:
            mapping_file_path: 映射文件路径
            batch_size: 每批处理的公司数量（默认20）
            output_dir: 输出目录（默认为当前目录下的output文件夹）
            delay_seconds: 每家公司处理完后的延迟秒数（默认3秒，避免请求过于频繁）
        """
        self.mapping_file_path = mapping_file_path
        self.batch_size = batch_size
        self.output_dir = output_dir or os.path.join(os.getcwd(), "output")
        self.delay_seconds = delay_seconds
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 生成时间戳（用于所有文件名）
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 初始化映射器
        print(f"正在加载映射规则...")
        self.mapper = FinancialDataMapper(mapping_file_path)
        
        # 初始化索引数据
        self.company_index = []  # 公司索引列表
        
        # 进度文件路径
        self.progress_file = os.path.join(self.output_dir, f"progress_{self.timestamp}.json")
        self.processed_codes = set()  # 已处理的股票代码
        
        # 当前批次信息
        self.current_batch_num = 0
        self.current_batch_file = None
        self.current_batch_writer = None
        self.current_batch_count = 0
        
        # 统计信息
        self.total_processed = 0
        self.total_success = 0
        self.total_failed = 0
        self.failed_list = []  # [(代码, 名称, 错误信息)]
        
        # 断点续传标志
        self.is_resumed = False  # 标记是否从断点恢复
        self.first_batch_after_resume = False  # 标记恢复后的第一个batch
    
    def _get_batch_filename(self, batch_num):
        """获取批次文件名（xlsx格式）"""
        return os.path.join(
            self.output_dir, 
            f"mapped_data_batch_{batch_num}_{self.timestamp}.xlsx"
        )
    
    def _get_index_filename(self):
        """获取索引文件名"""
        return os.path.join(self.output_dir, f"company_index_{self.timestamp}.xlsx")
    
    def _save_progress(self):
        """保存进度到JSON文件"""
        # 保存旧的进度文件路径
        old_progress_file = getattr(self, 'progress_file', None)
        
        # 使用当前时间生成新的进度文件名
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_progress_file = os.path.join(self.output_dir, f"progress_{current_time}.json")
        
        progress_data = {
            'timestamp': self.timestamp,  # 保留原始开始时间戳（用于数据文件命名）
            'save_time': current_time,  # 记录保存时间
            'total_processed': self.total_processed,
            'total_success': self.total_success,
            'total_failed': self.total_failed,
            'processed_codes': list(self.processed_codes),
            'failed_list': self.failed_list,
            'last_batch_num': self.current_batch_num,
            'current_batch_count': self.current_batch_count  # 保存当前批次已处理数量
        }
        
        # 先写入新文件
        try:
            with open(new_progress_file, 'w', encoding='utf-8') as f:
                json.dump(progress_data, f, ensure_ascii=False, indent=2)
            
            # 写入成功后，更新进度文件路径
            self.progress_file = new_progress_file
            
            # 删除旧的进度文件
            if old_progress_file and old_progress_file != new_progress_file and os.path.exists(old_progress_file):
                try:
                    os.remove(old_progress_file)
                except:
                    pass
                    
        except Exception as e:
            print(f"  ⚠️ 警告：保存进度文件失败: {e}")
            if old_progress_file:
                self.progress_file = old_progress_file
    
    def _load_progress(self, progress_file):
        """从JSON文件加载进度"""
        try:
            with open(progress_file, 'r', encoding='utf-8') as f:
                progress_data = json.load(f)
            
            self.timestamp = progress_data['timestamp']
            self.total_processed = progress_data['total_processed']
            self.total_success = progress_data['total_success']
            self.total_failed = progress_data['total_failed']
            self.processed_codes = set(progress_data['processed_codes'])
            self.failed_list = progress_data['failed_list']
            self.current_batch_num = progress_data['last_batch_num']
            self.current_batch_count = progress_data.get('current_batch_count', 0)
            
            # 更新文件路径
            self.progress_file = progress_file
            
            # 尝试从已存在的索引文件中恢复之前的索引数据
            index_file = self._get_index_filename()
            if os.path.exists(index_file):
                try:
                    index_df = pd.read_excel(index_file, sheet_name='公司索引')
                    self.company_index = index_df.to_dict('records')
                    print(f"✓ 已从索引文件恢复 {len(self.company_index)} 家公司的索引信息")
                except Exception as e:
                    print(f"  ⚠️ 警告：读取索引文件失败: {e}，将重新生成索引")
                    self.company_index = []
            else:
                print(f"  ℹ️  索引文件不存在，将创建新索引")
                self.company_index = []
            
            # 标记为断点续传模式
            self.is_resumed = True
            if self.current_batch_count < self.batch_size and self.current_batch_count > 0:
                self.first_batch_after_resume = True
            
            print(f"✓ 已加载进度：已处理 {self.total_processed} 家公司")
            print(f"  成功: {self.total_success}, 失败: {self.total_failed}")
            print(f"  当前批次 {self.current_batch_num}：已有 {self.current_batch_count} 家公司")
            
            return True
        except Exception as e:
            print(f"✗ 加载进度文件失败: {e}")
            return False
    
    def _start_new_batch(self, is_resume=False):
        """开始新的批次
        
        Args:
            is_resume: 是否是断点续传恢复的批次
        """
        # 关闭上一个批次的writer
        if self.current_batch_writer is not None:
            try:
                self.current_batch_writer.close()
            except:
                pass
        
        # 如果不是恢复模式，增加批次号
        if not is_resume:
            self.current_batch_num += 1
            self.current_batch_count = 0
        
        self.current_batch_file = self._get_batch_filename(self.current_batch_num)
        
        # 判断是追加模式还是新建模式
        if is_resume and os.path.exists(self.current_batch_file):
            # 断点续传：以追加模式打开已存在的Excel文件
            self.current_batch_writer = pd.ExcelWriter(
                self.current_batch_file, 
                engine='openpyxl',
                mode='a',
                if_sheet_exists='replace'
            )
            print(f"\n{'='*60}")
            print(f"续传批次 {self.current_batch_num}：{os.path.basename(self.current_batch_file)}")
            print(f"  已有 {self.current_batch_count} 家公司，继续追加...")
            print(f"{'='*60}")
        else:
            # 新建模式：创建新的Excel文件
            self.current_batch_writer = pd.ExcelWriter(self.current_batch_file, engine='openpyxl')
            print(f"\n{'='*60}")
            print(f"开始批次 {self.current_batch_num}：{os.path.basename(self.current_batch_file)}")
            print(f"{'='*60}")
    
    def _save_company_to_current_batch(self, company_name, stock_code, merged_df, market_type, status):
        """将公司数据保存到当前批次的xlsx文件
        
        Args:
            company_name: 公司名称
            stock_code: 股票代码
            merged_df: 数据DataFrame
            market_type: 市场类型
            status: eastmoney_v0.4返回的状态信息
        """
        # 生成sheet名称
        sheet_name = f"{company_name}_{stock_code}"
        
        # 如果sheet名称太长，截断（Excel限制31个字符）
        if len(sheet_name) > 31:
            sheet_name = f"{company_name[:10]}_{stock_code}"
        
        try:
            # 保存到当前批次的writer
            merged_df.to_excel(self.current_batch_writer, sheet_name=sheet_name, index=False)
            
            # 立即保存
            self.current_batch_writer.close()
            
            # 重新打开writer以便继续添加数据
            if os.path.exists(self.current_batch_file):
                self.current_batch_writer = pd.ExcelWriter(
                    self.current_batch_file, 
                    engine='openpyxl',
                    mode='a',
                    if_sheet_exists='replace'
                )
            else:
                self.current_batch_writer = pd.ExcelWriter(
                    self.current_batch_file, 
                    engine='openpyxl'
                )
            
            # 更新索引（v3.2：添加周期信息）
            market_name = {'A': 'A股', 'HK': '港股', 'US': '美股', 'UNKNOWN': '未知'}[market_type]
            self.company_index.append({
                '序号': self.total_processed + 1,
                '公司名称': company_name,
                '股票代码': stock_code,
                '市场类型': market_name,
                '状态': '成功',
                'Excel文件': os.path.basename(self.current_batch_file),
                'Sheet名称': sheet_name,
                '批次号': self.current_batch_num,
                '数据行数': len(merged_df),
                '数据列数': len(merged_df.columns),
                '错误信息': '',
                # v3.2新增：详细的状态信息（包含周期）
                '财务指标状态': '成功' if status['indicator']['success'] else '失败',
                '财务指标错误': status['indicator']['error'] or '',
                '财务指标周期': status['indicator'].get('period_used', ''),  # v3.2新增
                '财务报表状态': '成功' if status['statements']['success'] else '失败',
                '财务报表错误': status['statements']['error'] or '',
                '财务报表周期': status['statements'].get('period_used', ''),  # v3.2新增
                '数据来源': ', '.join(status['data_sources'])
            })
            
            # 更新计数
            self.current_batch_count += 1
            self.total_success += 1
            
            # 显示数据来源信息
            data_sources_str = ', '.join(status['data_sources'])
            print(f"  ✓ 已保存 {sheet_name} ({len(merged_df)} 行 × {len(merged_df.columns)} 列)")
            print(f"    └─ 数据来源: {data_sources_str}")
            # v3.2: 显示周期信息
            if status['indicator'].get('period_used'):
                print(f"    └─ 财务指标周期: {status['indicator']['period_used']}")
            if status['statements'].get('period_used'):
                print(f"    └─ 财务报表周期: {status['statements']['period_used']}")
            
            return True
            
        except Exception as e:
            print(f"  ✗ 保存 {sheet_name} 失败: {e}")
            return False
    
    def _save_index_file(self):
        """保存索引文件（XLSX格式）"""
        if not self.company_index:
            print("  ⚠ 没有可保存的索引数据")
            return
        
        index_file = self._get_index_filename()
        index_df = pd.DataFrame(self.company_index)
        
        # 按序号排序
        if '序号' in index_df.columns:
            index_df = index_df.sort_values('序号')
        
        # 保存为XLSX格式
        try:
            with pd.ExcelWriter(index_file, engine='openpyxl') as writer:
                index_df.to_excel(writer, sheet_name='公司索引', index=False)
                
                # 自动调整列宽
                worksheet = writer.sheets['公司索引']
                for idx, col in enumerate(index_df.columns):
                    try:
                        col_data = index_df[col]
                        if isinstance(col_data, pd.Series):
                            max_length = max(
                                col_data.astype(str).map(len).max(),
                                len(str(col))
                            ) + 2
                            # v3.2：限制最大宽度为60（因为有较长的字段）
                            worksheet.column_dimensions[chr(65 + idx)].width = min(max_length, 60)
                    except:
                        worksheet.column_dimensions[chr(65 + idx)].width = 15
        except Exception as e:
            print(f"  ⚠️ 保存索引文件时出错: {e}")
            return
        
        # 统计成功和失败数量
        success_count = len(index_df[index_df['状态'] == '成功'])
        failed_count = len(index_df[index_df['状态'] == '失败'])
        
        print(f"\n✓ 索引文件已更新: {os.path.basename(index_file)}")
        print(f"  共 {len(self.company_index)} 家公司 (成功: {success_count}, 失败: {failed_count})")
    
    def process_single_company(self, stock_code, company_name=''):
        """处理单个公司的数据"""
        # 检查是否已处理过
        if stock_code in self.processed_codes:
            print(f"  ⊙ {stock_code} 已处理过，跳过")
            return True
        
        # 检测市场类型
        market_type = detect_market_type(stock_code)
        market_name = {'A': 'A股', 'HK': '港股', 'US': '美股', 'UNKNOWN': '未知'}[market_type]
        
        print(f"\n[{self.total_processed + 1}] 处理 {market_name} {company_name if company_name else stock_code} ({stock_code})")
        
        try:
            # 1. 获取原始数据（调用v3.2的函数，返回data和status）
            result = get_single_stock_data(stock_code, company_name)
            df = result['data']
            status = result['status']  # v3.2：获取状态信息（包含周期）
            # stats = result['stats']  # 统计信息，当前未使用
            
            if df.empty:
                print(f"  ⚠ {stock_code} 未获取到数据")
                error_msg = "未获取到数据"
                self.failed_list.append((stock_code, company_name, error_msg))
                
                # 添加失败记录到索引（v3.2：包含详细状态和周期）
                self.company_index.append({
                    '序号': self.total_processed + 1,
                    '公司名称': company_name if company_name else stock_code,
                    '股票代码': stock_code,
                    '市场类型': market_name,
                    '状态': '失败',
                    'Excel文件': '',
                    'Sheet名称': '',
                    '批次号': '',
                    '数据行数': 0,
                    '数据列数': 0,
                    '错误信息': error_msg,
                    '财务指标状态': '失败' if status and not status['indicator']['success'] else '未获取',
                    '财务指标错误': status['indicator']['error'] if status and status['indicator']['error'] else '',
                    '财务指标周期': status['indicator'].get('period_used', '') if status else '',
                    '财务报表状态': '失败' if status and not status['statements']['success'] else '未获取',
                    '财务报表错误': status['statements']['error'] if status and status['statements']['error'] else '',
                    '财务报表周期': status['statements'].get('period_used', '') if status else '',
                    '数据来源': ''
                })
                
                self.total_failed += 1
                self.processed_codes.add(stock_code)
                self.total_processed += 1
                self._save_progress()
                self._save_index_file()
                
                return False
            
            # 2. 数据处理流程（使用v3.2的函数）
            print(f"  - 正在处理数据...")
            
            # 2.1 排序
            if 'REPORT_DATE' in df.columns:
                df = df.sort_values('REPORT_DATE', ascending=False)
            
            # 2.2 格式化日期列
            df = format_date_columns(df)
            
            # 2.3 处理NOTICE_DATE（v3.21：支持所有市场，混合方案）
            df = add_notice_date_column(df, stock_code, market_type)
            
            # 2.4 合并重复的REPORT_DATE
            df = merge_duplicate_report_dates(df)
            
            # 2.5 应用映射
            print(f"  - 正在应用映射规则...")
            df = self.mapper.map_dataframe(df, market_type, stock_code)
            
            # 2.6 添加缺失列并排序
            df = add_missing_columns_and_sort(df, market_type, self.mapper)
            
            # 2.7 计算自由现金流
            df = calculate_free_cash_flow(df, stock_code)
            
            # 2.8 填充缺失值
            print(f"  - 正在填充缺失值...")
            df = fill_missing_values_with_zero(df)
            
            # 3. 检查是否需要开始新批次
            if self.current_batch_writer is None or self.current_batch_count >= self.batch_size:
                is_resume_batch = self.first_batch_after_resume
                if is_resume_batch:
                    self.first_batch_after_resume = False
                self._start_new_batch(is_resume=is_resume_batch)
            
            # 4. 保存到当前批次（v3.2：传递status信息）
            success = self._save_company_to_current_batch(
                company_name if company_name else stock_code,
                stock_code,
                df,
                market_type,
                status  # v3.2：传递状态信息（包含周期）
            )
            
            if not success:
                error_msg = "保存失败"
                self.failed_list.append((stock_code, company_name, error_msg))
                
                # 移除之前添加的成功记录
                if self.company_index and self.company_index[-1]['股票代码'] == stock_code:
                    self.company_index.pop()
                
                # 添加失败记录（v3.2：包含详细状态和周期）
                self.company_index.append({
                    '序号': self.total_processed + 1,
                    '公司名称': company_name if company_name else stock_code,
                    '股票代码': stock_code,
                    '市场类型': market_name,
                    '状态': '失败',
                    'Excel文件': '',
                    'Sheet名称': '',
                    '批次号': '',
                    '数据行数': 0,
                    '数据列数': 0,
                    '错误信息': error_msg,
                    '财务指标状态': '成功' if status['indicator']['success'] else '失败',
                    '财务指标错误': status['indicator']['error'] or '',
                    '财务指标周期': status['indicator'].get('period_used', ''),
                    '财务报表状态': '成功' if status['statements']['success'] else '失败',
                    '财务报表错误': status['statements']['error'] or '',
                    '财务报表周期': status['statements'].get('period_used', ''),
                    '数据来源': ', '.join(status['data_sources'])
                })
                
                self.total_failed += 1
                self.total_success -= 1
            
            # 5. 更新进度
            self.processed_codes.add(stock_code)
            self.total_processed += 1
            self._save_progress()
            self._save_index_file()
            
            return success
            
        except Exception as e:
            print(f"  ✗ {stock_code} 处理失败: {e}")
            error_msg = str(e)
            self.failed_list.append((stock_code, company_name, error_msg))
            
            # 添加失败记录到索引（v3.2：包含详细状态和周期）
            self.company_index.append({
                '序号': self.total_processed + 1,
                '公司名称': company_name if company_name else stock_code,
                '股票代码': stock_code,
                '市场类型': market_name,
                '状态': '失败',
                'Excel文件': '',
                'Sheet名称': '',
                '批次号': '',
                '数据行数': 0,
                '数据列数': 0,
                '错误信息': error_msg,
                '财务指标状态': '未获取',
                '财务指标错误': '',
                '财务指标周期': '',
                '财务报表状态': '未获取',
                '财务报表错误': '',
                '财务报表周期': '',
                '数据来源': ''
            })
            
            self.total_failed += 1
            self.processed_codes.add(stock_code)
            self.total_processed += 1
            self._save_progress()
            self._save_index_file()
            
            return False
    
    def process_stock_list(self, stock_list, resume_from_progress=None):
        """批量处理股票列表"""
        # 如果提供了进度文件，加载进度
        if resume_from_progress and os.path.exists(resume_from_progress):
            print("检测到进度文件，正在恢复...")
            if self._load_progress(resume_from_progress):
                # 过滤掉已处理的股票
                stock_list = [(code, name) for code, name in stock_list 
                             if code not in self.processed_codes]
                print(f"剩余 {len(stock_list)} 家公司待处理\n")
        
        print(f"\n{'='*60}")
        print(f"开始批量处理")
        print(f"{'='*60}")
        print(f"总数量: {len(stock_list)} 家公司")
        print(f"批次大小: {self.batch_size} 家/批")
        print(f"预计生成: {(len(stock_list) + self.batch_size - 1) // self.batch_size} 个xlsx文件")
        print(f"输出目录: {self.output_dir}")
        print(f"延迟时间: 每家公司间隔 {self.delay_seconds} 秒")
        print(f"{'='*60}\n")
        
        # 开始处理
        start_time = datetime.now()
        
        for idx, (stock_code, company_name) in enumerate(stock_list, 1):
            try:
                self.process_single_company(stock_code, company_name)
                
                # 每10家公司显示一次进度
                if idx % 10 == 0:
                    elapsed = (datetime.now() - start_time).total_seconds()
                    avg_time = elapsed / idx
                    remaining = avg_time * (len(stock_list) - idx)
                    print(f"\n{'─'*60}")
                    print(f"进度: {idx}/{len(stock_list)} ({idx/len(stock_list)*100:.1f}%)")
                    print(f"成功: {self.total_success}, 失败: {self.total_failed}")
                    print(f"预计剩余时间: {remaining/60:.1f} 分钟")
                    print(f"{'─'*60}\n")
                
                # 间隔指定秒数
                if idx < len(stock_list):
                    print(f"  ⏳ 等待 {self.delay_seconds} 秒后处理下一家公司...\n")
                    time.sleep(self.delay_seconds)
                    
            except KeyboardInterrupt:
                print("\n\n检测到中断信号，正在保存进度...")
                self._finalize()
                print("\n✓ 进度已保存！")
                print(f"进度文件: {os.path.basename(self.progress_file)}")
                print(f"完整路径: {self.progress_file}")
                print("\n💡 要继续处理，请在main()函数中设置:")
                print(f'   resume_from_progress = "{self.progress_file}"')
                sys.exit(0)
            except Exception as e:
                print(f"\n✗ 处理 {stock_code} 时发生未知错误: {e}")
                continue
        
        # 完成处理
        self._finalize()
        
        # 显示最终统计
        elapsed = (datetime.now() - start_time).total_seconds()
        print(f"\n{'='*60}")
        print(f"处理完成")
        print(f"{'='*60}")
        print(f"总耗时: {elapsed/60:.1f} 分钟")
        print(f"总处理: {self.total_processed} 家公司")
        print(f"成功: {self.total_success} 家")
        print(f"失败: {self.total_failed} 家")
        print(f"生成文件: {self.current_batch_num} 个xlsx文件")
        print(f"{'='*60}\n")
        
        # 显示失败列表
        if self.failed_list:
            print("失败列表:")
            for code, name, error in self.failed_list:
                print(f"  - {name if name else code} ({code}): {error}")
            print()
        
        return self.timestamp
    
    def _finalize(self):
        """完成处理，保存索引和关闭文件"""
        # 关闭最后一个批次的writer
        if self.current_batch_writer is not None:
            try:
                self.current_batch_writer.close()
            except:
                pass
        
        # 保存索引文件
        self._save_index_file()
        
        # 保存最终进度
        self._save_progress()
        
        print(f"\n✓ 进度文件已保存: {os.path.basename(self.progress_file)}")


def calculate_free_cash_flow(df, stock_code):
    """计算自由现金流 (FCF = 经营现金流 - 资本支出)
    
    注意：此函数在映射后调用，使用映射后的统一列名
    
    自由现金流 (Free Cash Flow, FCF) 是企业在满足运营和资本支出需求后，
    可供投资者和债权人自由支配的现金流量。
    
    计算公式：
    FCF = 经营活动现金流净额 - 资本性支出
    
    Args:
        df: 映射后的DataFrame
        stock_code: 股票代码
        
    Returns:
        添加了自由现金流列的DataFrame
    """
    if df.empty:
        return df
    
    df = df.copy()
    
    # 映射后的统一列名
    operating_cash_flow_name = "经营活动现金流净额：经营活动产生的现金流量净额、经营活动所得现金净额、Net Cash from Operating Activities (CFO)、Net Cash from Operating Activities"
    capex_name = "资本性支出（CAPEX）：购买/购建固定资产、无形资产及其他长期资产支付的现金、资本开支、Capital Expenditures (CAPEX)、Purchase of Property, Plant and Equipment"
    fcf_name = "自由现金流：持续经营自由现金流、Free Cash Flow（FCF）"
    
    print(f"  - 正在计算 {stock_code} 的自由现金流...")
    
    if operating_cash_flow_name in df.columns and capex_name in df.columns:
        try:
            # FCF = 经营现金流 - 资本支出的绝对值
            df[fcf_name] = df[operating_cash_flow_name] - df[capex_name].abs()
            non_null_count = df[fcf_name].notna().sum()
            print(f"    ✓ 自由现金流已计算（{non_null_count} 条非空记录）")
        except Exception as e:
            print(f"    ⚠ 自由现金流计算失败: {e}")
            df[fcf_name] = None
    else:
        missing = []
        if operating_cash_flow_name not in df.columns:
            missing.append("经营活动现金流净额")
        if capex_name not in df.columns:
            missing.append("资本性支出")
        print(f"    ⚠ 无法计算自由现金流: 缺少字段 {', '.join(missing)}")
        df[fcf_name] = None
    
    return df


def load_stock_list(file_path, filter_by_column=None, filter_codes=None, sheet_name=0):
    """从CSV或Excel文件加载股票列表
    
    支持的文件格式：
    - CSV文件：.csv
    - Excel文件：.xlsx, .xls, .xlsm, .xlsb
    
    文件格式要求：
    股票代码,公司名称,备注[,处理]
    600519,贵州茅台,A股,Y
    002594,比亚迪,A股,N
    
    支持两种筛选方式：
    1. 通过文件中的"处理"列（或指定列名）筛选：
       - 如果filter_by_column='处理'，只会加载"处理"列为Y/True/1/是 的股票
       - 支持的值：Y, y, Yes, yes, True, true, 1, 是
    2. 通过股票代码列表筛选：
       - 如果filter_codes=['600519', '002594']，只加载指定的股票代码
    
    Args:
        file_path: 文件路径（CSV或Excel）
        filter_by_column: 可选，指定用于筛选的列名（如'处理'、'包含'、'include'等）
        filter_codes: 可选，要处理的股票代码列表（如['600519', '002594']）
        sheet_name: Excel文件的sheet名称或索引，默认为0（第一个sheet）。CSV文件会忽略此参数
        
    Returns:
        股票列表，格式为 [(代码, 名称), ...]；如果读取失败返回空列表
    """
    try:
        # 判断文件类型
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.csv':
            # 读取CSV文件
            df = pd.read_csv(file_path, encoding='utf-8')
            print(f"✓ 成功从CSV文件加载 {len(df)} 只股票")
        elif file_ext in ['.xlsx', '.xls', '.xlsm', '.xlsb']:
            # 读取Excel文件
            # 先读取第一行获取列名
            df_header = pd.read_excel(file_path, sheet_name=sheet_name, nrows=0)
            
            # 构建converters字典，将股票代码列强制读取为字符串（保留前导零）
            converters = {}
            if '股票代码' in df_header.columns:
                converters['股票代码'] = str
            
            # 读取完整数据，使用converters保留前导零
            df = pd.read_excel(file_path, sheet_name=sheet_name, converters=converters)
            
            # 确保股票代码列是字符串类型
            if '股票代码' in df.columns:
                df['股票代码'] = df['股票代码'].astype(str)
            
            print(f"✓ 成功从Excel文件加载 {len(df)} 只股票")
        else:
            print(f"✗ 不支持的文件格式: {file_ext}")
            print(f"  支持的格式：CSV (.csv), Excel (.xlsx, .xls, .xlsm, .xlsb)")
            return []
        
        original_count = len(df)
        
        # 方式1：根据CSV中的列进行筛选
        if filter_by_column and filter_by_column in df.columns:
            print(f"  正在根据列 '{filter_by_column}' 进行筛选...")
            # 支持的"是"值
            valid_yes_values = ['Y', 'y', 'Yes', 'yes', 'True', 'true', '1', '是', True, 1]
            # 筛选
            df_filtered = df[df[filter_by_column].isin(valid_yes_values)]
            print(f"  筛选后剩余 {len(df_filtered)} 只股票（过滤掉 {original_count - len(df_filtered)} 只）")
            df = df_filtered
        
        # 方式2：根据股票代码列表进行筛选
        if filter_codes:
            print(f"  正在根据股票代码列表进行筛选（共 {len(filter_codes)} 个代码）...")
            filter_codes_set = {str(code).strip() for code in filter_codes}
            before_filter_count = len(df)
            df_filtered = df[df['股票代码'].astype(str).str.strip().isin(filter_codes_set)]
            print(f"  筛选后剩余 {len(df_filtered)} 只股票（过滤掉 {before_filter_count - len(df_filtered)} 只）")
            
            # 检查是否有代码未找到
            found_codes = set(df_filtered['股票代码'].astype(str).str.strip())
            not_found = filter_codes_set - found_codes
            if not_found:
                print(f"  ⚠ 警告：以下股票代码在文件中未找到: {', '.join(sorted(not_found))}")
            
            df = df_filtered
        
        # 转换为元组列表格式
        stock_list = []
        for _, row in df.iterrows():
            code = str(row['股票代码']).strip()
            name = str(row['公司名称']).strip() if pd.notna(row['公司名称']) else ''
            stock_list.append((code, name))
        
        print(f"✓ 最终将处理 {len(stock_list)} 只股票\n")
        return stock_list
    
    except Exception as e:
        print(f"✗ 读取文件失败: {e}")
        print(f"  请检查文件格式是否正确，以及是否包含'股票代码'和'公司名称'列")
        return []


def get_single_stock_data(stock_code, company_name=''):
    """获取单个股票的财务数据（v3.2：支持多周期自动重试）
    
    v3.2 新增功能：
    - 使用 eastmoney_v0.4（支持周期参数）
    - indicator 和 statements 独立重试
    - 每个市场类型有不同的周期重试策略
    
    重试策略：
    - A股：
      * indicator: 按单季度 → 按报告期
      * statements: 无周期选择（不重试）
    - 港股：
      * indicator: 报告期 → 年度
      * statements: 报告期 → 年度
    - 美股：
      * indicator: 单季报 → 累计季报 → 年报
      * statements: 单季报 → 累计季报 → 年报
    
    Args:
        stock_code: 股票代码
        company_name: 公司名称（可选）
        
    Returns:
        包含数据、状态和统计信息的字典：
        {
            'data': DataFrame,      # 财务数据
            'status': dict,         # v3.2：数据源状态（包含使用的周期）
            'stats': dict           # 重复列合并统计
        }
    """
    market_type = detect_market_type(stock_code)
    market_name = {'A': 'A股', 'HK': '港股', 'US': '美股', 'UNKNOWN': '未知市场'}[market_type]
    
    if market_type == 'UNKNOWN':
        print(f"✗ 无法识别股票代码 {stock_code} 的市场类型")
        return {
            'data': pd.DataFrame(), 
            'status': {
                'indicator': {'success': False, 'error': '未知市场类型', 'period_used': ''},
                'statements': {'success': False, 'error': '未知市场类型', 'period_used': ''},
                'data_sources': []
            },
            'stats': {}
        }
    
    display_name = f"{company_name} ({stock_code})" if company_name else stock_code
    print(f"\n{'='*60}")
    print(f"正在获取 {market_name} {display_name} 的财务数据（v3.2多周期重试）...")
    print(f"{'='*60}")
    
    # 定义不同市场的周期优先级列表
    if market_type == 'A':
        indicator_periods = ["按单季度", "按报告期"]
        statements_periods = [None]  # A股statements无周期参数
    elif market_type == 'HK':
        indicator_periods = ["报告期", "年度"]
        statements_periods = ["报告期", "年度"]
    elif market_type == 'US':
        indicator_periods = ["单季报", "累计季报", "年报"]
        statements_periods = ["单季报", "累计季报", "年报"]
    else:
        indicator_periods = []
        statements_periods = []
    
    # 初始化状态记录（v3.2扩展：包含周期信息）
    status = {
        'indicator': {'success': False, 'error': None, 'period_used': ''},
        'statements': {'success': False, 'error': None, 'period_used': ''},
        'data_sources': []
    }
    
    stats = {}
    
    try:
        # 尝试不同周期获取完整数据
        for ind_period in indicator_periods:
            for stmt_period in statements_periods:
                df = pd.DataFrame()
                temp_status = None
                
                try:
                    # 构造参数
                    if market_type == 'A':
                        # A股：indicator有周期，statements无周期
                        print(f"\n尝试获取数据 - indicator周期: {ind_period}")
                        df, temp_status = get_full_financial_data_a(stock_code, indicator_type=ind_period)
                        
                    elif market_type == 'HK':
                        # 港股：indicator和statements使用相同周期
                        period_to_use = ind_period if ind_period == stmt_period else ind_period
                        print(f"\n尝试获取数据 - 周期: {period_to_use}")
                        df, temp_status = get_full_financial_data_hk(stock_code, indicator=period_to_use)
                        
                    elif market_type == 'US':
                        # 美股：indicator和statements使用相同周期
                        period_to_use = ind_period if ind_period == stmt_period else ind_period
                        print(f"\n尝试获取数据 - 周期: {period_to_use}")
                        df, temp_status = get_full_financial_data_us(stock_code, indicator=period_to_use)
                    
                    # 检查是否成功获取数据
                    if not df.empty:
                        # 成功获取数据，更新状态
                        status = temp_status.copy()
                        
                        # v3.2: 记录使用的周期
                        if status['indicator']['success']:
                            status['indicator']['period_used'] = ind_period if market_type in ['A', 'HK', 'US'] else ''
                        if status['statements']['success']:
                            status['statements']['period_used'] = stmt_period if market_type in ['HK', 'US'] else (ind_period if market_type == 'A' else '')
                        
                        # 合并重复列
                        df, stats = merge_duplicate_columns(df, market_name)
                        
                        print(f"\n✅ {display_name} 数据获取成功")
                        print(f"   使用周期 - 财务指标: {status['indicator'].get('period_used', 'N/A')}, "
                              f"财务报表: {status['statements'].get('period_used', 'N/A')}")
                        
                        return {'data': df, 'status': status, 'stats': stats}
                    
                    # 数据为空但API调用成功，尝试下一个周期
                    print(f"  ⚠️ 当前周期数据为空，尝试下一个周期...")
                    
                except Exception as e:
                    # 当前周期失败，记录错误并尝试下一个
                    error_msg = f"{type(e).__name__}: {str(e)}"
                    print(f"  ✗ 当前周期获取失败: {error_msg}")
                    print(f"  ℹ️ 尝试下一个周期...")
                    continue
                
                # A股statements无需重试多个周期，跳出内层循环
                if market_type == 'A':
                    break
        
        # 所有周期都失败
        print(f"\n❌ {display_name} 所有周期都获取失败")
        status['indicator']['error'] = "所有周期都获取失败"
        status['statements']['error'] = "所有周期都获取失败"
        return {'data': pd.DataFrame(), 'status': status, 'stats': {}}
        
    except Exception as e:
        print(f"❌ {display_name} 数据获取失败: {e}")
        import traceback
        traceback.print_exc()
        return {
            'data': pd.DataFrame(), 
            'status': {
                'indicator': {'success': False, 'error': str(e), 'period_used': ''},
                'statements': {'success': False, 'error': str(e), 'period_used': ''},
                'data_sources': []
            },
            'stats': {}
        }


def identify_duplicate_column_groups(columns):
    """
    识别所有重复列组
    
    参数:
        columns: list - 列名列表
    
    返回:
        dict - {基础列名: [原始列名, 列名_1, 列名_2, ...]}
    """
    duplicate_groups = defaultdict(list)
    
    for col in columns:
        col_str = str(col)
        
        # 检查是否是带编号的重复列（如 XX_1, XX_2）
        if '_' in col_str:
            parts = col_str.rsplit('_', 1)
            if len(parts) == 2 and parts[1].isdigit():
                base_name = parts[0]
                duplicate_groups[base_name].append(col)
            else:
                # 不是重复列，检查是否有对应的重复列
                if col_str not in duplicate_groups:
                    duplicate_groups[col_str].append(col)
        else:
            # 没有下划线的列，检查是否有对应的重复列
            duplicate_groups[col_str].append(col)
    
    # 只返回真正有重复的列组（包含原始列 + 至少一个重复列）
    result = {}
    for base_name, cols in duplicate_groups.items():
        # 检查是否存在带编号的重复列
        has_duplicates = any('_' in str(col) and str(col).rsplit('_', 1)[-1].isdigit() for col in cols)
        
        # 如果原始列存在，且有重复列
        if base_name in columns and has_duplicates:
            # 收集所有相关的列（原始列 + 所有编号列）
            group = [base_name]
            for col in columns:
                col_str = str(col)
                if col != base_name and '_' in col_str:
                    parts = col_str.rsplit('_', 1)
                    if len(parts) == 2 and parts[1].isdigit() and parts[0] == base_name:
                        group.append(col)
            
            if len(group) > 1:  # 确保有多个列
                result[base_name] = sorted(group, key=lambda x: (str(x) != base_name, str(x)))
    
    return result


def merge_duplicate_columns(df, market_type):
    """
    合并重复的列名
    
    参数:
        df: DataFrame - 要处理的数据框
        market_type: str - 市场类型 ('A股', '港股', '美股')
    
    返回:
        DataFrame - 处理后的数据框
        dict - 处理统计信息
    """
    if df.empty:
        return df, {}
    
    # 创建副本，避免修改原数据
    df_result = df.copy()
    
    # 识别所有重复列组
    duplicate_groups = identify_duplicate_column_groups(df_result.columns)
    
    if not duplicate_groups:
        return df_result, {'total_groups': 0, 'columns_removed': 0, 'details': []}
    
    print(f"\n🔍 发现 {len(duplicate_groups)} 组重复列，开始处理...")
    
    stats = {
        'total_groups': len(duplicate_groups),
        'columns_removed': 0,
        'details': []
    }
    
    columns_to_drop = []
    columns_to_rename = {}
    
    for base_name, group_cols in duplicate_groups.items():
        if len(group_cols) <= 1:
            continue
        
        detail = {
            'base_name': base_name,
            'original_columns': group_cols.copy(),
            'action': '',
            'kept_column': '',
            'removed_columns': []
        }
        
        if market_type == 'A股':
            # A股：保留原始列（第一个，即XX），删除其他列
            keep_col = group_cols[0]
            drop_cols = group_cols[1:]
            
            detail['action'] = 'A股策略：保留原始列'
            detail['kept_column'] = keep_col
            detail['removed_columns'] = drop_cols
            
            columns_to_drop.extend(drop_cols)
            stats['columns_removed'] += len(drop_cols)
            
            print(f"  📌 [{base_name}] A股策略")
            print(f"     ├─ 保留: {keep_col}")
            print(f"     └─ 删除: {', '.join(map(str, drop_cols))}")
            
        else:  # 港股或美股
            # 计算每列的非空值数量
            non_null_counts = {}
            for col in group_cols:
                non_null_counts[col] = df_result[col].notna().sum()
            
            # 找出非空值最多的列
            best_col = max(non_null_counts.items(), key=lambda x: x[1])[0]
            best_count = non_null_counts[best_col]
            
            # 要删除的列
            drop_cols = [col for col in group_cols if col != best_col]
            
            detail['action'] = f'{market_type}策略：保留非空数据最多的列'
            detail['kept_column'] = best_col
            detail['removed_columns'] = drop_cols
            detail['non_null_counts'] = non_null_counts
            
            columns_to_drop.extend(drop_cols)
            stats['columns_removed'] += len(drop_cols)
            
            # 如果保留的列有后缀，需要重命名为基础列名
            if best_col != base_name:
                columns_to_rename[best_col] = base_name
                detail['renamed_to'] = base_name
            
            print(f"  📌 [{base_name}] {market_type}策略")
            print(f"     ├─ 各列非空值数: {', '.join([f'{col}({non_null_counts[col]})' for col in group_cols])}")
            print(f"     ├─ 保留: {best_col} (非空值: {best_count})")
            if best_col != base_name:
                print(f"     ├─ 重命名: {best_col} → {base_name}")
            print(f"     └─ 删除: {', '.join(map(str, drop_cols))}")
        
        stats['details'].append(detail)
    
    # 执行删除操作
    if columns_to_drop:
        df_result = df_result.drop(columns=columns_to_drop)
    
    # 执行重命名操作
    if columns_to_rename:
        df_result = df_result.rename(columns=columns_to_rename)
    
    print(f"\n✅ 重复列合并完成:")
    print(f"   ├─ 处理了 {stats['total_groups']} 组重复列")
    print(f"   ├─ 删除了 {stats['columns_removed']} 个重复列")
    print(f"   └─ 当前列数: {len(df_result.columns)}")
    
    return df_result, stats


def main():
    """主函数 - 使用BatchProcessor进行批量处理"""
    
    print("\n" + "="*70)
    print("财务数据获取测试脚本 v3.21 - NOTICE_DATE混合获取版")
    print("支持分批保存、断点续传、索引生成、多周期自动重试、NOTICE_DATE混合获取")
    print("="*70)
    
    print("\n💡 功能特点:")
    print("  ✓ 批量分批处理：每N家公司生成一个Excel文件")
    print("  ✓ 断点续传：支持中断后继续运行")
    print("  ✓ 索引文件：记录所有公司的处理状态和存储位置")
    print("  ✓ 进度保存：自动保存处理进度")
    print("  ✓ 错误隔离：单个公司失败不影响其他公司")
    print("  ✓ 延迟控制：避免请求过于频繁")
    print("  ✓ 数据映射、清理、自由现金流计算等完整功能")
    print("  ✨ v3.21新增：NOTICE_DATE混合获取（API优先+规则补充）")
    print("  ✨ v3.21新增：使用eastmoney_v0.5，支持港股NOTICE_DATE API获取")
    print("  ✨ v3.21新增：支持所有市场（A股、港股、美股）的NOTICE_DATE处理")
    print("  ✓ v3.2继承：多周期自动重试，提高数据获取成功率")
    print("  ✓ v3.2继承：记录实际使用的数据周期到索引文件")
    print("="*70)
    
    # ==================== 配置参数 ====================
    
    # 映射文件路径
    mapping_file_path = "东方财富财务数据API映射最终版-陈俊同-20251030.xlsx"
    
    # 股票列表文件路径（支持CSV或Excel格式）
    stock_list_file = "股票代码汇总-陈俊同-20251110.xlsx"  # "stock_list.csv" 或 "股票代码汇总-陈俊同-20251110.xlsx"
    
    # Excel文件的sheet名称或索引（仅当使用Excel文件时有效）
    sheet_name = 0
    
    # 筛选选项
    filter_by_column = "处理"  # 设置为None表示不使用列筛选
    filter_codes = None  # 设置为None表示处理所有股票
    
    # ==================== 批量处理配置 ====================
    
    # 每批处理的公司数量（建议10-50家，1500家公司建议用20）
    batch_size = 20
    
    # 输出基础目录
    base_output_dir = os.getcwd()
    
    # 每家公司处理完后的延迟秒数（避免请求过于频繁）
    delay_seconds = 3
    
    # 断点续传：如果要从之前的进度继续，设置进度文件路径
    # 例如: resume_from_progress = "/path/to/download_batch_20251110_171536/progress_20251110_193045.json"
    resume_from_progress = None
    
    # ==================== 创建输出目录 ====================
    
    if resume_from_progress:
        output_dir = os.path.dirname(resume_from_progress)
        print(f"\n💡 断点续传模式：使用现有文件夹")
        print(f"   {output_dir}")
    else:
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(base_output_dir, f"download_batch_v3.21_{current_time}")
        print(f"\n💡 新建输出文件夹：{output_dir}")
    
    # ==================== 加载股票列表 ====================
    
    if not os.path.exists(stock_list_file):
        print(f"\n✗ 错误：未找到股票列表文件 {stock_list_file}")
        print(f"  请确保文件存在于当前目录")
        sys.exit(1)
    
    print(f"\n正在从 {stock_list_file} 加载股票列表...")
    if filter_by_column:
        print(f"  筛选模式：使用列 '{filter_by_column}' 进行筛选")
    elif filter_codes:
        print(f"  筛选模式：只处理指定的 {len(filter_codes)} 个股票代码")
    else:
        print(f"  筛选模式：处理所有股票")
    
    stock_codes = load_stock_list(
        stock_list_file, 
        filter_by_column=filter_by_column, 
        filter_codes=filter_codes, 
        sheet_name=sheet_name
    )
    
    if not stock_codes:
        print(f"\n✗ 错误：{stock_list_file} 文件为空或格式错误")
        print("  请检查文件格式是否正确，以及是否包含'股票代码'和'公司名称'列")
        sys.exit(1)
    
    # ==================== 批量处理 ====================
    
    print(f"\n{'='*60}")
    print(f"本次运行输出文件夹:")
    print(f"  {output_dir}")
    print(f"{'='*60}\n")
    
    # 创建批量处理器
    processor = BatchProcessor(
        mapping_file_path=mapping_file_path,
        batch_size=batch_size,
        output_dir=output_dir,
        delay_seconds=delay_seconds
    )
    
    # 开始处理
    timestamp = processor.process_stock_list(
        stock_codes, 
        resume_from_progress=resume_from_progress
    )
    
    # 显示完成信息
    print(f"\n{'='*70}")
    print(f"✅ 处理完成")
    print(f"{'='*70}")
    print(f"\n📁 输出目录: {os.path.abspath(output_dir)}")
    print(f"📋 索引文件: company_index_{timestamp}.xlsx")
    print(f"\n💡 索引文件记录了所有公司的处理状态和存储位置")
    print(f"💡 可以在Excel中打开索引文件，使用Ctrl+F搜索公司")
    print(f"💡 索引文件包含详细的数据源状态和实际使用的周期信息")
    print(f"💡 v3.21版本：所有数据包含NOTICE_DATE列（API优先+规则补充）")
    print(f"\n💡 如需断点续传，请在配置中设置 resume_from_progress 参数")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

