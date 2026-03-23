"""东方财富数据源模块 v0.5 - 港股NOTICE_DATE API获取版

v0.5 新增功能：
1. ✅ 港股NOTICE_DATE API获取：从东方财富API直接获取真实的披露日期
2. ✅ 港股日期数据完整性：支持获取历史真实的财报披露日期
3. ✅ 完全向后兼容：所有v0.4功能完全保留

v0.4 基础功能：
1. ✅ 美股财务指标API优化：使用自定义实现替代akshare，直接调用东方财富API
2. ✅ 支持不同企业类型：根据公司类型（一般企业/银行/保险）自动选择对应API接口
3. ✅ 完全向后兼容：所有函数签名保持不变，不影响现有代码
4. ✅ 保持所有v0.3功能：周期参数、容错处理等功能完全保留

v0.3 基础功能：
1. ✅ 港股周期参数：支持"年度"和"报告期"两种周期选择
2. ✅ 美股周期参数：支持"年报"、"单季报"、"累计季报"三种周期选择
3. ✅ 向后兼容：所有新增参数都有默认值，不影响现有代码
4. ✅ 统一接口：A股、港股、美股都可以通过参数灵活选择数据周期

v0.2 基础功能：
1. ✅ 完整的容错处理：单个表格获取失败不影响其他表格
2. ✅ 状态记录：详细记录每个表格的获取状态（成功/失败/原因）
3. ✅ 数据完整性：即使部分数据缺失，也返回已获取的数据
4. ✅ 适用所有市场：A股、港股、美股统一容错处理

v0.1 基础功能：
1. Pivot前重命名：保留同一报表内的所有重复数据，避免数据丢失
2. 全局重命名：统一所有报表合并后的列名，消除嵌套后缀
3. 统一处理：A股、港股、美股使用相同的命名规则

提供A股、港股、美股的财务数据和财务指标获取功能。

该模块通过akshare库从东方财富网获取股票财务数据，包括：
- 财务数据：资产负债表、利润表、现金流量表
- 财务指标：各种财务分析指标
- 港股披露日期：真实的历史财报披露日期（v0.5新增）
"""

import re
import functools
import akshare as ak
import pandas as pd
import requests
from datetime import datetime, timedelta
from us_financial_analysis_indicator import stock_financial_us_analysis_indicator_em


# ==================== v0.5 新增：港股NOTICE_DATE获取函数 ====================

def get_hk_financial_report_dates(stock_code):
    """
    从东方财富API获取港股真实的财报披露日期（v0.5新增）
    
    该函数直接调用东方财富F10数据接口，获取港股的重大事件数据，
    从中提取"报表披露"事件的NOTICE_DATE（公告日期），并反推对应的REPORT_DATE（财报期）。
    
    Args:
        stock_code (str): 港股代码，如 '00700'（不需要.HK后缀）
    
    Returns:
        pd.DataFrame or None: 
            - 成功：返回包含 [REPORT_DATE, NOTICE_DATE] 两列的DataFrame
            - 失败：返回 None
    
    示例：
        >>> dates_df = get_hk_financial_report_dates('00700')
        >>> print(dates_df)
           REPORT_DATE  NOTICE_DATE
        0   2024-12-31   2025-03-20
        1   2024-09-30   2024-11-13
        2   2024-06-30   2024-08-14
    
    注意：
        - 返回的REPORT_DATE是根据NOTICE_DATE的月份反推的（遵循香港财报规则）
        - 1-3月公告 → 上年12-31报表
        - 4-6月公告 → 当年03-31报表
        - 7-9月公告 → 当年06-30报表
        - 10-12月公告 → 当年09-30报表
    """
    try:
        # 构造API请求
        url = "https://datacenter.eastmoney.com/securities/api/data/get"
        params = {
            "type": "RPT_F10_HK_DETAIL",
            "params": f'{stock_code}.HK',
            "p": "1",
            "source": "F10",
            "client": "PC"
        }
        
        # 发送请求
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # 检查返回数据
        if 'data' not in data or not data['data']:
            print(f"  ⚠️  港股 {stock_code} API返回空数据")
            return None
        
        # 展平嵌套列表
        flat_data = [event for sublist in data['data'] for event in sublist]
        
        if not flat_data:
            print(f"  ⚠️  港股 {stock_code} 无事件数据")
            return None
        
        # 创建DataFrame
        df = pd.DataFrame(flat_data)
        
        # 过滤出"报表披露"事件
        if 'EVENT_TYPE' not in df.columns:
            print(f"  ⚠️  港股 {stock_code} 数据中无EVENT_TYPE字段")
            return None
        
        df = df[df['EVENT_TYPE'] == '报表披露']
        
        if df.empty:
            print(f"  ⚠️  港股 {stock_code} 无报表披露记录")
            return None
        
        # 检查必要的列
        if 'NOTICE_DATE' not in df.columns:
            print(f"  ⚠️  港股 {stock_code} 数据中无NOTICE_DATE字段")
            return None
        
        # 定义日期转换函数：根据公告日期反推财报期
        def calculate_financial_date(announcement_date_str):
            """根据公告月份推算对应的财报截止日期"""
            try:
                date = datetime.strptime(announcement_date_str, "%Y-%m-%d")
                year, month = date.year, date.month
                
                if month in [1, 2, 3]:
                    return f"{year - 1}-12-31"
                elif month in [4, 5, 6]:
                    return f"{year}-03-31"
                elif month in [7, 8, 9]:
                    return f"{year}-06-30"
                else:  # 10, 11, 12
                    return f"{year}-09-30"
            except:
                return None
        
        # 计算财报截止日期
        df['REPORT_DATE'] = df['NOTICE_DATE'].apply(calculate_financial_date)
        
        # 筛选有效数据
        result_df = df[['REPORT_DATE', 'NOTICE_DATE']].dropna()
        
        if result_df.empty:
            print(f"  ⚠️  港股 {stock_code} 无有效的日期映射")
            return None
        
        # 去重（保留第一条记录）
        result_df = result_df.drop_duplicates(subset=['REPORT_DATE'], keep='first')
        
        print(f"  ✓ 港股 {stock_code} API获取成功：{len(result_df)} 条披露日期")
        
        return result_df.reset_index(drop=True)
        
    except requests.Timeout:
        print(f"  ✗ 港股 {stock_code} API请求超时")
        return None
    except requests.RequestException as e:
        print(f"  ✗ 港股 {stock_code} API请求失败: {e}")
        return None
    except Exception as e:
        print(f"  ✗ 港股 {stock_code} NOTICE_DATE获取异常: {e}")
        return None


# ==================== 辅助函数：重命名和去重 ====================

def _rename_duplicates_before_pivot(df, date_col='REPORT_DATE', name_col='STD_ITEM_NAME'):
    """
    在pivot前对重复的(日期, 项目名)组合添加序号后缀，避免数据丢失
    
    处理逻辑：
    1. 对每个(日期, 项目)组合计算出现次数
    2. 第1次出现：保持原名
    3. 第2次出现：添加 _1
    4. 第3次出现：添加 _2
    ... 以此类推
    
    示例：
        输入：
            REPORT_DATE  | STD_ITEM_NAME | AMOUNT
            2024-12-31   | 总资产        | 1000000  ← 第1次
            2024-12-31   | 总资产        | 1000001  ← 第2次（重复）
            2024-12-31   | 总资产        | 1000002  ← 第3次（重复）
            2024-12-31   | 总负债        | 500000
        
        输出：
            REPORT_DATE  | STD_ITEM_NAME | AMOUNT
            2024-12-31   | 总资产        | 1000000
            2024-12-31   | 总资产_1      | 1000001
            2024-12-31   | 总资产_2      | 1000002
            2024-12-31   | 总负债        | 500000
    
    Args:
        df: 包含 date_col, name_col, AMOUNT 列的DataFrame
        date_col: 日期列名
        name_col: 项目名称列名
    
    Returns:
        重命名后的DataFrame
    """
    if df.empty:
        return df
    
    df_clean = df[[date_col, name_col, 'AMOUNT']].copy()
    
    # 计算每个(日期, 项目)组合的出现次数（从0开始）
    df_clean['_occurrence'] = df_clean.groupby([date_col, name_col]).cumcount()
    
    # 检测重复（occurrence > 0 表示第2次及以后出现）
    duplicates_mask = df_clean['_occurrence'] > 0
    duplicate_count = duplicates_mask.sum()
    
    if duplicate_count > 0:
        print(f"  ℹ️  检测到 {duplicate_count} 条重复数据，添加序号后缀（避免数据丢失）")
        
        # 为重复项添加后缀：项目名 + "_" + 序号
        df_clean.loc[duplicates_mask, name_col] = (
            df_clean.loc[duplicates_mask, name_col] + '_' + 
            df_clean.loc[duplicates_mask, '_occurrence'].astype(str)
        )
        
        # 显示重命名的例子（前5个不同的项目）
        renamed_items = df_clean.loc[duplicates_mask, name_col].unique()
        if len(renamed_items) > 0:
            sample_count = min(5, len(renamed_items))
            print(f"    示例: {', '.join(renamed_items[:sample_count])}")
            if len(renamed_items) > 5:
                print(f"    ... 还有 {len(renamed_items) - 5} 个")
    
    # 删除临时列
    df_clean = df_clean.drop(columns=['_occurrence'])
    
    return df_clean


def _get_base_column_name(col_name):
    """
    去除列名的所有后缀，获取基础名称
    
    处理的后缀类型：
    - _数字：如 _1, _2, _10
    - _x, _y：pandas merge 自动添加的后缀
    - 嵌套后缀：如 _1_x, _2_y_x
    
    示例：
        货币资金_1_x_y → 货币资金
        存货_2         → 存货
        货币资金       → 货币资金
        总资产_1_2_x   → 总资产
    
    Args:
        col_name: 列名
    
    Returns:
        基础列名（去除所有后缀）
    """
    # 使用正则表达式移除所有 _数字、_x、_y 后缀（可能有多个）
    # + 表示匹配1次或多次，$ 表示字符串结尾
    base = re.sub(r'(_\d+|_[xy])+$', '', col_name)
    return base


def _unify_column_names(df, key_col='REPORT_DATE'):
    """
    统一DataFrame中的列名，将所有后缀重新编号为连续序列
    
    处理流程：
    1. 提取每列的基础名称（去除所有后缀）
    2. 按基础名称分组
    3. 为每组内的列按出现顺序重新编号：基础名, 基础名_1, 基础名_2, ...
    
    示例：
        输入列名：
            货币资金, 货币资金_1, 货币资金_x, 货币资金_1_x, 货币资金_y, 总资产, 存货, 存货_y
        
        分组：
            货币资金组: [货币资金, 货币资金_1, 货币资金_x, 货币资金_1_x, 货币资金_y]
            总资产组: [总资产]
            存货组: [存货, 存货_y]
        
        输出列名：
            货币资金, 货币资金_1, 货币资金_2, 货币资金_3, 货币资金_4, 总资产, 存货, 存货_1
    
    Args:
        df: DataFrame
        key_col: 不需要重命名的键列（如日期列）
    
    Returns:
        重命名后的DataFrame
    """
    if df.empty:
        return df
    
    # 收集所有列名（除了键列）
    columns = [col for col in df.columns if col != key_col]
    
    # 按基础名称分组
    column_groups = {}
    for col in columns:
        base_name = _get_base_column_name(col)
        if base_name not in column_groups:
            column_groups[base_name] = []
        column_groups[base_name].append(col)
    
    # 生成重命名字典
    rename_dict = {}
    renamed_count = 0
    
    for base_name, col_list in column_groups.items():
        if len(col_list) == 1:
            # 只有一个列，检查是否需要去掉后缀
            old_name = col_list[0]
            if old_name != base_name:
                # 原名带后缀，重命名为基础名
                rename_dict[old_name] = base_name
                renamed_count += 1
        else:
            # 有多个列，统一重命名
            for idx, old_name in enumerate(col_list):
                if idx == 0:
                    new_name = base_name
                else:
                    new_name = f"{base_name}_{idx}"
                
                if old_name != new_name:
                    rename_dict[old_name] = new_name
                    renamed_count += 1
    
    if renamed_count > 0:
        print(f"  ✓ 统一重命名了 {renamed_count} 个列")
        # 显示部分重命名示例
        sample_renames = list(rename_dict.items())[:5]
        for old, new in sample_renames:
            print(f"    {old} → {new}")
        if len(rename_dict) > 5:
            print(f"    ... 还有 {len(rename_dict) - 5} 个")
    
    # 执行重命名
    df_renamed = df.rename(columns=rename_dict)
    
    return df_renamed


def _merge_tables_with_global_rename(tables, on='REPORT_DATE'):
    """
    合并多个表，并对所有列名进行全局统一重命名
    
    处理流程：
    1. 使用pandas merge逐个合并表（让pandas自动添加后缀）
    2. 合并完成后，调用全局重命名函数统一所有列名
    
    Args:
        tables: DataFrame列表
        on: 合并键（如 'REPORT_DATE'）
    
    Returns:
        合并并重命名后的DataFrame
    """
    if not tables:
        return pd.DataFrame()
    
    if len(tables) == 1:
        return tables[0]
    
    # Step 1: 逐个合并表（不指定suffixes，让pandas自动处理）
    result_df = tables[0]
    for i in range(1, len(tables)):
        result_df = pd.merge(
            result_df, 
            tables[i], 
            on=on, 
            how='outer'
            # 不指定suffixes，让pandas自动添加_x, _y
        )
    
    print(f"  📊 合并了 {len(tables)} 个表，共 {len(result_df.columns) - 1} 个数据列")
    
    # Step 2: 全局重命名，统一所有列名
    print(f"  🔄 正在进行全局列名统一...")
    result_df = _unify_column_names(result_df, key_col=on)
    
    return result_df


def _rename_duplicate_columns_in_single_df(df):
    """
    检查并重命名单个DataFrame中的重复列名
    
    处理逻辑：
    对于重复的列名，第1次出现保持原名，后续出现添加 _1, _2, ... 后缀
    
    示例：
        输入列名：[REPORT_DATE, ROE, ROA, ROE, 净利润, ROE, 营业收入]
        输出列名：[REPORT_DATE, ROE, ROA, ROE_1, 净利润, ROE_2, 营业收入]
    
    Args:
        df: DataFrame
    
    Returns:
        重命名后的DataFrame
    """
    if df.empty:
        return df
    
    # 检查是否有重复列名
    if not df.columns.duplicated().any():
        return df
    
    # 重命名重复列
    seen = {}
    new_columns = []
    renamed_count = 0
    
    for col in df.columns:
        if col not in seen:
            seen[col] = 0
            new_columns.append(col)  # 第1次出现，保持原名
        else:
            seen[col] += 1
            new_col = f"{col}_{seen[col]}"
            new_columns.append(new_col)  # 第2次及以后，添加后缀
            renamed_count += 1
    
    if renamed_count > 0:
        print(f"  ℹ️  检测到 {renamed_count} 个重复列名，已重命名")
        df_renamed = df.copy()
        df_renamed.columns = new_columns
        return df_renamed
    
    return df


# ==================== A股数据获取函数 ====================

def a_financial_indicator(stock_code, indicator="按单季度"):
    """获取A股财务指标
    
    Args:
        stock_code (str): 股票代码，如 '002594'
        indicator (str): 指标类型。可选值：
            - "按单季度": 仅获取按单季度的数据（默认）
            - "按报告期": 仅获取按报告期的数据
            - "全部": 同时获取按单季度和按报告期的数据
    
    Returns:
        pd.DataFrame 或 dict: 
            - 当indicator为"按单季度"或"按报告期"时，返回DataFrame
            - 当indicator为"全部"时，返回字典，包含两个键："按单季度"和"按报告期"
    """
    prefix = stock_code[:2]
    if prefix in ['60','68']:
        stock_code = stock_code + '.SH'
    elif prefix in ['00','30']:
        stock_code = stock_code + '.SZ'
    
    # 根据indicator参数返回相应数据
    if indicator == "按单季度":
        df = ak.stock_financial_analysis_indicator_em(symbol=stock_code, indicator="按单季度")
        return df
    elif indicator == "按报告期":
        df = ak.stock_financial_analysis_indicator_em(symbol=stock_code, indicator="按报告期")
        return df
    elif indicator == "全部":
        df_quarterly = ak.stock_financial_analysis_indicator_em(symbol=stock_code, indicator="按单季度")
        df_report = ak.stock_financial_analysis_indicator_em(symbol=stock_code, indicator="按报告期")
        return {
            "按单季度": df_quarterly,
            "按报告期": df_report
        }
    else:
        raise ValueError(f"不支持的indicator类型: {indicator}。可选值：'按单季度', '按报告期', '全部'")


def a_financial_statements(stock_code):
    """获取A股财务报表数据（改进版：使用全局重命名）
    
    改进说明：
    - 保留所有报表数据
    - 使用全局重命名统一列名
    
    Args:
        stock_code (str): A股股票代码，例如：'000001', '600519', '002594'
    
    Returns:
        pd.DataFrame: 合并后的财务报表数据
    """
    prefix = stock_code[:2]
    if prefix in ['60','68']:
        stock_code = 'SH' + stock_code
    elif prefix in ['00','30']:
        stock_code = 'SZ' + stock_code

    func_tuple = (
        "stock_balance_sheet_by_report_em",      # 资产负债表
        "stock_profit_sheet_by_quarterly_em",    # 利润表
        "stock_cash_flow_sheet_by_quarterly_em"  # 现金流量表
    )
    tables = []

    for i, func in enumerate(func_tuple):
        try:
            df = getattr(ak, func)(stock_code)
            if not df.empty:
                tables.append(df)
                print(f"  ✓ 获取{func}成功，共{len(df)}条记录")
        except Exception as e:
            print(f"  ✗ 获取 {func} 失败: {e}")
            continue

    if not tables:
        return pd.DataFrame()

    # 使用全局重命名方式合并
    print(f"  🔄 正在合并A股财务报表...")
    result_df = _merge_tables_with_global_rename(tables, on='REPORT_DATE')
    
    # 确保数据类型正确，避免Excel保存问题
    result_df = result_df.copy()
    
    # 转换日期列为字符串，避免Excel保存问题
    if 'REPORT_DATE' in result_df.columns:
        result_df['REPORT_DATE'] = result_df['REPORT_DATE'].astype(str)
    if 'NOTICE_DATE' in result_df.columns:
        result_df['NOTICE_DATE'] = result_df['NOTICE_DATE'].astype(str)
    
    return result_df.sort_values('REPORT_DATE', ascending=False)


def get_full_financial_data_a(stock_code, indicator_type="按单季度"):
    """
    获取A股完整财务数据（v0.5 版本）
    
    v0.5: 保持与 v0.4 相同的功能
    v0.4: 保持与 v0.3 相同的功能
    v0.3: 保持与 v0.2 相同的功能
    v0.2 改进：
    - ✅ 财务指标获取失败时，继续获取财务报表
    - ✅ 记录每个数据源的获取状态
    - ✅ 即使部分数据缺失，也返回已获取的数据
    
    包含：财务指标 + 财务报表，所有重复列名统一编号
    
    Args:
        stock_code (str): A股股票代码，如 '600519'
        indicator_type (str): 财务指标类型，可选：
            - "按单季度": 获取单季度财务指标（默认）
            - "按报告期": 获取报告期财务指标
    
    Returns:
        tuple: (DataFrame, dict)
            - DataFrame: 合并后的完整财务数据
            - dict: 获取状态信息
                {
                    'indicator': {'success': bool, 'error': str or None},
                    'statements': {'success': bool, 'error': str or None},
                    'data_sources': list  # 成功获取的数据源列表
                }
    
    示例：
        >>> df, status = get_full_financial_data_a('600519')
        >>> print(status)
        {'indicator': {'success': True, 'error': None},
         'statements': {'success': True, 'error': None},
         'data_sources': ['indicator', 'statements']}
    """
    print(f"\n{'='*60}")
    print(f"正在获取A股 {stock_code} 完整财务数据（{indicator_type}）")
    print(f"{'='*60}")
    
    # 初始化状态记录
    status = {
        'indicator': {'success': False, 'error': None},
        'statements': {'success': False, 'error': None},
        'data_sources': []
    }
    
    # 用于存储成功获取的数据
    available_tables = []
    
    # 1. 尝试获取财务指标
    print(f"[1/3] 获取财务指标（{indicator_type}）...")
    indicator_df = None
    try:
        indicator_df = a_financial_indicator(stock_code, indicator=indicator_type)
        
        if indicator_df is not None and not indicator_df.empty:
            print(f"  ✓ 获取成功：{len(indicator_df)} 行 × {len(indicator_df.columns)} 列")
            # 处理indicator表格内的重复列
            indicator_df = _rename_duplicate_columns_in_single_df(indicator_df)
            available_tables.append(indicator_df)
            status['indicator']['success'] = True
            status['data_sources'].append('indicator')
        else:
            print("  ⚠️  财务指标数据为空")
            status['indicator']['error'] = "数据为空"
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        print(f"  ✗ 财务指标获取失败: {error_msg}")
        status['indicator']['error'] = error_msg
    
    # 2. 尝试获取财务报表（无论指标是否成功）
    print(f"[2/3] 获取财务报表...")
    statements_df = None
    try:
        statements_df = a_financial_statements(stock_code)
        
        if statements_df is not None and not statements_df.empty:
            print(f"  ✓ 财务报表获取成功")
            available_tables.append(statements_df)
            status['statements']['success'] = True
            status['data_sources'].append('statements')
        else:
            print("  ⚠️  财务报表数据为空")
            status['statements']['error'] = "数据为空"
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        print(f"  ✗ 财务报表获取失败: {error_msg}")
        status['statements']['error'] = error_msg
    
    # 3. 合并可用的数据
    print(f"[3/3] 合并数据...")
    
    if not available_tables:
        print(f"\n❌ A股 {stock_code} 数据获取失败：所有数据源都无法获取")
        print(f"{'='*60}\n")
        return pd.DataFrame(), status
    
    if len(available_tables) == 1:
        result_df = available_tables[0]
        print(f"  ℹ️  仅获取到 {status['data_sources'][0]} 数据")
    else:
        # 合并多个数据源
        result_df = _merge_tables_with_global_rename(available_tables, on='REPORT_DATE')
    
    print(f"\n✅ A股 {stock_code} 数据获取完成")
    print(f"   最终数据：{len(result_df)} 行 × {len(result_df.columns)} 列")
    print(f"   数据来源：{', '.join(status['data_sources'])}")
    if status['indicator']['error'] or status['statements']['error']:
        print(f"   ⚠️  部分数据获取失败：")
        if status['indicator']['error']:
            print(f"      - 财务指标: {status['indicator']['error']}")
        if status['statements']['error']:
            print(f"      - 财务报表: {status['statements']['error']}")
    print(f"{'='*60}\n")
    
    return result_df.sort_values('REPORT_DATE', ascending=False), status


# ==================== 港股数据获取函数 ====================

def _get_hk_financial_indicator(stock_code, indicator="报告期"):
    """获取港股财务指标（v0.5 支持周期参数）
    
    Args:
        stock_code (str): 港股代码，如 '00700'
        indicator (str): 指标周期，可选值：
            - "报告期": 获取报告期数据（默认）
            - "年度": 获取年度数据
    
    Returns:
        pd.DataFrame: 财务指标数据
    """
    df = ak.stock_financial_hk_analysis_indicator_em(symbol=stock_code, indicator=indicator)
    return df


def _get_hk_financial_statements(stock_code, indicator="报告期"):
    """获取港股财务报表数据（v0.5 支持周期参数）
    
    改进说明：
    1. Pivot前重命名：保留同一报表内的所有重复数据
    2. 全局重命名：统一所有报表合并后的列名
    3. 周期参数：支持"年度"和"报告期"
    
    Args:
        stock_code (str): 港股代码，如 '00700'
        indicator (str): 报表周期，可选值：
            - "报告期": 获取报告期数据（默认）
            - "年度": 获取年度数据
    
    Returns:
        pd.DataFrame: 合并后的财务报表数据
    """
    reports = ["资产负债表", "利润表", "现金流量表"]
    tables = []
    
    for report in reports:
        try:
            df = ak.stock_financial_hk_report_em(stock=stock_code, symbol=report, indicator=indicator)
            
            if df.empty:
                print(f"  ⚠️  {report}无数据")
                continue
            
            # ⭐ 改进1：重命名重复项而不是删除（避免数据丢失）
            df_clean = _rename_duplicates_before_pivot(
                df, 
                date_col='REPORT_DATE', 
                name_col='STD_ITEM_NAME'
            )
            
            # 执行pivot操作（现在不会报错了）
            df_pivoted = (df_clean.pivot(index='REPORT_DATE', columns='STD_ITEM_NAME', values='AMOUNT')
                          .reset_index())
            
            tables.append(df_pivoted)
            print(f"  ✓ {report}处理完成，共{len(df_pivoted)}条记录，{len(df_pivoted.columns)-1}个字段")
            
        except Exception as e:
            print(f"  ✗ 处理{report}失败: {e}")
            continue
    
    if not tables:
        return pd.DataFrame()
    
    # ⭐ 改进2：使用全局重命名方式合并
    print(f"  🔄 正在合并港股财务报表...")
    result_df = _merge_tables_with_global_rename(tables, on='REPORT_DATE')
    
    return result_df.sort_values('REPORT_DATE', ascending=False)


def get_full_financial_data_hk(stock_code, indicator="报告期"):
    """
    获取港股完整财务数据（v0.5 支持周期参数）
    
    v0.5: 保持与 v0.4 相同的功能
    v0.4: 保持与 v0.3 相同的功能
    v0.3 新增：
    - ✅ 支持周期参数：可选择"报告期"或"年度"
    - ✅ 向后兼容：默认值为"报告期"，不影响现有代码
    
    v0.2 基础功能：
    - ✅ 财务指标获取失败时，继续获取财务报表
    - ✅ 记录每个数据源的获取状态
    - ✅ 即使部分数据缺失，也返回已获取的数据
    
    包含：财务指标 + 财务报表，所有重复列名统一编号
    
    Args:
        stock_code (str): 港股代码，如 '00700'
        indicator (str): 数据周期，可选值：
            - "报告期": 获取报告期数据（默认）
            - "年度": 获取年度数据
    
    Returns:
        tuple: (DataFrame, dict)
            - DataFrame: 合并后的完整财务数据
            - dict: 获取状态信息
    
    示例：
        >>> # 使用默认周期（报告期）
        >>> df, status = get_full_financial_data_hk('00700')
        
        >>> # 指定年度数据
        >>> df, status = get_full_financial_data_hk('00700', indicator='年度')
    """
    print(f"\n{'='*60}")
    print(f"正在获取港股 {stock_code} 完整财务数据（{indicator}）")
    print(f"{'='*60}")
    
    # 初始化状态记录
    status = {
        'indicator': {'success': False, 'error': None},
        'statements': {'success': False, 'error': None},
        'data_sources': []
    }
    
    # 用于存储成功获取的数据
    available_tables = []
    
    # 1. 尝试获取财务指标
    print(f"[1/3] 获取财务指标（{indicator}）...")
    indicator_df = None
    try:
        indicator_df = _get_hk_financial_indicator(stock_code, indicator=indicator)
        
        if indicator_df is not None and not indicator_df.empty:
            print(f"  ✓ 获取成功：{len(indicator_df)} 行 × {len(indicator_df.columns)} 列")
            # 处理indicator表格内的重复列
            indicator_df = _rename_duplicate_columns_in_single_df(indicator_df)
            available_tables.append(indicator_df)
            status['indicator']['success'] = True
            status['data_sources'].append('indicator')
        else:
            print("  ⚠️  财务指标数据为空")
            status['indicator']['error'] = "数据为空"
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        print(f"  ✗ 财务指标获取失败: {error_msg}")
        status['indicator']['error'] = error_msg
    
    # 2. 尝试获取财务报表（无论指标是否成功）
    print(f"[2/3] 获取财务报表（{indicator}）...")
    statements_df = None
    try:
        statements_df = _get_hk_financial_statements(stock_code, indicator=indicator)
        
        if statements_df is not None and not statements_df.empty:
            print(f"  ✓ 财务报表获取成功")
            available_tables.append(statements_df)
            status['statements']['success'] = True
            status['data_sources'].append('statements')
        else:
            print("  ⚠️  财务报表数据为空")
            status['statements']['error'] = "数据为空"
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        print(f"  ✗ 财务报表获取失败: {error_msg}")
        status['statements']['error'] = error_msg
    
    # 3. 合并可用的数据
    print(f"[3/3] 合并数据...")
    
    if not available_tables:
        print(f"\n❌ 港股 {stock_code} 数据获取失败：所有数据源都无法获取")
        print(f"{'='*60}\n")
        return pd.DataFrame(), status
    
    if len(available_tables) == 1:
        result_df = available_tables[0]
        print(f"  ℹ️  仅获取到 {status['data_sources'][0]} 数据")
    else:
        # 合并多个数据源
        result_df = _merge_tables_with_global_rename(available_tables, on='REPORT_DATE')
    
    print(f"\n✅ 港股 {stock_code} 数据获取完成")
    print(f"   最终数据：{len(result_df)} 行 × {len(result_df.columns)} 列")
    print(f"   数据来源：{', '.join(status['data_sources'])}")
    if status['indicator']['error'] or status['statements']['error']:
        print(f"   ⚠️  部分数据获取失败：")
        if status['indicator']['error']:
            print(f"      - 财务指标: {status['indicator']['error']}")
        if status['statements']['error']:
            print(f"      - 财务报表: {status['statements']['error']}")
    print(f"{'='*60}\n")
    
    return result_df.sort_values('REPORT_DATE', ascending=False), status


# ==================== 美股数据获取函数 ====================

def _get_us_financial_indicator(stock_code, indicator="单季报"):
    """获取美股财务指标（v0.5 使用优化版API）
    
    v0.4 改进：
    - 使用自定义实现替代akshare，直接调用东方财富API
    - 根据公司类型（一般企业/银行/保险）自动选择对应API接口
    - 支持更完整的数据字段
    
    Args:
        stock_code (str): 美股代码，如 'AAPL'
        indicator (str): 指标周期，可选值：
            - "单季报": 获取单季度数据（默认）
            - "年报": 获取年度数据
            - "累计季报": 获取累计季度数据
    
    Returns:
        pd.DataFrame: 财务指标数据
    """
    df = stock_financial_us_analysis_indicator_em(symbol=stock_code, indicator=indicator)
    return df


def _get_us_financial_statements(stock_code, indicator="单季报"):
    """获取美股财务数据（v0.5 支持周期参数）
    
    改进说明：
    1. Pivot前重命名：保留同一报表内的所有重复数据
    2. 全局重命名：统一所有报表合并后的列名
    3. 周期参数：支持"年报"、"单季报"、"累计季报"
    
    Args:
        stock_code (str): 美股代码，如 'AAPL'
        indicator (str): 报表周期，可选值：
            - "单季报": 获取单季度数据（默认）
            - "年报": 获取年度数据
            - "累计季报": 获取累计季度数据
    
    Returns:
        pd.DataFrame: 合并后的财务报表数据
    """
    configs = {
        "资产负债表": [indicator],
        "综合损益表": [indicator],
        "现金流量表": [indicator]
    }

    tables = []
    for symbol, indicators in configs.items():
        dfs = []
        for ind in indicators:
            try:
                df = ak.stock_financial_us_report_em(stock=stock_code, symbol=symbol, indicator=ind)
                
                if df.empty:
                    print(f"  ⚠️  {symbol}-{ind}无数据")
                    continue
                
                # ⭐ 改进1：重命名重复项而不是删除（避免数据丢失）
                df_clean = _rename_duplicates_before_pivot(
                    df, 
                    date_col='REPORT_DATE', 
                    name_col='ITEM_NAME'
                )
                
                # 执行pivot操作
                df_pivoted = (df_clean.pivot(index='REPORT_DATE', columns='ITEM_NAME', values='AMOUNT')
                              .reset_index())
                
                dfs.append(df_pivoted)
                print(f"  ✓ {symbol}-{ind}处理完成，共{len(df_pivoted)}条记录，{len(df_pivoted.columns)-1}个字段")
                
            except Exception as e:
                print(f"  ✗ 处理{symbol}-{ind}失败: {e}")
                continue
        
        if dfs:
            # 合并同一报表的不同indicator（如果有多个）
            combined = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]
            tables.append(combined)
    
    if not tables:
        return pd.DataFrame()
    
    # ⭐ 改进2：使用全局重命名方式合并
    print(f"  🔄 正在合并美股财务报表...")
    result_df = _merge_tables_with_global_rename(tables, on='REPORT_DATE')
    
    return result_df.sort_values('REPORT_DATE', ascending=False)


def get_full_financial_data_us(stock_code, indicator="单季报"):
    """
    获取美股完整财务数据（v0.5 美股指标API优化版）
    
    v0.5: 保持与 v0.4 相同的功能
    v0.4 新增：
    - ✅ 美股财务指标API优化：使用自定义实现，直接调用东方财富API
    - ✅ 支持不同企业类型：自动识别一般企业/银行/保险，选择对应API接口
    
    v0.3 基础功能：
    - ✅ 支持周期参数：可选择"年报"、"单季报"、"累计季报"
    - ✅ 向后兼容：默认值为"单季报"，不影响现有代码
    
    v0.2 基础功能：
    - ✅ 财务指标获取失败时，继续获取财务报表（解决ACGL等公司的问题）
    - ✅ 记录每个数据源的获取状态
    - ✅ 即使部分数据缺失，也返回已获取的数据
    
    包含：财务指标 + 财务报表，所有重复列名统一编号
    
    Args:
        stock_code (str): 美股代码，如 'AAPL', 'ACGL'
        indicator (str): 数据周期，可选值：
            - "单季报": 获取单季度数据（默认）
            - "年报": 获取年度数据
            - "累计季报": 获取累计季度数据
    
    Returns:
        tuple: (DataFrame, dict)
            - DataFrame: 合并后的完整财务数据
            - dict: 获取状态信息
                {
                    'indicator': {'success': bool, 'error': str or None},
                    'statements': {'success': bool, 'error': None},
                    'data_sources': list  # 成功获取的数据源列表
                }
    
    示例：
        >>> # 使用默认周期（单季报）
        >>> df, status = get_full_financial_data_us('AAPL')
        >>> print(status['data_sources'])
        ['indicator', 'statements']
        
        >>> # 指定年报数据
        >>> df, status = get_full_financial_data_us('AAPL', indicator='年报')
        
        >>> # 指定累计季报
        >>> df, status = get_full_financial_data_us('TSLA', indicator='累计季报')
    """
    print(f"\n{'='*60}")
    print(f"正在获取美股 {stock_code} 完整财务数据（{indicator}）")
    print(f"{'='*60}")
    
    # 初始化状态记录
    status = {
        'indicator': {'success': False, 'error': None},
        'statements': {'success': False, 'error': None},
        'data_sources': []
    }
    
    # 用于存储成功获取的数据
    available_tables = []
    
    # 1. 尝试获取财务指标（添加容错处理）
    print(f"[1/3] 获取财务指标（{indicator}）...")
    indicator_df = None
    try:
        indicator_df = _get_us_financial_indicator(stock_code, indicator=indicator)
        
        if indicator_df is not None and not indicator_df.empty:
            print(f"  ✓ 获取成功：{len(indicator_df)} 行 × {len(indicator_df.columns)} 列")
            # 处理indicator表格内的重复列
            indicator_df = _rename_duplicate_columns_in_single_df(indicator_df)
            available_tables.append(indicator_df)
            status['indicator']['success'] = True
            status['data_sources'].append('indicator')
        else:
            print("  ⚠️  财务指标数据为空")
            status['indicator']['error'] = "数据为空"
    except TypeError as e:
        # 专门处理 'NoneType' object is not subscriptable 错误
        if "'NoneType' object is not subscriptable" in str(e):
            error_msg = "东方财富数据库中无此公司的财务指标"
            print(f"  ⚠️  财务指标不可用：{error_msg}")
            print(f"  ℹ️  将继续获取财务报表数据...")
            status['indicator']['error'] = error_msg
        else:
            error_msg = f"TypeError: {str(e)}"
            print(f"  ✗ 财务指标获取失败: {error_msg}")
            status['indicator']['error'] = error_msg
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        print(f"  ✗ 财务指标获取失败: {error_msg}")
        status['indicator']['error'] = error_msg
    
    # 2. 尝试获取财务报表（无论指标是否成功，都继续执行）
    print(f"[2/3] 获取财务报表（{indicator}）...")
    statements_df = None
    try:
        statements_df = _get_us_financial_statements(stock_code, indicator=indicator)
        
        if statements_df is not None and not statements_df.empty:
            print(f"  ✓ 财务报表获取成功")
            available_tables.append(statements_df)
            status['statements']['success'] = True
            status['data_sources'].append('statements')
        else:
            print("  ⚠️  财务报表数据为空")
            status['statements']['error'] = "数据为空"
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        print(f"  ✗ 财务报表获取失败: {error_msg}")
        status['statements']['error'] = error_msg
    
    # 3. 合并可用的数据
    print(f"[3/3] 合并数据...")
    
    if not available_tables:
        print(f"\n❌ 美股 {stock_code} 数据获取失败：所有数据源都无法获取")
        print(f"{'='*60}\n")
        return pd.DataFrame(), status
    
    if len(available_tables) == 1:
        result_df = available_tables[0]
        print(f"  ℹ️  仅获取到 {status['data_sources'][0]} 数据")
    else:
        # 合并多个数据源
        result_df = _merge_tables_with_global_rename(available_tables, on='REPORT_DATE')
    
    print(f"\n✅ 美股 {stock_code} 数据获取完成")
    print(f"   最终数据：{len(result_df)} 行 × {len(result_df.columns)} 列")
    print(f"   数据来源：{', '.join(status['data_sources'])}")
    if status['indicator']['error'] or status['statements']['error']:
        print(f"   ⚠️  部分数据获取失败：")
        if status['indicator']['error']:
            print(f"      - 财务指标: {status['indicator']['error']}")
        if status['statements']['error']:
            print(f"      - 财务报表: {status['statements']['error']}")
    print(f"{'='*60}\n")
    
    return result_df.sort_values('REPORT_DATE', ascending=False), status


# ==================== 历史数据获取函数（保持不变）====================

def get_historical_data(stock_code, market='A', period='daily', days=365):
    """获取股票历史数据
    
    从东方财富网获取指定股票的历史价格数据。
    
    Args:
        stock_code (str): 股票代码
        market (str): 市场类型。可选值：
            - 'A': A股市场
            - 'HK': 港股市场
            - 'US': 美股市场
        period (str): 数据周期。可选值：
            - 'daily': 日K线
            - 'weekly': 周K线
        days (int): 获取天数，默认365天
    
    Returns:
        pd.DataFrame: 历史数据，包含日期、开盘价、收盘价、最高价、最低价、成交量等
    """
    current_date = datetime.now()
    start_date = current_date - timedelta(days=days)
    start_date_str = start_date.strftime("%Y%m%d")
    end_date_str = current_date.strftime("%Y%m%d")
    
    if market == 'A':
        return _get_a_historical_data(stock_code, period, start_date_str, end_date_str)
    elif market == 'HK':
        return _get_hk_historical_data(stock_code, period, start_date_str, end_date_str)
    elif market == 'US':
        return _get_us_historical_data(stock_code, period, start_date_str, end_date_str)
    else:
        raise ValueError(f"不支持的市场类型: {market}")


def _get_a_historical_data(symbol, period, start_date, end_date):
    """获取A股历史数据"""
    return ak.stock_zh_a_hist(symbol=symbol, period=period, start_date=start_date, end_date=end_date, adjust="")


def _get_hk_historical_data(symbol, period, start_date, end_date):
    """获取港股历史数据"""
    return ak.stock_hk_hist(symbol=symbol, period=period, start_date=start_date, end_date=end_date, adjust="")


def _get_us_historical_data(symbol, period, start_date, end_date):
    """获取美股历史数据"""
    return ak.stock_us_hist(symbol=symbol, period=period, start_date=start_date, end_date=end_date, adjust="")


# ==================== 其他辅助函数（保持不变）====================

def a_dividend_distribution_detail(stock_code):
    """获取A股分红配送详情
    
    Args:
        stock_code (str): A股股票代码，例如：'000001', '600519', '002594'
    
    Returns:
        pd.DataFrame: 分红配送详情数据框
    """
    try:
        df = ak.stock_fhps_detail_em(symbol=stock_code)
        
        if not isinstance(df, pd.DataFrame):
            print(f"获取A股 {stock_code} 分红配送详情返回数据格式异常")
            return pd.DataFrame()
        
        # 转换日期列为字符串格式
        date_columns = ['股权登记日', '除权除息日', '预案公告日', '股东大会预案公告日']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce').dt.strftime('%Y-%m-%d')
        
        return df
        
    except Exception as e:
        print(f"获取A股 {stock_code} 分红配送详情失败: {e}")
        return pd.DataFrame()


def a_shareholder_number_detail(stock_code):
    """获取A股股东户数详情
    
    Args:
        stock_code (str): A股股票代码，例如：'000001', '600519', '002594'
    
    Returns:
        pd.DataFrame: 股东户数详情数据框
    """
    try:
        df = ak.stock_zh_a_gdhs_detail_em(symbol=stock_code)
        
        if not isinstance(df, pd.DataFrame):
            print(f"获取A股 {stock_code} 股东户数详情返回数据格式异常")
            return pd.DataFrame()
        
        # 转换日期列为字符串格式
        if '股东户数统计截止日' in df.columns:
            df['股东户数统计截止日'] = pd.to_datetime(df['股东户数统计截止日'], errors='coerce').dt.strftime('%Y-%m-%d')
        
        return df
        
    except Exception as e:
        print(f"获取A股 {stock_code} 股东户数详情失败: {e}")
        return pd.DataFrame()


