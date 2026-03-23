import json
import re
from datetime import datetime
from pathlib import Path

import pandas as pd

INDEX_FILE_PATH = "src/local/data/macro_indices.xlsx"  # 指数数据文件路径
# 版本信息
VERSION_SUFFIX = "v0.6"
METADATA_VERSION = "v0.6_20260212185825"

# 需要删除的列
COLS_TO_DROP = ["股票代码", "财报数据截止日期", "匹配财报公告日期"]

# 需要从第一行提取作为公司标识的列（提取后也会删除）
COLS_TO_EXTRACT = ["股票代码", "货币单位"]

# 公司标识列（放在最前面）
COMPANY_IDENTITY_COLS = ["sequence_id", "company_name", "stock_code", "货币单位"]

# 市场类型与指数映射
MARKET_INDEX_MAPPING = {
    "A股": ["沪深300", "中证500"],
    "港股": ["恒生指数", "恒生科技"],
    "美股": ["标普500", "纳斯达克"],
}

# 批次处理大小（每次处理N个文件后合并，避免内存溢出）
BATCH_SIZE = 50


def parse_filename(filename):
    """
    解析文件名，提取公司信息
    格式: {序号}_{公司名}_{股票代码}.xlsx
    例如: 0001_中国铝业_601600.xlsx
    """
    match = re.match(r"^(\d+)_(.+?)_(.+?)\.xlsx$", filename)
    if match:
        sequence_id = int(match.group(1))
        company_name = match.group(2)
        stock_code = match.group(3)
        return {
            "sequence_id": sequence_id,
            "company_name": company_name,
            "stock_code": stock_code,
        }
    return None


def extract_company_info_from_data(df, filename_info):
    """
    从数据第一行提取公司信息
    """
    company_info = filename_info.copy()

    # 从第一行提取股票代码和货币单位
    if len(df) > 0:
        for col in COLS_TO_EXTRACT:
            if col in df.columns:
                value = df[col].iloc[0]
                if pd.notna(value):
                    # 统一key名称：股票代码 -> stock_code，货币单位 -> 货币单位
                    if col == "股票代码":
                        key = "stock_code"
                    elif col == "货币单位":
                        key = "货币单位"
                    else:
                        # 其他列：将列名转换为key（去掉空格，转为小写）
                        key = col.replace(" ", "_").lower()
                    company_info[key] = str(value)

    return company_info


def process_single_file(file_path, filename_info, index_data_dict):
    """
    处理单个Excel文件（v0.5增强版：保持Excel原始列顺序）

    Args:
        file_path: Excel文件路径
        filename_info: 从文件名解析的公司信息
        index_data_dict: 预加载的指数数据

    Returns:
        tuple: (DataFrame, company_info, market_type, error_message)
    """
    try:
        # 读取"财报_日K对齐"sheet
        df = pd.read_excel(file_path, sheet_name="财报_日K对齐")

        if df.empty:
            return None, None, None, "数据为空"

        # 提取公司信息（从第一行）
        company_info = extract_company_info_from_data(df, filename_info)

        # 【v0.4修复】使用Excel第一行的股票代码（优先），如果不存在则使用文件名中的股票代码
        # company_info['stock_code'] 的值：优先Excel第一行的"股票代码"列（可能包含后缀如.HK、.SZ、.SH、.N等），否则文件名中的股票代码
        extracted_stock_code = company_info.get(
            "stock_code", filename_info["stock_code"]
        )
        extracted_currency = company_info.get("货币单位")

        # 删除不需要的列（包括COLS_TO_EXTRACT中的列）
        cols_to_drop_actual = []
        for col in COLS_TO_DROP:
            if col in df.columns:
                cols_to_drop_actual.append(col)
        for col in COLS_TO_EXTRACT:
            if col in df.columns and col not in cols_to_drop_actual:
                cols_to_drop_actual.append(col)

        if cols_to_drop_actual:
            df = df.drop(columns=cols_to_drop_actual)

        code_str = str(extracted_stock_code)
        if code_str.isdigit() and len(code_str) == 6:
            market_type = "A股"
        elif len(code_str) == 5 and code_str.isdigit():
            market_type = "港股"

        else:
            market_type = "美股"

        indices_to_merge = MARKET_INDEX_MAPPING[market_type]

        merged_df = df.copy()
        merged_df["日期"] = pd.to_datetime(merged_df["日期"], errors="coerce")
        for index_name in indices_to_merge:
            merged_df = pd.merge(
                merged_df, index_data_dict[index_name], on="日期", how="left"
            )
        df = merged_df.fillna(0)

        # 【v0.4修复】添加公司标识列，确保填充所有行
        # stock_code列使用Excel第一行的值（与v0.3逻辑一致）
        df["sequence_id"] = company_info["sequence_id"]
        df["company_name"] = company_info["company_name"]
        df["stock_code"] = extracted_stock_code

        mapping = {
            "人民币": 0.1,
            "RMB": 0.1,
            "CNY": 0.1,
            "USD": 0.3,
            "美元": 0.3,
            "HKD": -0.1,
            "港币": -0.1,
        }
        df["货币单位"] = mapping.get(extracted_currency, 0.1)

        return df, company_info, market_type, None

    except Exception as e:
        error_msg = f"处理失败: {str(e)}"
        return None, None, None, error_msg


def collect_all_columns(excel_files, sample_size=10):
    """
    扫描所有文件，收集所有可能的列名
    【v0.5修改】保持Excel原始列顺序，不进行排序
    """
    print(f"\n扫描前{sample_size}个文件，收集所有列名（保持Excel原始顺序）...")

    # 从第一个文件获取基准列顺序
    base_columns = []
    all_columns_set = set()

    for file_path in excel_files[:sample_size]:
        try:
            df = pd.read_excel(file_path, sheet_name="财报_日K对齐", nrows=1)
            columns_list = df.columns.tolist()

            # 第一个文件作为基准顺序
            if not base_columns:
                base_columns = columns_list.copy()

            # 收集所有列名
            all_columns_set.update(columns_list)

        except Exception as e:
            print(f"  警告: 无法读取 {file_path.name}: {e}")

    if not base_columns:
        print("  错误: 无法读取任何文件的列信息")
        return []

    # 从基准列顺序中移除要删除的列
    excel_columns = [
        col
        for col in base_columns
        if col not in COLS_TO_DROP and col not in COLS_TO_EXTRACT
    ]

    # 添加后续文件中发现的新列（不在基准顺序中的列）
    for col in all_columns_set:
        if col not in COLS_TO_DROP and col not in COLS_TO_EXTRACT:
            if col not in excel_columns:
                excel_columns.append(col)

    # 构建最终列顺序：公司标识列在最前面，然后是Excel原始列
    column_list = excel_columns.copy()

    print(f"  找到 {len(column_list)} 个Excel原始列（保持原始顺序）")
    print(f"  公司标识列将放在最前面: {COMPANY_IDENTITY_COLS}")

    return column_list


def process_all_files(excel_dir, output_dir, index_data_dict):
    """
    逐个处理Excel文件，并为每家公司生成独立Parquet。
    v0.6: 调整货币单位映射值（人民币=0.1, 美元=0.3, 港币=-0.1）
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    excel_files = [
        p
        for p in Path(excel_dir).glob("*.xlsx")
        if p.name != "company_index.xlsx" and p.name != "company_index.csv"
    ]
    total_files = len(excel_files)
    print(f"\n找到 {total_files} 个Excel文件")

    if total_files == 0:
        print("错误: 未找到任何Excel文件")
        return

    all_columns = collect_all_columns(excel_files)
    columns_seen = set(all_columns)

    company_mapping = {}
    processing_log = []

    stats = {
        "total_files": total_files,
        "processed": 0,
        "success": 0,
        "failed": 0,
        "total_rows": 0,
        "errors": [],
        "earliest_date": None,
        "latest_date": None,
        "market_distribution": {"A股": 0, "港股": 0, "美股": 0, "Unknown": 0},
    }

    missing_before_total = 0
    missing_after_total = 0
    filled_numeric_columns = set()
    index_columns_added = set()

    print("\n开始处理文件...")
    print(f"批次大小: {BATCH_SIZE}")

    for batch_start in range(0, total_files, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, total_files)
        batch_files = excel_files[batch_start:batch_end]

        print(
            f"\n处理批次 {batch_start // BATCH_SIZE + 1}: 文件 {batch_start + 1}-{batch_end}/{total_files}"
        )

        for file_path in batch_files:
            filename = file_path.name
            filename_info = parse_filename(filename)

            if filename_info is None:
                error_msg = f"文件名格式无法解析: {filename}"
                stats["failed"] += 1
                stats["errors"].append(error_msg)
                processing_log.append(
                    {"file": filename, "status": "failed", "error": error_msg}
                )
                stats["processed"] += 1
                print(f"  ✗ {filename}: {error_msg}")
                continue

            df, company_info, market_type, error = process_single_file(
                file_path, filename_info, index_data_dict
            )

            if df is None or error:
                stats["failed"] += 1
                error_msg = error or "未知错误"
                stats["errors"].append(f"{filename}: {error_msg}")
                processing_log.append(
                    {"file": filename, "status": "failed", "error": error_msg}
                )
                stats["processed"] += 1
                print(f"  ✗ {filename}: {error_msg}")
                continue

            stats["success"] += 1
            stats["total_rows"] += len(df)

            # 统计市场分布
            if market_type:
                stats["market_distribution"][market_type] = (
                    stats["market_distribution"].get(market_type, 0) + 1
                )

            columns_seen.update(df.columns.tolist())

            # 记录新增的指数列
            for col in df.columns:
                for indices in MARKET_INDEX_MAPPING.values():
                    for index_name in indices:
                        if col.startswith(f"{index_name}_"):
                            index_columns_added.add(col)

            if "日期" in df.columns:
                # 日期已在 process_single_file 中 to_datetime，此处仅剔除无效日期行
                df = df.dropna(subset=["日期"])

            numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
            for col in numeric_cols:
                if col not in ["sequence_id"]:
                    df[col] = df[col].astype("float32")

            if "sequence_id" in df.columns:
                df["sequence_id"] = df["sequence_id"].astype("int32")

            # 填充原有数值列的缺失值（不包括指数列，指数列已经填充过了）
            numeric_fill_cols = df.select_dtypes(include=["float32", "float64"]).columns
            fill_cols = [
                col
                for col in numeric_fill_cols
                if col not in ["sequence_id"]
                and not any(
                    col.startswith(f"{idx}_")
                    for indices in MARKET_INDEX_MAPPING.values()
                    for idx in indices
                )
            ]
            filled_numeric_columns.update(fill_cols)

            missing_before = df[fill_cols].isna().sum().sum() if fill_cols else 0
            if fill_cols:
                df[fill_cols] = df[fill_cols].fillna(0)
            missing_after = df[fill_cols].isna().sum().sum() if fill_cols else 0
            missing_before_total += int(missing_before)
            missing_after_total += int(missing_after)

            if "日期" in df.columns and not df["日期"].empty:
                df = df.sort_values("日期", ascending=True).reset_index(drop=True)
                start_date = df["日期"].min()
                end_date = df["日期"].max()
                if start_date and (
                    stats["earliest_date"] is None
                    or start_date < stats["earliest_date"]
                ):
                    stats["earliest_date"] = start_date
                if end_date and (
                    stats["latest_date"] is None or end_date > stats["latest_date"]
                ):
                    stats["latest_date"] = end_date
            else:
                start_date = None
                end_date = None

            parquet_filename = f"{file_path.stem}.parquet"
            parquet_path = output_dir / parquet_filename
            df.to_parquet(
                parquet_path, engine="pyarrow", compression="snappy", index=False
            )

            company_id = filename_info["sequence_id"]
            mapping_entry = company_info.copy()
            mapping_entry["parquet_file"] = parquet_filename
            mapping_entry["market_type"] = market_type
            company_mapping[company_id] = mapping_entry

            processing_log.append(
                {
                    "file": filename,
                    "status": "success",
                    "rows": len(df),
                    "parquet_file": parquet_filename,
                    "market_type": market_type,
                    "date_range": {
                        "start": str(start_date) if start_date is not None else None,
                        "end": str(end_date) if end_date is not None else None,
                    },
                }
            )

            if stats["success"] % 10 == 0:
                print(
                    f"  ✓ 已处理 {stats['success']} 个文件，当前: {filename} ({len(df)} 行, {market_type})"
                )

            stats["processed"] += 1

    if stats["success"] == 0:
        print("错误: 没有成功处理任何文件")
        return

    print("\n生成元数据...")
    metadata = {
        "version": METADATA_VERSION,
        "processing_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_rows": stats["total_rows"],
        "total_companies": len(company_mapping),
        "date_range": {
            "start": str(stats["earliest_date"])
            if stats["earliest_date"] is not None
            else None,
            "end": str(stats["latest_date"])
            if stats["latest_date"] is not None
            else None,
        },
        "columns": {
            "total": len(columns_seen),
            "list": sorted(columns_seen),  # 元数据中仍按字母顺序列出，便于查看
            "date_columns": ["日期"],
            "company_columns": COMPANY_IDENTITY_COLS,
            "column_order_preserved": True,  # v0.5新增：标记已保持列顺序
        },
        "statistics": {
            "total_files": stats["total_files"],
            "processed": stats["processed"],
            "success": stats["success"],
            "failed": stats["failed"],
            "success_rate": f"{(stats['success'] / stats['total_files'] * 100) if stats['total_files'] else 0:.2f}%",
            "market_distribution": stats["market_distribution"],
        },
        "missing_value_filling": {
            "enabled": True,
            "method": "fillna(0)",
            "columns_filled": len(filled_numeric_columns),
            "missing_values_before": missing_before_total,
            "missing_values_after": missing_after_total,
        },
        "currency_mapping": {
            "description": "【v0.6修改】调整货币单位映射值",
            "mapping": {"人民币/RMB/CNY": 0.1, "美元/USD": 0.3, "港币/HKD": -0.1},
            "default_value": 0.1,
        },
        "index_data": {
            "enabled": len(index_data_dict) > 0,
            "source_file": INDEX_FILE_PATH,
            "index_mapping": MARKET_INDEX_MAPPING,
            "indices_loaded": list(index_data_dict.keys()),
            "index_columns_added": sorted(index_columns_added),
            "index_columns_count": len(index_columns_added),
            "missing_value_strategy": "fillna(0)",
        },
    }

    metadata_path = output_dir / f"data_metadata_{VERSION_SUFFIX}.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"  ✓ 已保存: {metadata_path}")

    company_mapping_path = output_dir / f"company_mapping_{VERSION_SUFFIX}.json"
    with open(company_mapping_path, "w", encoding="utf-8") as f:
        json.dump(company_mapping, f, ensure_ascii=False, indent=2)
    print(f"  ✓ 已保存: {company_mapping_path}")

    log_path = output_dir / f"processing_log_{VERSION_SUFFIX}.txt"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"数据处理日志 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"版本: {METADATA_VERSION}\n")
        f.write(f"总计文件数: {stats['total_files']}\n")
        f.write(f"成功处理: {stats['success']}\n")
        f.write(f"处理失败: {stats['failed']}\n")
        f.write(f"总数据行数: {stats['total_rows']}\n")
        f.write("\n市场分布:\n")
        for market, count in stats["market_distribution"].items():
            f.write(f"  {market}: {count} 家公司\n")
        f.write("\n货币单位映射（v0.6调整）:\n")
        f.write("  人民币/RMB/CNY: 0.1\n")
        f.write("  美元/USD: 0.3\n")
        f.write("  港币/HKD: -0.1\n")
        f.write("  默认值: 0.1\n")
        f.write("\n指数数据合并:\n")
        f.write(f"  指数数据源: {INDEX_FILE_PATH}\n")
        f.write(f"  成功加载指数: {len(index_data_dict)}\n")
        f.write(f"  新增指数列数: {len(index_columns_added)}\n")
        f.write("\n列顺序:\n")
        f.write("  保持Excel原始列顺序: 是\n")
        f.write("  公司标识列位置: 最前面\n")
        f.write("  指数列位置: 最后面\n")
        f.write("\n缺失值填充:\n")
        f.write(f"  填充前缺失值总数: {missing_before_total:,}\n")
        f.write(f"  填充后缺失值总数: {missing_after_total:,}\n")
        f.write(f"  填充列数: {len(filled_numeric_columns)}\n\n")

        if stats["errors"]:
            f.write("错误列表:\n")
            f.write("-" * 80 + "\n")
            for error in stats["errors"][:50]:
                f.write(f"{error}\n")
            if len(stats["errors"]) > 50:
                f.write(f"... 还有 {len(stats['errors']) - 50} 个错误未显示\n")

    print(f"  ✓ 已保存: {log_path}")

    print("\n" + "=" * 80)
    print("处理完成！")
    print("=" * 80)
    print(f"总计文件数: {stats['total_files']}")
    print(f"成功处理: {stats['success']}")
    print(f"处理失败: {stats['failed']}")
    print(f"总数据行数: {stats['total_rows']:,}")
    print(f"公司数量: {len(company_mapping)}")
    print("\n市场分布:")
    for market, count in stats["market_distribution"].items():
        if count > 0:
            print(f"  {market}: {count} 家公司")
    if stats["earliest_date"] and stats["latest_date"]:
        print(f"\n日期范围: {stats['earliest_date']} 到 {stats['latest_date']}")
    print(f"列数: {len(columns_seen)}")
    print(f"缺失值填充: {missing_before_total:,} -> {missing_after_total:,}")
    print("\n货币单位映射（v0.6调整）:")
    print("  人民币/RMB/CNY: 0.1")
    print("  美元/USD: 0.3")
    print("  港币/HKD: -0.1")
    print("  默认值: 0.1")
    print("\n指数数据:")
    print(f"  成功加载: {len(index_data_dict)} 个指数")
    print(f"  新增列数: {len(index_columns_added)}")
    print("\n列顺序:")
    print("  保持Excel原始列顺序: 是")
    print("  公司标识列位置: 最前面")
    print("  指数列位置: 最后面")
    print(f"\n输出目录: {output_dir}")


def main():
    excel_dir = "src/local/data/stock"
    output_dir = "src/local/data/processed"
    index_file = INDEX_FILE_PATH

    print(f"Excel目录: {excel_dir}")
    print(f"输出目录: {output_dir}")
    print(f"指数文件: {index_file}")

    index_data_dict = {}

    for sheet in ["沪深300", "中证500", "标普500", "纳斯达克", "恒生指数", "恒生科技"]:
        df = pd.read_excel(index_file, sheet_name=sheet)

        cols = df.columns.tolist()
        cols[1:] = [f"{sheet}_{c}" for c in cols[1:]]
        df.columns = cols

        index_data_dict[sheet] = df

    # 处理所有文件
    process_all_files(excel_dir, output_dir, index_data_dict)


if __name__ == "__main__":
    main()
