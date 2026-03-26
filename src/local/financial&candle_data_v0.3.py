#!/Users/juntongchen/Downloads/财务数据AI训练_测试代码_20251017/mac-test-venv/bin/python
# -*- coding: utf-8 -*-
"""
综合财务 + K线数据抓取脚本 v0.3

功能：
1. 读取股票列表，逐家公司拉取财务数据与日K/周K数据
2. 每家公司生成一个独立Excel，包含财务数据、日K、周K、财报×日K对齐结果
3. 自动生成company_index索引文件与progress断点文件，支持断点续传
4. 财报-日K对齐：以"财报发布日期：财报公开日期、公告日期"为基准，向前填充覆盖所有日K日期

v0.3 更新：
- 切换到 eastmoney_v0.7 模块（增强请求头 + 详细错误日志）
- 详细记录所有K线获取失败原因，保存到独立日志文件
- 在公司索引中增加"日K失败原因"和"周K失败原因"字段
- 连续5次K线获取失败则停止（已在v0.2实现，v0.3保持）
- 成功后失败计数归零（已在v0.2实现，v0.3保持）
- 批量暂停控制：每成功处理n家公司后，暂停m分钟（可配置）
- 财务数据失败跳过策略：当财务数据获取失败时，直接跳过该公司的K线数据获取
"""

import os
import sys
import json
import time
import random
import traceback
import importlib.util
from datetime import datetime, timedelta

import pandas as pd


# =========================== 配置区域（可修改） ===========================
# 股票列表及筛选
STOCK_LIST_FILE = "docs\股票代码汇总-陈俊同-20251118.xlsx"
SHEET_NAME = 0                    # Excel sheet索引或名称
FILTER_BY_COLUMN = None        # 为None则不筛选
FILTER_CODES = None               # 如 ['600519', '00700.HK']，为None处理全部

# 财务映射文件
MAPPING_FILE_PATH = "docs\东方财富财务数据API映射最终版-陈俊同-20251030.xlsx"

# K线抓取控制
FETCH_DAILY = True
FETCH_WEEKLY = True

# K线时间范围默认值（格式："YYYYMMDD-YYYYMMDD" 或 None表示使用365天）
# 注：如果API返回的数据不足设置的范围，则返回实际可用的数据量
DEFAULT_DAILY_RANGE = "20200101-20251115"    # 日K默认范围
DEFAULT_WEEKLY_RANGE = "20200101-20251115"   # 周K默认范围

DELAY_SECONDS = (15, 20)          # 每次请求随机延迟区间
MAX_CONSECUTIVE_FAILURES = 5      # 连续失败判定IP封禁

# 批量暂停控制（每成功n家后暂停m分钟，避免过于频繁请求）
PAUSE_AFTER_N_SUCCESS = 20        # 每成功处理n家公司后暂停（设为0则不启用）
PAUSE_MINUTES = 5                # 暂停时长（分钟）

# 输出与断点
OUTPUT_BASE_DIR = os.path.join("data", "stock")
RESUME_PROGRESS_FILE = ''       # 若需续传，填入progress文件绝对或相对路径

# 对齐设置
NOTICE_COL = "财报发布日期：财报公开日期、公告日期"

# ========================================================================


# ---------- 动态导入依赖脚本 ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# financial_data_mapper_v3.21_batch_period.py
_financial_module_path = os.path.join(BASE_DIR, "spider", "financial_data_mapper_v3.21_batch_period.py")
_financial_spec = importlib.util.spec_from_file_location("financial_mapper_v321", _financial_module_path)
financial_module = importlib.util.module_from_spec(_financial_spec)
_financial_spec.loader.exec_module(financial_module)

# eastmoney_v0.7（用于日K/周K）
sys.path.append(os.path.join(BASE_DIR, "spider"))
_eastmoney_path = os.path.join(BASE_DIR, "spider", "eastmoney_v0.7.py")
_eastmoney_spec = importlib.util.spec_from_file_location("eastmoney_v0_7", _eastmoney_path)
eastmoney_module = importlib.util.module_from_spec(_eastmoney_spec)
_eastmoney_spec.loader.exec_module(eastmoney_module)
get_historical_data = eastmoney_module.get_historical_data


class IPBlockedException(Exception):
    """IP被封禁异常"""


def ensure_output_dir(base_dir):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(base_dir, f"download_financial&candle_v0.3_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir, timestamp


def parse_date_range(date_str):
    """解析 'YYYYMMDD-YYYYMMDD' 字符串"""
    if pd.isna(date_str) or not date_str:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        return start_date.strftime("%Y%m%d"), end_date.strftime("%Y%m%d"), 365

    try:
        date_str = str(date_str).strip()
        parts = date_str.split('-')
        if len(parts) != 2:
            return None
        start = datetime.strptime(parts[0].strip(), "%Y%m%d")
        end = datetime.strptime(parts[1].strip(), "%Y%m%d")
        days = (end - start).days
        if days <= 0:
            return None
        return parts[0].strip(), parts[1].strip(), days
    except Exception:
        return None


def get_random_delay():
    if isinstance(DELAY_SECONDS, (list, tuple)):
        return random.uniform(float(DELAY_SECONDS[0]), float(DELAY_SECONDS[1]))
    return float(DELAY_SECONDS)




class CandleFetcher:
    """K线数据获取器（v0.3 增强版）
    
    v0.3 新增：
    - 详细记录每次失败的原因、时间、股票代码等信息
    - 提供失败日志列表供外部保存
    """
    def __init__(self):
        self.consecutive_failures = 0
        self.max_consecutive_failures = MAX_CONSECUTIVE_FAILURES
        self.ip_blocked = False
        # v0.3 新增：失败日志记录
        self.failure_log = []

    def _is_ip_blocked(self, error):
        message = str(error).lower()
        keywords = ["403", "429", "forbidden", "too many requests", "访问频繁", "ip限制", "blocked", "rate limit"]
        if any(keyword in message for keyword in keywords):
            return True
        if isinstance(error, ConnectionRefusedError):
            return True
        return False

    def _log_failure(self, stock_code, market, period, error_msg, error_type="unknown"):
        """记录失败详情到日志"""
        log_entry = {
            "时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "股票代码": stock_code,
            "市场": market,
            "周期": period,
            "错误类型": error_type,
            "错误信息": error_msg,
            "连续失败次数": self.consecutive_failures
        }
        self.failure_log.append(log_entry)

    def fetch(self, stock_code, market, period, date_range=None):
        """
        获取K线数据，失败时累加计数器并记录详细原因
        
        v0.3 修改：
        - 所有失败场景都会记录到 failure_log
        - 返回值增加失败原因字符串
        
        Returns:
            tuple: (DataFrame, count, error_message)
                - DataFrame: K线数据（失败时为None）
                - count: 数据行数（失败时为0）
                - error_message: 失败原因（成功时为空字符串）
        """
        if not (FETCH_DAILY or FETCH_WEEKLY):
            return None, 0, ""

        parsed = parse_date_range(date_range) if date_range else None
        if parsed is None:
            parsed = parse_date_range(None)
        _, _, days = parsed

        try:
            df = get_historical_data(stock_code=stock_code, market=market, period=period, days=days)
            if df is not None and not df.empty:
                # 成功获取数据，重置计数器
                self.consecutive_failures = 0
                return df, len(df), ""
            
            # 数据为空也算失败
            self.consecutive_failures += 1
            error_msg = f"获取{period}数据为空"
            self._log_failure(stock_code, market, period, error_msg, "空数据")
            print(f"  ✗ {error_msg}（连续失败{self.consecutive_failures}次）")
            
            if self.consecutive_failures >= self.max_consecutive_failures:
                self.ip_blocked = True
                raise IPBlockedException(
                    f"K线数据连续{self.consecutive_failures}次获取失败（包括空数据），疑似IP被封"
                )
            
            return None, 0, error_msg
            
        except IPBlockedException:
            # IP封禁异常直接向上抛出
            raise
            
        except Exception as exc:
            # 检测IP封禁特征
            if self._is_ip_blocked(exc):
                self.ip_blocked = True
                error_msg = f"检测到IP封禁特征：{exc}"
                self._log_failure(stock_code, market, period, error_msg, "IP封禁")
                raise IPBlockedException(error_msg)

            # 其他异常也累加计数器
            self.consecutive_failures += 1
            error_msg = str(exc)
            self._log_failure(stock_code, market, period, error_msg, "异常")
            print(f"  ✗ 获取{period}数据失败（连续失败{self.consecutive_failures}次）：{exc}")
            
            if self.consecutive_failures >= self.max_consecutive_failures:
                self.ip_blocked = True
                raise IPBlockedException(
                    f"K线数据连续{self.consecutive_failures}次请求失败，疑似IP被封：{exc}"
                )
            
            return None, 0, error_msg

    def get_failure_log_df(self):
        """将失败日志转换为DataFrame"""
        if not self.failure_log:
            return pd.DataFrame()
        return pd.DataFrame(self.failure_log)


def align_financial_with_daily(fin_df, daily_df):
    if fin_df is None or fin_df.empty or daily_df is None or daily_df.empty:
        return pd.DataFrame()

    df_fin = fin_df.copy()
    if NOTICE_COL not in df_fin.columns:
        print("  ⚠ 财务数据缺少公告日期列，无法对齐")
        return pd.DataFrame()

    df_fin[NOTICE_COL] = pd.to_datetime(df_fin[NOTICE_COL], errors="coerce")
    df_fin = df_fin.dropna(subset=[NOTICE_COL])
    if df_fin.empty:
        return pd.DataFrame()

    df_daily = daily_df.copy()
    df_daily["日期"] = pd.to_datetime(df_daily["日期"], errors="coerce")
    df_daily = df_daily.dropna(subset=["日期"])
    if df_daily.empty:
        return pd.DataFrame()

    # 按时间排序，不再提前过滤财报范围
    df_fin = df_fin.sort_values(NOTICE_COL).reset_index(drop=True)
    df_daily = df_daily.sort_values("日期").reset_index(drop=True)

    # merge_asof 自动匹配：每条日K找最近的不晚于它的财报（包括早于日K起点的财报）
    merged = pd.merge_asof(
        df_daily,
        df_fin,
        left_on="日期",
        right_on=NOTICE_COL,
        direction="backward"
    )

    # 对于没有匹配到任何财报的日K行（最早财报之前且无更早财报可用），填充0
    # 判断依据：NOTICE_COL列为空
    no_match_mask = merged[NOTICE_COL].isna()
    if no_match_mask.any():
        financial_cols = [col for col in df_fin.columns if col != NOTICE_COL]
        for col in financial_cols + [NOTICE_COL]:
            if col not in merged.columns:
                continue
            if merged[col].dtype.kind in ("i", "u", "f"):
                merged.loc[no_match_mask, col] = 0
            else:
                merged.loc[no_match_mask, col] = ""

    merged = merged.rename(columns={NOTICE_COL: "匹配财报公告日期"})
    merged["日期"] = pd.to_datetime(merged["日期"], errors="coerce")
    merged = merged.sort_values("日期", ascending=False).reset_index(drop=True)
    merged["日期"] = merged["日期"].dt.strftime("%Y-%m-%d").fillna("")
    if "匹配财报公告日期" in merged.columns:
        matched_dates = pd.to_datetime(merged["匹配财报公告日期"], errors="coerce")
        merged["匹配财报公告日期"] = matched_dates.dt.strftime("%Y-%m-%d").fillna("")
    return merged


class ProgressManager:
    def __init__(self, output_dir, timestamp, resume_file=None):
        self.output_dir = output_dir
        self.timestamp = timestamp
        self.progress_file = os.path.join(output_dir, f"progress_{timestamp}.json")
        self.processed_codes = set()
        self.total_processed = 0
        self.total_success = 0
        self.total_failed = 0
        if resume_file:
            self._load(resume_file)

    def _load(self, path):
        if not os.path.isabs(path):
            path = os.path.join(self.output_dir, os.path.basename(path))
        if not os.path.exists(path):
            print(f"⚠️ 进度文件不存在：{path}")
            return
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.timestamp = data.get("timestamp", self.timestamp)
        self.processed_codes = set(data.get("processed_codes", []))
        self.total_processed = data.get("total_processed", 0)
        self.total_success = data.get("total_success", 0)
        self.total_failed = data.get("total_failed", 0)
        self.progress_file = path
        print(f"✓ 已加载进度：处理{self.total_processed}家，成功{self.total_success}，失败{self.total_failed}")

    def save(self):
        os.makedirs(self.output_dir, exist_ok=True)
        save_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        data = {
            "timestamp": self.timestamp,
            "save_time": save_time,
            "processed_codes": list(self.processed_codes),
            "total_processed": self.total_processed,
            "total_success": self.total_success,
            "total_failed": self.total_failed,
        }
        with open(self.progress_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


class CompanyProcessor:
    def __init__(self):
        self.mapper = financial_module.FinancialDataMapper(MAPPING_FILE_PATH)
        self.progress = None
        self.output_dir = None
        self.timestamp = None
        self.session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.candle_fetcher = CandleFetcher()
        self.index_records = []
        self.session_success_count = 0  # v0.3 新增：本次运行成功处理的公司数量

    def setup(self):
        if RESUME_PROGRESS_FILE:
            self.output_dir = os.path.dirname(RESUME_PROGRESS_FILE)
        else:
            self.output_dir, self.timestamp = ensure_output_dir(OUTPUT_BASE_DIR)
            self.session_timestamp = self.timestamp
        self.progress = ProgressManager(
            output_dir=self.output_dir,
            timestamp=self.timestamp or self.session_timestamp,
            resume_file=RESUME_PROGRESS_FILE
        )
        if RESUME_PROGRESS_FILE:
            self.timestamp = self.progress.timestamp
            self.session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # 续传时加载已有的索引记录
            index_file = os.path.join(self.output_dir, f"company_index_{self.timestamp}.xlsx")
            if os.path.exists(index_file):
                try:
                    existing_index = pd.read_excel(index_file, sheet_name="公司索引")
                    self.index_records = existing_index.to_dict("records")
                    print(f"✓ 已加载 {len(self.index_records)} 条历史索引记录")
                except Exception as e:
                    print(f"⚠️ 加载索引文件失败: {e}，将创建新索引")
        print(f"\n输出目录：{self.output_dir}")

    def load_stock_dataframe(self):
        print(f"正在读取股票列表：{STOCK_LIST_FILE}")
        
        # 定义清理函数，直接在读取时应用
        def clean_name(name):
            if name is None or pd.isna(name):
                return ""
            s = str(name)
            for char in ["\r", "\n", "_x000D_", "_x000d_", "_x000A_", "_x000a_"]:
                s = s.replace(char, "")
            return s.strip()
        
        # 读取时指定converters，在解析阶段就清理公司名称
        converters = {}
        if "公司名称" in pd.read_excel(STOCK_LIST_FILE, sheet_name=SHEET_NAME, nrows=0).columns:
            converters["公司名称"] = clean_name
        
        df = pd.read_excel(STOCK_LIST_FILE, sheet_name=SHEET_NAME, converters=converters)
        if FILTER_BY_COLUMN and FILTER_BY_COLUMN in df.columns:
            df = df[df[FILTER_BY_COLUMN].notna()]
        if FILTER_CODES:
            if isinstance(FILTER_CODES, (list, tuple, set)):
                codes_iter = FILTER_CODES
            else:
                codes_iter = [FILTER_CODES]
            filter_set = {str(code).strip() for code in codes_iter}
            df = df[df["股票代码"].astype(str).str.strip().isin(filter_set)]
        if "市场类型" not in df.columns:
            df["市场类型"] = df["股票代码"].apply(financial_module.detect_market_type)
        df = df.reset_index(drop=True)

        if self.progress and self.progress.processed_codes:
            processed = {str(code).strip() for code in self.progress.processed_codes}
            before = len(df)
            df = df[~df["股票代码"].astype(str).str.strip().isin(processed)].reset_index(drop=True)
            skipped = before - len(df)
            if skipped > 0:
                print(f"ℹ️  已跳过 {skipped} 家已完成的公司（基于进度文件）")

        return df

    def process_all(self):
        self.setup()
        stock_df = self.load_stock_dataframe()
        if stock_df.empty:
            print("✗ 没有可处理的公司")
            return

        start_time = time.time()
        for _, row in stock_df.iterrows():
            stock_code = str(row["股票代码"]).strip()

            company_name = str(row.get("公司名称", stock_code)).strip() or stock_code
            market = row.get("市场类型") or financial_module.detect_market_type(stock_code)
            print(f"\n{'='*60}")
            next_seq = self.progress.total_processed + 1
            print(f"[{next_seq}] 处理 {company_name} ({stock_code})")
            print(f"{'='*60}")

            daily_range = row.get("日K范围")
            weekly_range = row.get("周K范围")

            success = self.process_single_company(
                sequence=next_seq,
                stock_code=stock_code,
                company_name=company_name,
                market=market,
                daily_range=daily_range,
                weekly_range=weekly_range
            )

            if success:
                self.progress.total_success += 1
                self.session_success_count += 1  # v0.3 新增：累加本次运行成功数
                
                # v0.3 新增：检查是否需要批量暂停
                if PAUSE_AFTER_N_SUCCESS > 0 and self.session_success_count >= PAUSE_AFTER_N_SUCCESS:
                    print(f"\n⏸️  已成功处理 {self.session_success_count} 家公司，暂停 {PAUSE_MINUTES} 分钟以避免频繁请求...")
                    print(f"   暂停时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    pause_end_time = datetime.now() + timedelta(minutes=PAUSE_MINUTES)
                    print(f"   预计恢复：{pause_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
                    time.sleep(PAUSE_MINUTES * 60)
                    self.session_success_count = 0  # 重置计数器
                    print("✓  暂停结束，继续处理...\n")
            else:
                self.progress.total_failed += 1

            self.progress.total_processed += 1
            self.progress.processed_codes.add(stock_code)
            self.progress.save()
            # 索引已在 process_single_company 内部保存，此处不再重复调用

        elapsed = (time.time() - start_time) / 60
        print(f"\n处理完成！耗时 {elapsed:.1f} 分钟")
        print(f"成功 {self.progress.total_success} 家，失败 {self.progress.total_failed} 家")
        print(f"结果目录：{self.output_dir}")
        
        # v0.3 新增：保存失败日志
        self.save_failure_log()

    def process_single_company(self, sequence, stock_code, company_name, market, daily_range, weekly_range):
        # v0.3 新增：记录本次处理的日K/周K失败原因
        daily_error = ""
        weekly_error = ""
        
        try:
            fin_df, status = self.fetch_financial_data(stock_code, company_name)
            
            # v0.3 新增：财务数据失败时直接跳过K线获取
            if fin_df is None or fin_df.empty:
                print("  ✗ 财务数据为空，跳过K线数据获取")
                self.append_index_record(
                    sequence=sequence,
                    company_name=company_name,
                    stock_code=stock_code,
                    market=market,
                    status_info=status,
                    excel_path="",
                    fin_df=pd.DataFrame(),
                    daily_count=0,
                    weekly_count=0,
                    daily_error="未获取（财务数据失败）",
                    weekly_error="未获取（财务数据失败）",
                    process_status="失败",
                    failure_reason="财务数据为空"
                )
                self.save_company_index()
                return False

            daily_df, daily_count = (None, 0)
            weekly_df, weekly_count = (None, 0)

            if FETCH_DAILY:
                print("  📊 获取日K数据...")
                # 如果Excel中没有指定范围，使用配置的默认值
                actual_daily_range = daily_range if (pd.notna(daily_range) and daily_range) else DEFAULT_DAILY_RANGE
                daily_df, daily_count, daily_error = self.fetch_candles(stock_code, market, "daily", actual_daily_range)
                if daily_df is not None and not daily_df.empty:
                    print(f"    ✓ 日K: {daily_count} 条记录")
                else:
                    print("    ✗ 日K数据为空")

            if FETCH_WEEKLY:
                if FETCH_DAILY:
                    delay = get_random_delay()
                    print(f"  ⏱️  等待 {delay:.1f} 秒后获取周K数据...")
                    time.sleep(delay)
                print("  📊 获取周K数据...")
                # 如果Excel中没有指定范围，使用配置的默认值
                actual_weekly_range = weekly_range if (pd.notna(weekly_range) and weekly_range) else DEFAULT_WEEKLY_RANGE
                weekly_df, weekly_count, weekly_error = self.fetch_candles(stock_code, market, "weekly", actual_weekly_range)
                if weekly_df is not None and not weekly_df.empty:
                    print(f"    ✓ 周K: {weekly_count} 条记录")
                else:
                    print("    ✗ 周K数据为空")

            aligned_df = align_financial_with_daily(fin_df, daily_df) if FETCH_DAILY else pd.DataFrame()
            if FETCH_DAILY:
                if aligned_df is not None and not aligned_df.empty:
                    latest_date = aligned_df["日期"].iloc[0]
                    print(f"  🔗 财报×日K对齐完成：{len(aligned_df)} 行（最新日期 {latest_date}）")
                else:
                    print("  ⚠ 财报×日K对齐数据为空")

            excel_path = self.save_company_excel(
                sequence=sequence,
                company_name=company_name,
                stock_code=stock_code,
                fin_df=fin_df,
                daily_df=daily_df,
                weekly_df=weekly_df,
                aligned_df=aligned_df
            )

            self.append_index_record(
                sequence=sequence,
                company_name=company_name,
                stock_code=stock_code,
                market=market,
                status_info=status,
                excel_path=excel_path,
                fin_df=fin_df,
                daily_count=daily_count,
                weekly_count=weekly_count,
                daily_error=daily_error,
                weekly_error=weekly_error,
                process_status="成功",
                failure_reason=""
            )
            # 立即保存索引，避免延迟导致索引文件不同步
            self.save_company_index()
            return True

        except IPBlockedException as ip_exc:
            print(f"\n⚠️ 检测到IP封禁：{ip_exc}")
            self.append_index_record(
                sequence=sequence,
                company_name=company_name,
                stock_code=stock_code,
                market=market,
                status_info=None,
                excel_path="",
                fin_df=pd.DataFrame(),
                daily_count=0,
                weekly_count=0,
                daily_error=daily_error or str(ip_exc),
                weekly_error=weekly_error or str(ip_exc),
                process_status="失败",
                failure_reason="IP封禁"
            )
            self.progress.save()
            self.save_company_index()
            raise
        except Exception as exc:
            print(f"✗ 处理失败：{exc}")
            traceback.print_exc()
            self.append_index_record(
                sequence=sequence,
                company_name=company_name,
                stock_code=stock_code,
                market=market,
                status_info=None,
                excel_path="",
                fin_df=pd.DataFrame(),
                daily_count=0,
                weekly_count=0,
                daily_error=daily_error,
                weekly_error=weekly_error,
                process_status="失败",
                failure_reason=str(exc)
            )
            # 失败时也立即保存索引
            self.save_company_index()
            return False
        finally:
            delay = get_random_delay()
            if delay > 0:
                print(f"  ⏱️ 等待 {delay:.1f} 秒后继续...")
                time.sleep(delay)

    def fetch_financial_data(self, stock_code, company_name):
        result = financial_module.get_single_stock_data(stock_code, company_name)
        df = result.get("data", pd.DataFrame())
        status = result.get("status", {})
        market_type = financial_module.detect_market_type(stock_code)

        if df.empty:
            return df, status

        if "REPORT_DATE" in df.columns:
            df = df.sort_values("REPORT_DATE", ascending=False)

        df = financial_module.format_date_columns(df)
        df = financial_module.add_notice_date_column(df, stock_code, market_type)
        df = financial_module.merge_duplicate_report_dates(df)
        df = self.mapper.map_dataframe(df, market_type, stock_code)
        df = financial_module.add_missing_columns_and_sort(df, market_type, self.mapper)
        df = financial_module.calculate_free_cash_flow(df, stock_code)
        df = financial_module.fill_missing_values_with_zero(df)
        return df, status

    def fetch_candles(self, stock_code, market, period, date_range):
        """v0.3 修改：返回三元组 (df, count, error_message)"""
        df, count, error_msg = self.candle_fetcher.fetch(stock_code, market, period, date_range)
        if df is not None and not df.empty:
            df["日期"] = pd.to_datetime(df["日期"]).dt.strftime("%Y-%m-%d")
        return df, count, error_msg

    def save_company_excel(self, sequence, company_name, stock_code, fin_df, daily_df, weekly_df, aligned_df):
        # 生成当前保存时刻的时间戳
        current_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"{sequence}_{company_name}_{stock_code}_{current_timestamp}.xlsx"
        safe_name = "".join(c if c not in r'\/:*?"<>|' else "_" for c in file_name)
        file_path = os.path.join(self.output_dir, safe_name)

        with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
            fin_df.to_excel(writer, sheet_name="财务数据", index=False)
            if FETCH_DAILY and daily_df is not None and not daily_df.empty:
                daily_df.to_excel(writer, sheet_name="日K数据", index=False)
            if FETCH_WEEKLY and weekly_df is not None and not weekly_df.empty:
                weekly_df.to_excel(writer, sheet_name="周K数据", index=False)
            if aligned_df is not None and not aligned_df.empty:
                aligned_df.to_excel(writer, sheet_name="财报_日K对齐", index=False)
        print(f"  ✓ 已保存：{file_path}")
        return file_path

    def append_index_record(
        self, sequence, company_name, stock_code, market, status_info, excel_path,
        fin_df, daily_count, weekly_count, daily_error, weekly_error, process_status, failure_reason
    ):
        """v0.3 修改：增加 daily_error 和 weekly_error 参数"""
        data_sources = ""
        indicator_status = "未获取"
        indicator_error = ""
        indicator_period = ""
        statements_status = "未获取"
        statements_error = ""
        statements_period = ""

        if status_info:
            data_sources = ", ".join(status_info.get("data_sources", []))
            indicator = status_info.get("indicator", {})
            statements = status_info.get("statements", {})
            indicator_status = "成功" if indicator.get("success") else "失败"
            indicator_error = indicator.get("error") or ""
            indicator_period = indicator.get("period_used", "")
            statements_status = "成功" if statements.get("success") else "失败"
            statements_error = statements.get("error") or ""
            statements_period = statements.get("period_used", "")

        record = {
            "序号": sequence,
            "公司名称": company_name,
            "股票代码": stock_code,
            "市场类型": market,
            "状态": "成功" if process_status == "成功" else "失败",
            "Excel文件": os.path.basename(excel_path) if excel_path else "",
            "数据行数": len(fin_df) if fin_df is not None else 0,
            "数据列数": len(fin_df.columns) if fin_df is not None and not fin_df.empty else 0,
            "错误信息": failure_reason if failure_reason else "",
            "财务指标状态": indicator_status,
            "财务指标错误": indicator_error,
            "财务报表状态": statements_status,
            "财务报表错误": statements_error,
            "数据来源": data_sources,
            "财务指标周期": indicator_period,
            "财务报表周期": statements_period,
            "日K数据量": daily_count,
            "周K数据量": weekly_count,
            "日K失败原因": daily_error,  # v0.3 新增
            "周K失败原因": weekly_error,  # v0.3 新增
            "处理状态": process_status,
            "失败原因": failure_reason
        }
        self.index_records.append(record)

    def save_company_index(self):
        if not self.index_records:
            return
        index_df = pd.DataFrame(self.index_records)
        index_file = os.path.join(self.output_dir, f"company_index_{self.timestamp}.xlsx")
        with pd.ExcelWriter(index_file, engine="openpyxl") as writer:
            index_df.to_excel(writer, sheet_name="公司索引", index=False)
        print(f"  ✓ 索引已更新：{index_file}")

    def save_failure_log(self):
        """v0.3 新增：保存K线获取失败的详细日志"""
        failure_df = self.candle_fetcher.get_failure_log_df()
        if failure_df.empty:
            print("\n✓ 无K线获取失败记录")
            return
        
        log_file = os.path.join(self.output_dir, f"kline_failure_log_{self.timestamp}.xlsx")
        try:
            with pd.ExcelWriter(log_file, engine="openpyxl") as writer:
                failure_df.to_excel(writer, sheet_name="K线失败日志", index=False)
            print(f"\n✓ K线失败日志已保存：{log_file}")
            print(f"  共记录 {len(failure_df)} 次失败")
        except Exception as e:
            print(f"\n⚠️ 保存K线失败日志时出错：{e}")


def main():
    processor = CompanyProcessor()
    try:
        processor.process_all()
    except IPBlockedException:
        print("程序因IP限制而停止，请稍后重试或更换网络。")
    except KeyboardInterrupt:
        print("\n检测到中断，已保存当前进度。")
    except Exception as exc:
        print(f"\n发生未预期异常：{exc}")
        traceback.print_exc()
        if processor.progress:
            processor.progress.save()
            processor.save_company_index()


if __name__ == "__main__":
    main()





