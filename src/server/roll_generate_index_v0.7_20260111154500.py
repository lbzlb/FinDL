#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
滚动窗口数据集索引生成工具 v0.7
用于从股票数据中生成时间序列滚动窗口样本索引

版本: v0.7
日期: 20260111

主要变化：
1. 不拷贝文件，只记录原始文件路径
2. 预测目标改为未来第N天（单日），而不是未来N天窗口
3. 索引结构简化，target_row代替output_row_start/end
4. v0.4修改：只要训练集或验证集中任意一边达到最小数据量要求即可（不再要求两边都满足）
5. v0.5新增：训练集和验证集可以分别设置不同的滚动步长（例如训练集每100天，验证集每10天）
6. v0.6修改：删除冗余的验证检查，简化验证逻辑
7. v0.7修改：训练集数据范围扩展（基于交易日行号），使训练集和验证集在时间上更接近
   - 以SPLIT_DATE在数据中的行号位置为基准（split_row）
   - 训练集输入窗口最后一行 < split_row + INPUT_WINDOW（500个交易日）
   - 训练集目标行 < split_row + TRAIN_EXTENSION_DAYS（620个交易日）
   - TRAIN_EXTENSION_DAYS = INPUT_WINDOW + PREDICTION_DAY（自动计算）
   - 验证集从 SPLIT_DATE 开始，保持不变

v0.7的核心逻辑（基于交易日行号）：
- 找到SPLIT_DATE对应的行号split_row
- 训练集输入窗口最后一行号 < split_row + 500
- 训练集目标行号 < split_row + 620
- 验证集输入窗口起始行号 >= split_row
- 确保训练集和验证集目标日期不重叠
"""

import sys
from pathlib import Path
from datetime import datetime
import json
import pandas as pd
from tqdm import tqdm

# ============================================================================
# 配置区域：所有参数都在这里设置
# ============================================================================
# 数据源路径
SOURCE_DATA_DIR = '/data/project_20251211/data/raw/processed_data_20260308'

# 输出路径
OUTPUT_BASE_DIR = '/data/project_20251211/data/processed'

# 时间范围筛选
START_DATE = '2010-01-01'  # 数据开始时间
END_DATE = '2050-12-31'    # 数据结束时间

# 滚动窗口参数
INPUT_WINDOW = 500   # 输入窗口大小（用于训练的历史数据长度）
PREDICTION_DAY = 120   # 预测未来第几天（例如30表示预测第30天）

# 训练集/验证集滚动步长（v0.5新增）
TRAIN_STRIDE = 20   # 训练集滚动步长（每次滚动多少天）
VAL_STRIDE = 3      # 验证集滚动步长（每次滚动多少天）

# 训练集/验证集划分
SPLIT_DATE = '2023-01-01'  # 时间分界点，之前为训练集，之后为验证集

# v0.7新增：训练集扩展交易日数（自动计算）
# 以SPLIT_DATE在数据中的行号为基准，训练集目标行可以扩展到 split_row + TRAIN_EXTENSION_DAYS
TRAIN_EXTENSION_DAYS = INPUT_WINDOW + PREDICTION_DAY  # 620个交易日

# 注意：程序会自动计算 MIN_REQUIRED_DAYS = INPUT_WINDOW + PREDICTION_DAY
#       只有满足以下条件的公司才会被处理：
#       1. 时间范围内总数据天数 >= MIN_REQUIRED_DAYS
#       2. v0.7修改：训练集基于交易日行号扩展
#          - split_row = SPLIT_DATE在数据中的行号位置
#          - 训练集输入窗口最后一行 < split_row + INPUT_WINDOW (500个交易日)
#          - 训练集目标行 < split_row + TRAIN_EXTENSION_DAYS (620个交易日)
#          - 验证集输入窗口起始行 >= split_row
# ============================================================================


class RollingDatasetGenerator:
    """滚动窗口数据集生成器 v0.7"""
    
    def __init__(self):
        self.source_dir = Path(SOURCE_DATA_DIR)
        self.output_base_dir = Path(OUTPUT_BASE_DIR)
        self.start_date = pd.to_datetime(START_DATE)
        self.end_date = pd.to_datetime(END_DATE)
        self.input_window = INPUT_WINDOW
        self.prediction_day = PREDICTION_DAY
        self.train_stride = TRAIN_STRIDE
        self.val_stride = VAL_STRIDE
        self.split_date = pd.to_datetime(SPLIT_DATE)
        self.train_extension_days = TRAIN_EXTENSION_DAYS
        self.min_required_days = INPUT_WINDOW + PREDICTION_DAY
        
        # v0.7: 训练集边界基于交易日行号（在处理每个公司时动态计算）
        # 训练集输入窗口最后一行 < split_row + INPUT_WINDOW
        # 训练集目标行 < split_row + TRAIN_EXTENSION_DAYS
        
        # 统计信息
        self.stats = {
            'total_files': 0,
            'valid_companies': 0,
            'skipped_companies': 0,
            'total_samples': 0,
            'train_samples': 0,
            'val_samples': 0,
            'skipped_details': [],
            # 异常检测统计
            'validation_errors': {
                'row_out_of_range': 0,
                'target_row_out_of_range': 0,
                'row_diff_mismatch': 0,  # 行号差异不匹配（真错误）
                'target_date_overflow': 0,  # v0.7新增：目标日期超出训练集上限
                'total_invalid_samples': 0
            },
            'validation_warnings': [],
            # 差异分布统计（用于分析）
            'row_diff_distribution': {}  # {行号差异: 样本数量}
        }
        
        # 创建输出目录（不创建filtered_companies子目录）
        self.output_dir = self._create_output_dir()
        
        # 日志文件
        self.log_file = self.output_dir / 'processing_log.txt'
        self.log_buffer = []
    
    def _create_output_dir(self):
        """创建输出目录"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        dir_name = f'roll_generate_index_v0.7_{timestamp}'
        output_dir = self.output_base_dir / dir_name
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
    
    def log(self, message, level='INFO'):
        """记录日志"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_line = f'{timestamp} | {level:5s} | {message}'
        print(log_line)
        self.log_buffer.append(log_line)
        
        # 每100条日志写入一次文件
        if len(self.log_buffer) >= 100:
            self._flush_log()
    
    def _flush_log(self):
        """将日志缓冲区写入文件"""
        if self.log_buffer:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write('\n'.join(self.log_buffer) + '\n')
            self.log_buffer = []
    
    def validate_sample(self, sample, df, company_name, stock_code, split_row=None):
        """
        验证单个样本的有效性
        
        Args:
            sample: 样本字典
            df: 原始数据DataFrame
            company_name: 公司名称
            stock_code: 股票代码
            split_row: SPLIT_DATE在df中的行号位置（用于v0.7训练集边界检查）
        
        Returns:
            (is_valid, error_messages)
        """
        errors = []
        
        # 1. 检查行号范围
        total_rows = len(df)
        input_start = sample['input_row_start']
        input_end = sample['input_row_end']
        target_row = sample['target_row']
        
        if input_start < 0 or input_start >= total_rows:
            errors.append(f"input_row_start ({input_start}) 超出范围 [0, {total_rows})")
            self.stats['validation_errors']['row_out_of_range'] += 1
        
        if input_end < 0 or input_end >= total_rows:
            errors.append(f"input_row_end ({input_end}) 超出范围 [0, {total_rows})")
            self.stats['validation_errors']['row_out_of_range'] += 1
        
        if input_start > input_end:
            errors.append(f"input_row_start ({input_start}) > input_row_end ({input_end})")
            self.stats['validation_errors']['row_out_of_range'] += 1
        
        if target_row < 0 or target_row >= total_rows:
            errors.append(f"target_row ({target_row}) 超出范围 [0, {total_rows})")
            self.stats['validation_errors']['target_row_out_of_range'] += 1
        
        # 2. 检查行号差异（预测交易日数）
        try:
            row_diff = target_row - input_end
            if row_diff != self.prediction_day:
                errors.append(f"预测交易日数不匹配: 期望={self.prediction_day}, 实际={row_diff}")
                self.stats['validation_errors']['row_diff_mismatch'] += 1
            
            # 统计行号差异分布（用于分析）
            row_diff_key = str(row_diff)
            self.stats['row_diff_distribution'][row_diff_key] = \
                self.stats['row_diff_distribution'].get(row_diff_key, 0) + 1
        except Exception as e:
            errors.append(f"行号差异检查失败: {str(e)}")
        
        # 3. v0.7新增：对训练集样本，检查行号是否超出上限（基于交易日）
        if sample['split'] == 'train' and split_row is not None:
            # 检查输入窗口最后一行是否超限
            train_input_end_limit = split_row + self.input_window - 1
            if input_end >= split_row + self.input_window:
                errors.append(f"训练集输入窗口最后一行 ({input_end}) >= 上限 (split_row + {self.input_window} = {split_row + self.input_window})")
                self.stats['validation_errors']['target_date_overflow'] += 1
            
            # 检查目标行是否超限
            train_target_limit = split_row + self.train_extension_days - 1
            if target_row >= split_row + self.train_extension_days:
                errors.append(f"训练集目标行 ({target_row}) >= 上限 (split_row + {self.train_extension_days} = {split_row + self.train_extension_days})")
                self.stats['validation_errors']['target_date_overflow'] += 1
        
        is_valid = len(errors) == 0
        
        if not is_valid:
            self.stats['validation_errors']['total_invalid_samples'] += 1
            sample_id_str = sample.get('sample_id', '临时样本(未分配ID)')
            error_msg = f"样本 {sample_id_str} ({company_name}/{stock_code}) 验证失败: {'; '.join(errors)}"
            self.stats['validation_warnings'].append(error_msg)
        
        return is_valid, errors
    
    def scan_company_files(self):
        """扫描所有公司数据文件"""
        parquet_files = list(self.source_dir.glob('*.parquet'))
        
        # 按文件名中的数字ID进行排序（而不是字符串排序）
        def extract_numeric_id(file_path):
            """从文件名中提取数字ID，例如: 123_公司名_xxx.parquet -> 123"""
            try:
                filename = file_path.stem  # 去掉扩展名
                # 提取第一个下划线之前的数字
                id_str = filename.split('_')[0]
                return int(id_str)
            except (ValueError, IndexError):
                # 如果无法提取数字，返回一个很大的数（排在最后）
                return 999999
        
        parquet_files = sorted(parquet_files, key=extract_numeric_id)
        self.stats['total_files'] = len(parquet_files)
        self.log(f'找到 {len(parquet_files)} 个公司文件')
        return parquet_files
    
    def parse_filename(self, file_path):
        """解析文件名获取公司信息"""
        filename = file_path.stem
        parts = filename.split('_')
        if len(parts) >= 3:
            company_id = parts[0]
            company_name = parts[1]
            stock_code = parts[2]
            return company_id, company_name, stock_code
        return None, None, None
    
    def _generate_samples_for_split(self, split_df, filtered_df, original_start_idx, df, company_id, 
                                   company_name, stock_code, file_path, stride, split_name, global_sample_id, 
                                   split_row=None, max_input_end_row=None, max_target_row=None):
        """
        为指定的数据集（训练集或验证集）生成滚动窗口样本索引
        
        Args:
            split_df: 训练集或验证集的DataFrame（已筛选时间范围）
            filtered_df: 筛选时间范围后的完整DataFrame（用于计算索引偏移）
            original_start_idx: 筛选后数据在原始文件中的起始行号
            df: 原始数据DataFrame
            company_id: 公司ID
            company_name: 公司名称
            stock_code: 股票代码
            file_path: 文件路径
            stride: 滚动步长
            split_name: 数据集名称（'train' 或 'val'）
            global_sample_id: 全局样本ID起始值
            split_row: SPLIT_DATE在df中的行号位置（用于v0.7训练集边界检查）
            max_input_end_row: 训练集输入窗口最后一行的上限（仅用于训练集）
            max_target_row: 训练集目标行的上限（仅用于训练集）
        
        Returns:
            (samples_list, next_global_sample_id)
        """
        samples = []
        
        if len(split_df) < self.min_required_days:
            return samples, global_sample_id
        
        # 计算理论最大样本数（不考虑stride）
        max_samples_without_stride = len(split_df) - self.min_required_days + 1
        # 计算考虑stride后的理论样本数
        num_possible_samples = (max_samples_without_stride + stride - 1) // stride
        
        # 找到split_df在filtered_df中的起始位置
        # split_df的第一行日期在filtered_df中的位置索引（位置索引，不是标签索引）
        split_first_date = split_df.iloc[0]['日期']
        # 使用位置索引而不是标签索引
        mask = filtered_df['日期'] == split_first_date
        if mask.any():
            # 获取标签索引，然后转换为位置索引
            label_idx = filtered_df[mask].index[0]
            split_start_in_filtered = filtered_df.index.get_loc(label_idx)
        else:
            # 如果找不到（理论上不应该发生），使用重置索引的方法
            filtered_df_reset = filtered_df.reset_index(drop=True)
            split_start_in_filtered = filtered_df_reset[filtered_df_reset['日期'] == split_first_date].index[0]
        
        # split_df的第一行在原始df中的索引 = original_start_idx + split_start_in_filtered
        split_start_in_original = original_start_idx + split_start_in_filtered
        
        for i in range(0, max_samples_without_stride, stride):
            # 在split_df中的索引
            split_input_start = i
            split_input_end = i + self.input_window - 1
            split_target_idx = i + self.input_window + self.prediction_day - 1
            
            # 检查是否超出split_df的范围
            if split_target_idx >= len(split_df):
                # 跳过这个样本，因为目标日期超出实际数据范围
                continue
            
            # 转换为原始文件中的索引
            input_start_idx = split_start_in_original + split_input_start
            input_end_idx = split_start_in_original + split_input_end
            target_idx = split_start_in_original + split_target_idx
            
            # v0.7新增：对训练集样本，检查是否超出行号上限
            if split_name == 'train' and (max_input_end_row is not None or max_target_row is not None):
                # 检查输入窗口最后一行
                if max_input_end_row is not None and input_end_idx >= max_input_end_row:
                    continue  # 跳过超出上限的样本
                # 检查目标行
                if max_target_row is not None and target_idx >= max_target_row:
                    continue  # 跳过超出上限的样本
            
            # 获取日期信息
            start_date = split_df.iloc[split_input_start]['日期']
            input_end_date = split_df.iloc[split_input_end]['日期']
            target_date = split_df.iloc[split_target_idx]['日期']
            
            # 先创建临时样本对象用于验证（不包含sample_id）
            temp_sample = {
                'company_id': company_id,
                'company_name': company_name,
                'stock_code': stock_code,
                'start_date': start_date,
                'input_end_date': input_end_date,
                'target_date': target_date,
                'input_row_start': input_start_idx,
                'input_row_end': input_end_idx,
                'target_row': target_idx,
                'prediction_day': self.prediction_day,
                'source_file': str(file_path),
                'split': split_name
            }
            
            # 验证样本
            is_valid, validation_msgs = self.validate_sample(temp_sample, df, company_name, stock_code, split_row)
            
            if is_valid:
                # 验证通过后，分配sample_id并添加到samples
                sample = temp_sample.copy()
                sample['sample_id'] = global_sample_id
                samples.append(sample)
                global_sample_id += 1
            else:
                # 记录验证失败的样本（使用临时ID用于日志）
                if len(validation_msgs) > 0:
                    self.log(
                        f'样本 (临时ID, 未分配) ({company_name}/{stock_code}, {split_name}) 验证失败: {validation_msgs[0]}',
                        'WARN'
                    )
                # 跳过无效样本，不分配sample_id
        
        return samples, global_sample_id
    
    def process_company(self, file_path, global_sample_id):
        """处理单个公司数据"""
        company_id, company_name, stock_code = self.parse_filename(file_path)
        
        if not company_name:
            self.log(f'无法解析文件名: {file_path.name}', 'WARN')
            self.stats['skipped_companies'] += 1
            return [], global_sample_id
        
        try:
            # 读取数据
            df = pd.read_parquet(file_path)
            
            # 确保日期列是datetime类型
            if '日期' not in df.columns:
                self.log(f'公司 {company_name}({stock_code}) 缺少日期列', 'ERROR')
                self.stats['skipped_companies'] += 1
                self.stats['skipped_details'].append({
                    'company': company_name,
                    'code': stock_code,
                    'reason': '缺少日期列'
                })
                return [], global_sample_id
            
            df['日期'] = pd.to_datetime(df['日期'])
            
            # 按日期排序
            df = df.sort_values('日期').reset_index(drop=True)
            
            # 筛选时间范围
            filtered_df = df[(df['日期'] >= self.start_date) & (df['日期'] <= self.end_date)].copy()
            
            # 检查总数据量
            if len(filtered_df) < self.min_required_days:
                self.log(
                    f'跳过 {company_name}({stock_code}): 时间范围内仅有 {len(filtered_df)} 天数据，'
                    f'需要至少 {self.min_required_days} 天 (输入{self.input_window} + 预测第{self.prediction_day}天)',
                    'WARN'
                )
                self.stats['skipped_companies'] += 1
                self.stats['skipped_details'].append({
                    'company': company_name,
                    'code': stock_code,
                    'reason': f'总数据不足: {len(filtered_df)}天 < {self.min_required_days}天'
                })
                return [], global_sample_id
            
            # 获取筛选后数据在原始文件中的起始行号
            original_start_idx = df[df['日期'] >= self.start_date].index[0]
            
            # v0.7修改：基于交易日行号划分训练集和验证集
            # 1. 找到SPLIT_DATE在filtered_df中的行号位置
            split_mask = filtered_df['日期'] >= self.split_date
            if split_mask.any():
                # 获取第一个>=SPLIT_DATE的位置索引
                split_idx_in_filtered = split_mask.idxmax()
                split_pos_in_filtered = filtered_df.index.get_loc(split_idx_in_filtered)
            else:
                # 如果没有>=SPLIT_DATE的数据，说明所有数据都在SPLIT_DATE之前
                split_pos_in_filtered = len(filtered_df)
            
            # 2. 计算SPLIT_DATE在原始df中的行号
            split_row_in_df = original_start_idx + split_pos_in_filtered
            
            # 3. 计算训练集的行号边界（在原始df中）
            # 训练集输入窗口最后一行 < split_row + INPUT_WINDOW
            max_train_input_end_row = split_row_in_df + self.input_window
            # 训练集目标行 < split_row + TRAIN_EXTENSION_DAYS
            max_train_target_row = split_row_in_df + self.train_extension_days
            
            # 4. 计算训练集可以使用的数据范围（在原始df中的行号）
            # 目标行需要的最大数据行 = max_train_target_row - 1
            # 但要确保不超过df的长度
            max_train_data_row = min(max_train_target_row, len(df))
            
            # 5. 划分训练集和验证集（基于行号）
            # 训练集：从original_start_idx到max_train_data_row的数据
            train_end_in_filtered = min(max_train_data_row - original_start_idx, len(filtered_df))
            train_df = filtered_df.iloc[:train_end_in_filtered].copy()
            
            # 验证集：从SPLIT_DATE开始
            val_df = filtered_df.iloc[split_pos_in_filtered:].copy()
            
            train_days = len(train_df)
            val_days = len(val_df)
            
            # 检查划分日期前后的数据量（v0.4修改：只要任意一边达到要求即可）
            if train_days < self.min_required_days and val_days < self.min_required_days:
                self.log(
                    f'跳过 {company_name}({stock_code}): 训练集数据范围(可用行数)有 {train_days} 天数据，'
                    f'验证集数据范围(从{SPLIT_DATE}开始)有 {val_days} 天数据，'
                    f'需要至少一边达到 {self.min_required_days} 天 (输入{self.input_window} + 预测第{self.prediction_day}天)',
                    'WARN'
                )
                self.stats['skipped_companies'] += 1
                self.stats['skipped_details'].append({
                    'company': company_name,
                    'code': stock_code,
                    'reason': f'训练集和验证集数据都不足: 训练集{train_days}天 < {self.min_required_days}天 且 验证集{val_days}天 < {self.min_required_days}天'
                })
                return [], global_sample_id
            
            # v0.5修改：分别对训练集和验证集使用不同的步长生成样本
            # v0.7修改：训练集生成时传递行号边界
            all_samples = []
            
            # 生成训练集样本（传递split_row和边界行号）
            train_samples, global_sample_id = self._generate_samples_for_split(
                train_df, filtered_df, original_start_idx, df, company_id, company_name, 
                stock_code, file_path, self.train_stride, 'train', global_sample_id,
                split_row=split_row_in_df, 
                max_input_end_row=max_train_input_end_row,
                max_target_row=max_train_target_row
            )
            all_samples.extend(train_samples)
            
            # 生成验证集样本（不需要边界限制）
            val_samples, global_sample_id = self._generate_samples_for_split(
                val_df, filtered_df, original_start_idx, df, company_id, company_name, 
                stock_code, file_path, self.val_stride, 'val', global_sample_id,
                split_row=split_row_in_df
            )
            all_samples.extend(val_samples)
            
            num_samples = len(all_samples)
            train_count = len(train_samples)
            val_count = len(val_samples)
            
            # 计算理论样本数（用于统计）
            train_max_samples = (train_days - self.min_required_days + 1 + self.train_stride - 1) // self.train_stride if train_days >= self.min_required_days else 0
            val_max_samples = (val_days - self.min_required_days + 1 + self.val_stride - 1) // self.val_stride if val_days >= self.min_required_days else 0
            total_max_samples = train_max_samples + val_max_samples
            invalid_count = total_max_samples - num_samples
            
            if invalid_count > 0:
                self.log(
                    f'处理 {company_name}({stock_code}): 时间范围内有 {len(filtered_df)} 天数据 '
                    f'(训练集范围:{train_days}天, 验证集范围:{val_days}天)，'
                    f'理论可生成 {total_max_samples} 个样本（训练集stride={self.train_stride}, 验证集stride={self.val_stride}），'
                    f'实际生成 {num_samples} 个有效样本 (训练集:{train_count}, 验证集:{val_count})，'
                    f'验证失败 {invalid_count} 个',
                    'WARN'
                )
            else:
                self.log(
                    f'处理 {company_name}({stock_code}): 时间范围内有 {len(filtered_df)} 天数据 '
                    f'(训练集范围:{train_days}天, 验证集范围:{val_days}天)，'
                    f'生成 {num_samples} 个样本 (训练集:{train_count}, 验证集:{val_count})'
                )
            
            self.stats['valid_companies'] += 1
            self.stats['total_samples'] += num_samples
            self.stats['train_samples'] += train_count
            self.stats['val_samples'] += val_count
            
            return all_samples, global_sample_id
            
        except Exception as e:
            self.log(f'处理 {company_name}({stock_code}) 时出错: {str(e)}', 'ERROR')
            self.stats['skipped_companies'] += 1
            self.stats['skipped_details'].append({
                'company': company_name,
                'code': stock_code,
                'reason': f'处理错误: {str(e)}'
            })
            return [], global_sample_id
    
    def generate(self):
        """主生成流程"""
        self.log('=' * 80)
        self.log('开始生成滚动窗口数据集索引 (v0.7)')
        self.log('=' * 80)
        self.log(f'配置参数:')
        self.log(f'  数据源目录: {self.source_dir}')
        self.log(f'  输出目录: {self.output_dir}')
        self.log(f'  时间范围: {START_DATE} 至 {END_DATE}')
        self.log(f'  输入窗口: {self.input_window} 天')
        self.log(f'  预测目标: 未来第 {self.prediction_day} 天')
        self.log(f'  训练集滚动步长: {self.train_stride} 天')
        self.log(f'  验证集滚动步长: {self.val_stride} 天')
        self.log(f'  最小数据要求: {self.min_required_days} 天 ({self.input_window} + {self.prediction_day})')
        self.log(f'  训练/验证集划分点: {SPLIT_DATE}')
        self.log(f'  v0.7特性：训练集基于交易日行号扩展，使训练集更接近验证集')
        self.log(f'  - 以SPLIT_DATE在数据中的行号为基准(split_row)')
        self.log(f'  - 训练集输入窗口最后一行 < split_row + {INPUT_WINDOW}个交易日')
        self.log(f'  - 训练集目标行 < split_row + {self.train_extension_days}个交易日')
        self.log(f'  - 验证集从SPLIT_DATE开始(行号 >= split_row)')
        self.log(f'  数据文件: 不拷贝，直接引用原始文件')
        self.log(f'  数据检查: 只要训练集或验证集中任意一边达到 {self.min_required_days} 天数据即可')
        self.log('=' * 80)
        
        # 扫描文件
        company_files = self.scan_company_files()
        
        if not company_files:
            self.log('未找到任何公司文件', 'ERROR')
            return
        
        # 处理所有公司
        all_samples = []
        global_sample_id = 0
        
        for file_path in tqdm(company_files, desc='处理公司数据'):
            samples, global_sample_id = self.process_company(file_path, global_sample_id)
            all_samples.extend(samples)
        
        # 转换为DataFrame
        if not all_samples:
            self.log('没有生成任何样本', 'ERROR')
            self._flush_log()
            return
        
        samples_df = pd.DataFrame(all_samples)
        
        # 分离训练集和验证集
        train_df = samples_df[samples_df['split'] == 'train'].drop(columns=['split'])
        val_df = samples_df[samples_df['split'] == 'val'].drop(columns=['split'])
        
        # 保存索引文件
        self.log('=' * 80)
        self.log('保存索引文件...')
        train_df.to_parquet(self.output_dir / 'train_samples_index.parquet', index=False)
        self.log(f'训练集索引已保存: train_samples_index.parquet ({len(train_df)} 个样本)')
        
        val_df.to_parquet(self.output_dir / 'val_samples_index.parquet', index=False)
        self.log(f'验证集索引已保存: val_samples_index.parquet ({len(val_df)} 个样本)')
        
        # 保存元数据
        self.save_metadata(samples_df)
        
        # 最终统计
        self.log('=' * 80)
        self.log('处理完成！')
        self.log('=' * 80)
        self.log(f'总公司文件数: {self.stats["total_files"]}')
        self.log(f'有效公司数: {self.stats["valid_companies"]}')
        self.log(f'跳过公司数: {self.stats["skipped_companies"]}')
        self.log(f'总样本数: {self.stats["total_samples"]:,}')
        self.log(f'训练集样本数: {self.stats["train_samples"]:,}')
        self.log(f'验证集样本数: {self.stats["val_samples"]:,}')
        
        # 显示验证统计
        self.log('=' * 80)
        self.log('异常检测统计:')
        validation_errors = self.stats['validation_errors']
        self.log(f'  行号超出范围错误: {validation_errors["row_out_of_range"]}')
        self.log(f'  目标行号超出范围错误: {validation_errors["target_row_out_of_range"]}')
        self.log(f'  行号差异不匹配错误: {validation_errors["row_diff_mismatch"]}')
        self.log(f'  训练集目标日期超限错误: {validation_errors["target_date_overflow"]}')
        self.log(f'  无效样本总数: {validation_errors["total_invalid_samples"]}')
        self.log(f'  验证警告总数: {len(self.stats["validation_warnings"])}')
        
        # 显示行号差异分布（交易日数量）
        self.log('=' * 80)
        self.log('行号差异分布统计（交易日数量）:')
        if self.stats['row_diff_distribution']:
            sorted_row_diffs = sorted(
                self.stats['row_diff_distribution'].items(),
                key=lambda x: int(x[0])
            )
            for row_diff, count in sorted_row_diffs[:20]:  # 显示前20个
                percentage = count / self.stats['total_samples'] * 100
                marker = '✓' if int(row_diff) == self.prediction_day else '✗'
                self.log(f'  {marker} 行号差异={row_diff}: {count:,} 个样本 ({percentage:.2f}%)')
            if len(sorted_row_diffs) > 20:
                self.log(f'  ... 还有 {len(sorted_row_diffs) - 20} 种差异未显示')
        else:
            self.log('  无统计数据')
        
        # 显示前10个验证警告
        if self.stats['validation_warnings']:
            self.log('=' * 80)
            self.log('前10个验证警告/错误:')
            for i, warning in enumerate(self.stats['validation_warnings'][:10], 1):
                self.log(f'  {i}. {warning}')
            if len(self.stats['validation_warnings']) > 10:
                self.log(f'  ... 还有 {len(self.stats["validation_warnings"]) - 10} 个警告未显示')
        
        self.log('=' * 80)
        self.log(f'输出目录: {self.output_dir}')
        self.log('=' * 80)
        
        # 刷新日志
        self._flush_log()
    
    def save_metadata(self, samples_df):
        """保存元数据"""
        # 读取一个示例文件获取列信息
        sample_file_path = samples_df.iloc[0]['source_file']
        all_columns = []
        if Path(sample_file_path).exists():
            sample_df = pd.read_parquet(sample_file_path)
            all_columns = sample_df.columns.tolist()
        
        metadata = {
            'script_name': 'roll_generate_index_v0.7_20260111154500.py',
            'version': 'v0.7',
            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'config': {
                'source_data_dir': str(self.source_dir),
                'start_date': START_DATE,
                'end_date': END_DATE,
                'input_window': self.input_window,
                'prediction_day': self.prediction_day,
                'train_stride': self.train_stride,
                'val_stride': self.val_stride,
                'split_date': SPLIT_DATE,
                'train_extension_days': self.train_extension_days,
                'train_input_end_offset': self.input_window,
                'train_target_offset': self.train_extension_days,
                'min_required_days': self.min_required_days,
                'copy_files': False
            },
            'statistics': {
                'total_company_files': self.stats['total_files'],
                'valid_companies': self.stats['valid_companies'],
                'skipped_companies': self.stats['skipped_companies'],
                'total_samples': self.stats['total_samples'],
                'train_samples': self.stats['train_samples'],
                'val_samples': self.stats['val_samples']
            },
            'validation_statistics': {
                'errors': self.stats['validation_errors'],
                'warnings_count': len(self.stats['validation_warnings']),
                'warnings_sample': self.stats['validation_warnings'][:20],  # 保存前20个警告作为示例
                'row_diff_distribution': self.stats['row_diff_distribution']
            },
            'data_info': {
                'feature_columns': len(all_columns),
                'date_column': '日期',
                'all_columns': all_columns,
                'target_type': 'single_day',
                'target_description': f'预测未来第{self.prediction_day}天的数据'
            },
            'index_structure': {
                'description': '索引文件包含原始数据文件路径，不拷贝文件',
                'columns': {
                    'sample_id': '样本唯一ID',
                    'company_id': '公司编号',
                    'company_name': '公司名称',
                    'stock_code': '股票代码',
                    'start_date': '输入窗口开始日期',
                    'input_end_date': '输入窗口结束日期',
                    'target_date': '预测目标日期',
                    'input_row_start': '输入数据在原始文件中的起始行',
                    'input_row_end': '输入数据在原始文件中的结束行',
                    'target_row': '目标数据在原始文件中的行号',
                    'prediction_day': '预测未来第几天',
                    'source_file': '原始数据文件路径'
                }
            },
            'skipped_companies_details': self.stats['skipped_details'],
            'validation_logic': {
                'description': 'v0.7修改：训练集基于交易日行号扩展，使训练集样本能接近验证集时间段，但确保训练集和验证集目标日期不重叠。',
                'v0.7_changes': [
                    f'基于SPLIT_DATE在每个公司数据中的行号位置(split_row)进行划分',
                    f'训练集输入窗口最后一行 < split_row + {INPUT_WINDOW}个交易日',
                    f'训练集目标行 < split_row + {self.train_extension_days}个交易日',
                    f'验证集输入窗口起始行 >= split_row',
                    f'验证集最早目标行约为：split_row + INPUT_WINDOW + PREDICTION_DAY',
                    f'确保：训练集最晚目标行 < 验证集最早目标行',
                    f'目的：让训练集使用更接近验证集时间段的数据，减少时间差异',
                    f'注意：使用交易日（行号）而不是日历日，每个公司独立计算'
                ],
                'checks': [
                    f'总数据天数 >= {self.min_required_days}天',
                    f'训练集数据范围有数据天数 >= {self.min_required_days}天（用于训练集，步长={self.train_stride}）',
                    f'  OR 验证集数据范围有数据天数 >= {self.min_required_days}天（用于验证集，步长={self.val_stride}）',
                    f'只要训练集或验证集中任意一边达到要求即可',
                    f'训练集和验证集分别使用不同的滚动步长进行采样',
                    f'训练集样本的输入窗口最后一行必须 < split_row + {INPUT_WINDOW}',
                    f'训练集样本的目标行必须 < split_row + {self.train_extension_days}'
                ]
            }
        }
        
        metadata_path = self.output_dir / 'metadata.json'
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        self.log(f'元数据已保存: metadata.json')


def main():
    """主函数"""
    try:
        generator = RollingDatasetGenerator()
        generator.generate()
        print(f'\n✓ 所有文件已保存到: {generator.output_dir}')
        
    except KeyboardInterrupt:
        print('\n用户中断执行')
        sys.exit(1)
    except Exception as e:
        print(f'\n执行出错: {str(e)}')
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
