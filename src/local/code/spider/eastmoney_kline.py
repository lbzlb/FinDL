"""东方财富数据源模块 v0.7 - K线数据专用版（增强请求头 + 详细错误日志）

v0.7 新增功能：
1. ✅ 扩充User-Agent池：5个→30个，覆盖更多浏览器和操作系统
2. ✅ 美股智能缓存：记住成功的市场代码组合，避免重复尝试
3. ✅ 增强限流检测：识别429、403等限流状态码
4. ✅ 详细错误日志：记录所有失败原因、请求参数、响应详情，便于后期分析
5. ✅ 股票代码格式：与 v0.6 保持一致（4种格式×3个市场=12次尝试）

v0.7 修复（重要）：
- ✅ 简化请求头：只保留 User-Agent 和 Referer 两个字段（与 v0.6 一致）
  原因：过多请求头字段（Accept、Accept-Language、Connection等）反而被识别为爬虫
- ✅ 禁用Session连接复用：每次独立请求，避免被服务器识别
- ✅ 美股尝试策略：从6次恢复到12次，提高兼容性

相比v0.6的改进：
- 30个User-Agent池（vs v0.6的5个），更好的随机性
- 详细的错误分类和诊断信息，便于排查问题
- 智能缓存机制，减少无效请求
- 更丰富的股票代码格式支持

提供A股、港股、美股的K线数据获取功能。
"""

import random
import requests
import pandas as pd
from datetime import datetime, timedelta
import json


# ==================== 全局配置 ====================

# 全局Session对象（连接复用）
_session = None

# 美股成功市场代码缓存 {股票代码: (市场代码, 代码格式)}
_us_market_cache = {}

# Session模式开关：False=每次新建连接（更隐蔽，但速度慢），True=复用连接（快但可能被识别）
USE_SESSION = False


def _get_session():
    """获取或创建全局Session对象"""
    global _session
    if _session is None:
        _session = requests.Session()
        # 设置连接池大小
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=10,
            pool_maxsize=20,
            max_retries=0  # 不自动重试，由上层控制
        )
        _session.mount('http://', adapter)
        _session.mount('https://', adapter)
    return _session


# ==================== 动态请求头生成 ====================

def _get_random_headers():
    """
    生成随机化的HTTP请求头
    
    v0.7 策略（修复后）：
    - User-Agent池：30个（覆盖Chrome、Firefox、Safari、Edge）
    - 请求头字段：仅2个（User-Agent、Referer）- 简单自然，避免过度伪装
    
    Returns:
        dict: HTTP请求头字典
    """
    # User-Agent池：30个常见浏览器（Chrome、Firefox、Safari、Edge）
    user_agents = [
        # Chrome - Windows
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36',
        
        # Chrome - Mac
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 13_6_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        
        # Chrome - Linux
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
        
        # Firefox - Windows
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:122.0) Gecko/20100101 Firefox/122.0',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:119.0) Gecko/20100101 Firefox/119.0',
        
        # Firefox - Mac
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:121.0) Gecko/20100101 Firefox/121.0',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:120.0) Gecko/20100101 Firefox/120.0',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 13.6; rv:121.0) Gecko/20100101 Firefox/121.0',
        
        # Safari - Mac
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.6 Safari/605.1.15',
        
        # Edge - Windows
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36 Edg/121.0.0.0',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0',
        
        # Edge - Mac
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36 Edg/121.0.0.0',
        
        # 其他版本混合
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36',
        'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:121.0) Gecko/20100101 Firefox/121.0',
    ]
    
    # Referer池：东方财富常见页面
    referers = [
        'https://quote.eastmoney.com/',
        'https://data.eastmoney.com/',
        'https://www.eastmoney.com/',
        'https://quote.eastmoney.com/center/',
        'https://data.eastmoney.com/stockdata/',
    ]
    
    # v0.7 修复：简化请求头，只保留2个关键字段（与 v0.6 一致）
    # 过多的请求头字段反而容易被识别为爬虫
    headers = {
        'User-Agent': random.choice(user_agents),
        'Referer': random.choice(referers),
    }
    
    return headers


# ==================== 错误诊断和日志 ====================

class RequestError:
    """请求错误详情记录类"""
    
    def __init__(self, error_type, message, details=None):
        """
        初始化错误记录
        
        Args:
            error_type: 错误类型（如 'RATE_LIMIT', 'NETWORK', 'PARSE', 'EMPTY_DATA'）
            message: 错误描述信息
            details: 详细信息字典
        """
        self.timestamp = datetime.now().strftime("%H:%M:%S")  # 只保留时分秒，更紧凑
        self.error_type = error_type
        self.message = message
        self.details = details or {}
    
    def to_dict(self):
        """转换为字典格式"""
        return {
            'timestamp': self.timestamp,
            'error_type': self.error_type,
            'message': self.message,
            'details': self.details
        }
    
    def to_log_string(self):
        """
        转换为紧凑的两行日志字符串
        
        格式：
        ❌ [时间] 市场-股票 | 错误类型 | 关键参数
           异常详情
        """
        # 第一行：主要信息
        market = self.details.get('market', '')
        stock_code = self.details.get('stock_code', '')
        secid = self.details.get('secid', '')
        status_code = self.details.get('status_code', '')
        
        # 构建标识：市场-股票
        identifier = f"{market}-{stock_code}" if market and stock_code else stock_code or "未知"
        
        # 构建关键参数
        key_params = []
        if secid:
            key_params.append(f"secid={secid}")
        if status_code:
            key_params.append(f"HTTP {status_code}")
        
        params_str = " | ".join(key_params) if key_params else ""
        
        first_line = f"❌ [{self.timestamp}] {identifier} | {self.error_type}"
        if params_str:
            first_line += f" | {params_str}"
        
        # 第二行：异常详情或响应详情
        second_line_parts = []
        
        # 优先显示异常信息
        exc_type = self.details.get('exception_type')
        exc_msg = self.details.get('exception_message')
        if exc_type and exc_msg:
            # 截断过长的异常消息
            if len(exc_msg) > 150:
                exc_msg = exc_msg[:150] + "..."
            second_line_parts.append(f"{exc_type}: {exc_msg}")
        
        # 或显示HTTP响应预览
        elif 'response_text_preview' in self.details:
            preview = self.details['response_text_preview']
            if preview:
                if len(preview) > 150:
                    preview = preview[:150] + "..."
                second_line_parts.append(f"Response: {preview}")
        
        # 或显示响应头（如Retry-After）
        elif 'response_headers' in self.details:
            headers = self.details['response_headers']
            if isinstance(headers, dict):
                retry_after = headers.get('Retry-After') or headers.get('retry-after')
                if retry_after:
                    second_line_parts.append(f"Retry-After: {retry_after}s")
        
        # 或显示基本描述
        if not second_line_parts:
            second_line_parts.append(self.message)
        
        # 添加其他关键信息
        period = self.details.get('period')
        date_range = self.details.get('date_range')
        if period:
            second_line_parts.append(f"period={period}")
        if date_range:
            second_line_parts.append(f"range={date_range}")
        elif 'start_date' in self.details and 'end_date' in self.details:
            second_line_parts.append(f"range={self.details['start_date']}→{self.details['end_date']}")
        
        second_line = "   " + " | ".join(second_line_parts)
        
        return f"{first_line}\n{second_line}"


def _diagnose_error(response=None, exception=None, context=None):
    """
    诊断错误类型并生成详细错误信息
    
    Args:
        response: requests.Response对象（如果有）
        exception: 异常对象（如果有）
        context: 上下文信息字典（股票代码、市场、周期等）
    
    Returns:
        RequestError: 错误详情对象
    """
    context = context or {}
    
    # HTTP响应错误
    if response is not None:
        status_code = response.status_code
        
        # 限流错误
        if status_code == 429:
            return RequestError(
                error_type='RATE_LIMIT_429',
                message='服务器返回429 Too Many Requests，请求过于频繁',
                details={
                    'status_code': status_code,
                    'url': response.url,
                    'stock_code': context.get('stock_code'),
                    'market': context.get('market'),
                    'period': context.get('period'),
                    'secid': context.get('secid'),
                    'response_headers': dict(response.headers)
                }
            )
        
        # 访问被拒绝
        elif status_code == 403:
            return RequestError(
                error_type='ACCESS_FORBIDDEN_403',
                message='服务器返回403 Forbidden，访问被拒绝',
                details={
                    'status_code': status_code,
                    'url': response.url,
                    'stock_code': context.get('stock_code'),
                    'market': context.get('market'),
                    'secid': context.get('secid'),
                    'response_text_preview': response.text[:200] if response.text else ''
                }
            )
        
        # 服务器错误
        elif 500 <= status_code < 600:
            return RequestError(
                error_type='SERVER_ERROR',
                message=f'服务器错误 {status_code}',
                details={
                    'status_code': status_code,
                    'url': response.url,
                    'stock_code': context.get('stock_code'),
                    'market': context.get('market'),
                    'secid': context.get('secid'),
                    'response_text_preview': response.text[:200] if response.text else ''
                }
            )
        
        # 其他HTTP错误
        elif status_code >= 400:
            return RequestError(
                error_type='HTTP_ERROR',
                message=f'HTTP错误 {status_code}',
                details={
                    'status_code': status_code,
                    'url': response.url,
                    'stock_code': context.get('stock_code'),
                    'market': context.get('market'),
                    'secid': context.get('secid'),
                    'response_text_preview': response.text[:200] if response.text else ''
                }
            )
    
    # 异常错误
    if exception is not None:
        exc_type = type(exception).__name__
        exc_message = str(exception)
        
        # 超时错误
        if isinstance(exception, requests.Timeout):
            return RequestError(
                error_type='TIMEOUT',
                message='请求超时',
                details={
                    'exception_type': exc_type,
                    'exception_message': exc_message,
                    'stock_code': context.get('stock_code'),
                    'market': context.get('market'),
                    'secid': context.get('secid'),
                    'timeout_seconds': context.get('timeout', 30)
                }
            )
        
        # 连接错误
        elif isinstance(exception, requests.ConnectionError):
            # 检查是否是远程断开连接
            if 'RemoteDisconnected' in exc_message or 'Connection aborted' in exc_message:
                return RequestError(
                    error_type='REMOTE_DISCONNECTED',
                    message='服务器主动断开连接',
                    details={
                        'exception_type': exc_type,
                        'exception_message': exc_message,
                        'stock_code': context.get('stock_code'),
                        'market': context.get('market'),
                        'secid': context.get('secid'),
                        'period': context.get('period'),
                        'start_date': context.get('start_date'),
                        'end_date': context.get('end_date')
                    }
                )
            else:
                return RequestError(
                    error_type='CONNECTION_ERROR',
                    message='网络连接失败',
                    details={
                        'exception_type': exc_type,
                        'exception_message': exc_message,
                        'stock_code': context.get('stock_code'),
                        'market': context.get('market'),
                        'secid': context.get('secid')
                    }
                )
        
        # JSON解析错误
        elif isinstance(exception, (ValueError, json.JSONDecodeError)):
            return RequestError(
                error_type='JSON_PARSE_ERROR',
                message='响应内容不是有效的JSON格式',
                details={
                    'exception_type': exc_type,
                    'exception_message': exc_message,
                    'stock_code': context.get('stock_code'),
                    'market': context.get('market'),
                    'secid': context.get('secid'),
                    'response_text_preview': context.get('response_text', '')[:200]
                }
            )
        
        # 其他异常
        else:
            return RequestError(
                error_type='UNKNOWN_EXCEPTION',
                message=f'未知异常：{exc_type}',
                details={
                    'exception_type': exc_type,
                    'exception_message': exc_message,
                    'stock_code': context.get('stock_code'),
                    'market': context.get('market'),
                    'secid': context.get('secid')
                }
            )
    
    # 默认未知错误
    return RequestError(
        error_type='UNKNOWN',
        message='未知错误',
        details=context
    )


def _print_error_log(error, prefix=""):
    """
    打印格式化的错误日志
    
    Args:
        error: RequestError对象
        prefix: 日志前缀（如空格缩进）
    """
    log_lines = error.to_log_string().split('\n')
    for line in log_lines:
        print(f"{prefix}{line}")


# ==================== 东方财富K线API直接调用 ====================

def _get_a_historical_data(symbol, period, start_date, end_date):
    """
    获取A股历史K线数据（直接调用东方财富API）
    
    v0.7 改进：
    - 简化请求头（仅 User-Agent + Referer）
    - 详细的错误日志记录
    
    Args:
        symbol (str): A股代码，如 '000001', '600519'
        period (str): 周期类型，'daily'=日K, 'weekly'=周K
        start_date (str): 开始日期，格式 'YYYYMMDD'
        end_date (str): 结束日期，格式 'YYYYMMDD'
    
    Returns:
        pd.DataFrame: K线数据
    """
    # 判断市场代码
    prefix = symbol[:2]
    if prefix in ['60', '68']:
        secid = f'1.{symbol}'  # 上海
    elif prefix in ['00', '30']:
        secid = f'0.{symbol}'  # 深圳
    else:
        secid = f'1.{symbol}'  # 默认上海
    
    # 周期映射
    klt_map = {
        'daily': 101,   # 日K
        'weekly': 102,  # 周K
    }
    klt = klt_map.get(period, 101)
    
    # 构建API URL
    url = 'https://push2his.eastmoney.com/api/qt/stock/kline/get'
    
    # 请求参数
    params = {
        'secid': secid,
        'ut': 'fa5fd1943c7b386f172d6893dbfba10b',
        'fields1': 'f1,f2,f3,f4,f5,f6',
        'fields2': 'f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61',
        'klt': klt,
        'fqt': 0,  # 0=不复权
        'beg': start_date,
        'end': end_date,
        '_': str(int(datetime.now().timestamp() * 1000)),
    }
    
    # 上下文信息（用于错误诊断）
    context = {
        'stock_code': symbol,
        'market': 'A股',
        'period': period,
        'secid': secid,
        'start_date': start_date,
        'end_date': end_date,
    }
    
    try:
        # 使用动态请求头（v0.7修复：禁用Session以避免被识别）
        headers = _get_random_headers()
        
        print(f"    🔍 请求: secid={secid}, period={period}, {start_date}→{end_date}")
        
        # 根据配置选择使用 Session 或直接请求
        if USE_SESSION:
            session = _get_session()
            response = session.get(url, params=params, headers=headers, timeout=30)
        else:
            response = requests.get(url, params=params, headers=headers, timeout=30)
        
        # 检查HTTP状态码
        if response.status_code != 200:
            error = _diagnose_error(response=response, context=context)
            _print_error_log(error, prefix="    ")
            return pd.DataFrame()
        
        # 检查返回内容
        if not response.text:
            error = RequestError(
                error_type='EMPTY_RESPONSE',
                message='API返回空内容',
                details={
                    'stock_code': symbol,
                    'secid': secid,
                    'url': url,
                }
            )
            _print_error_log(error, prefix="    ")
            return pd.DataFrame()
        
        # 尝试解析JSON
        try:
            data = response.json()
        except (ValueError, json.JSONDecodeError) as json_err:
            context['response_text'] = response.text
            error = _diagnose_error(exception=json_err, context=context)
            _print_error_log(error, prefix="    ")
            return pd.DataFrame()
        
        # 解析数据
        if 'data' not in data or not data['data']:
            error = RequestError(
                error_type='NO_DATA_FIELD',
                message='响应JSON中缺少data字段',
                details={
                    'stock_code': symbol,
                    'secid': secid,
                    'response_keys': list(data.keys()),
                }
            )
            _print_error_log(error, prefix="    ")
            return pd.DataFrame()
        
        klines = data['data'].get('klines', [])
        if not klines:
            error = RequestError(
                error_type='EMPTY_KLINES',
                message='返回的K线数据为空',
                details={
                    'stock_code': symbol,
                    'secid': secid,
                    'period': period,
                    'date_range': f'{start_date}→{end_date}',
                }
            )
            _print_error_log(error, prefix="    ")
            return pd.DataFrame()
        
        # 解析K线数据
        records = []
        parse_errors = 0
        for i, kline in enumerate(klines):
            parts = kline.split(',')
            if len(parts) >= 11:
                try:
                    records.append({
                        '日期': parts[0],
                        '开盘': float(parts[1]),
                        '收盘': float(parts[2]),
                        '最高': float(parts[3]),
                        '最低': float(parts[4]),
                        '成交量': float(parts[5]),
                        '成交额': float(parts[6]),
                        '振幅': float(parts[7]),
                        '涨跌幅': float(parts[8]),
                        '涨跌额': float(parts[9]),
                        '换手率': float(parts[10]),
                    })
                except (ValueError, IndexError) as parse_err:
                    parse_errors += 1
            else:
                parse_errors += 1
        
        if parse_errors > 0:
            print(f"    ⚠️  解析警告: {parse_errors}/{len(klines)} 条K线数据格式异常，已跳过")
        
        if not records:
            error = RequestError(
                error_type='PARSE_ALL_FAILED',
                message='所有K线数据解析失败',
                details={
                    'stock_code': symbol,
                    'total_klines': len(klines),
                    'sample_kline': klines[0] if klines else '',
                }
            )
            _print_error_log(error, prefix="    ")
            return pd.DataFrame()
        
        print(f"    ✅ 成功获取 {len(records)} 条K线数据")
        return pd.DataFrame(records)
        
    except requests.RequestException as req_err:
        error = _diagnose_error(exception=req_err, context=context)
        _print_error_log(error, prefix="    ")
        return pd.DataFrame()
    except Exception as exc:
        error = _diagnose_error(exception=exc, context=context)
        _print_error_log(error, prefix="    ")
        return pd.DataFrame()


def _get_hk_historical_data(symbol, period, start_date, end_date):
    """
    获取港股历史K线数据（直接调用东方财富API）
    
    v0.7 改进：
    - 简化请求头（仅 User-Agent + Referer）
    - 详细的错误日志记录
    
    Args:
        symbol (str): 港股代码，如 '00700'
        period (str): 周期类型，'daily'=日K, 'weekly'=周K
        start_date (str): 开始日期，格式 'YYYYMMDD'
        end_date (str): 结束日期，格式 'YYYYMMDD'
    
    Returns:
        pd.DataFrame: K线数据
    """
    # 港股代码格式
    secid = f'116.{symbol}'
    
    # 周期映射
    klt_map = {
        'daily': 101,   # 日K
        'weekly': 102,  # 周K
    }
    klt = klt_map.get(period, 101)
    
    # 构建API URL
    url = 'https://push2his.eastmoney.com/api/qt/stock/kline/get'
    
    # 请求参数
    params = {
        'secid': secid,
        'ut': 'fa5fd1943c7b386f172d6893dbfba10b',
        'fields1': 'f1,f2,f3,f4,f5,f6',
        'fields2': 'f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61',
        'klt': klt,
        'fqt': 0,  # 0=不复权
        'beg': start_date,
        'end': end_date,
        '_': str(int(datetime.now().timestamp() * 1000)),
    }
    
    # 上下文信息
    context = {
        'stock_code': symbol,
        'market': '港股',
        'period': period,
        'secid': secid,
        'start_date': start_date,
        'end_date': end_date,
    }
    
    try:
        # 使用动态请求头（v0.7修复：禁用Session以避免被识别）
        headers = _get_random_headers()
        
        print(f"    🔍 请求: secid={secid}, period={period}, {start_date}→{end_date}")
        
        # 根据配置选择使用 Session 或直接请求
        if USE_SESSION:
            session = _get_session()
            response = session.get(url, params=params, headers=headers, timeout=30)
        else:
            response = requests.get(url, params=params, headers=headers, timeout=30)
        
        # 检查HTTP状态码
        if response.status_code != 200:
            error = _diagnose_error(response=response, context=context)
            _print_error_log(error, prefix="    ")
            return pd.DataFrame()
        
        # 检查返回内容
        if not response.text:
            error = RequestError(
                error_type='EMPTY_RESPONSE',
                message='API返回空内容',
                details={
                    'stock_code': symbol,
                    'secid': secid,
                    'url': url,
                }
            )
            _print_error_log(error, prefix="    ")
            return pd.DataFrame()
        
        # 尝试解析JSON
        try:
            data = response.json()
        except (ValueError, json.JSONDecodeError) as json_err:
            context['response_text'] = response.text
            error = _diagnose_error(exception=json_err, context=context)
            _print_error_log(error, prefix="    ")
            return pd.DataFrame()
        
        # 解析数据
        if 'data' not in data or not data['data']:
            error = RequestError(
                error_type='NO_DATA_FIELD',
                message='响应JSON中缺少data字段',
                details={
                    'stock_code': symbol,
                    'secid': secid,
                    'response_keys': list(data.keys()),
                }
            )
            _print_error_log(error, prefix="    ")
            return pd.DataFrame()
        
        klines = data['data'].get('klines', [])
        if not klines:
            error = RequestError(
                error_type='EMPTY_KLINES',
                message='返回的K线数据为空',
                details={
                    'stock_code': symbol,
                    'secid': secid,
                    'period': period,
                    'date_range': f'{start_date}→{end_date}',
                }
            )
            _print_error_log(error, prefix="    ")
            return pd.DataFrame()
        
        # 解析K线数据
        records = []
        parse_errors = 0
        for kline in klines:
            parts = kline.split(',')
            if len(parts) >= 11:
                try:
                    records.append({
                        '日期': parts[0],
                        '开盘': float(parts[1]),
                        '收盘': float(parts[2]),
                        '最高': float(parts[3]),
                        '最低': float(parts[4]),
                        '成交量': float(parts[5]),
                        '成交额': float(parts[6]),
                        '振幅': float(parts[7]),
                        '涨跌幅': float(parts[8]),
                        '涨跌额': float(parts[9]),
                        '换手率': float(parts[10]),
                    })
                except (ValueError, IndexError):
                    parse_errors += 1
            else:
                parse_errors += 1
        
        if parse_errors > 0:
            print(f"    ⚠️  解析警告: {parse_errors}/{len(klines)} 条K线数据格式异常，已跳过")
        
        if not records:
            error = RequestError(
                error_type='PARSE_ALL_FAILED',
                message='所有K线数据解析失败',
                details={
                    'stock_code': symbol,
                    'total_klines': len(klines),
                    'sample_kline': klines[0] if klines else '',
                }
            )
            _print_error_log(error, prefix="    ")
            return pd.DataFrame()
        
        print(f"    ✅ 成功获取 {len(records)} 条K线数据")
        return pd.DataFrame(records)
        
    except requests.RequestException as req_err:
        error = _diagnose_error(exception=req_err, context=context)
        _print_error_log(error, prefix="    ")
        return pd.DataFrame()
    except Exception as exc:
        error = _diagnose_error(exception=exc, context=context)
        _print_error_log(error, prefix="    ")
        return pd.DataFrame()


def _get_us_historical_data(symbol, period, start_date, end_date):
    """
    获取美股历史K线数据（直接调用东方财富API，支持智能缓存）
    
    v0.7 改进：
    - 智能缓存：记住成功的市场代码组合，避免重复尝试
    - 尝试策略：4种格式×3个市场=12次（与 v0.6 一致）
    - 简化请求头（仅 User-Agent + Referer）
    - 详细的错误日志记录
    
    Args:
        symbol (str): 美股代码，如 'AAPL', 'TSLA', 'BRK.B'
        period (str): 周期类型，'daily'=日K, 'weekly'=周K
        start_date (str): 开始日期，格式 'YYYYMMDD'
        end_date (str): 结束日期，格式 'YYYYMMDD'
    
    Returns:
        pd.DataFrame: K线数据
    """
    global _us_market_cache
    
    # 检查缓存：如果之前成功过，直接使用缓存的市场代码
    cache_key = symbol
    if cache_key in _us_market_cache:
        cached_market, cached_format = _us_market_cache[cache_key]
        print(f"    💾 使用缓存: {symbol} → {cached_format} (市场代码: {cached_market})")
        secid = f'{cached_market}.{cached_format}'
        result_df = _try_us_request(
            secid=secid,
            symbol=symbol,
            period=period,
            start_date=start_date,
            end_date=end_date,
            is_cached=True
        )
        if result_df is not None and not result_df.empty:
            return result_df
        else:
            # 缓存失效（可能退市或代码变更），清除缓存
            print(f"    ⚠️  缓存失效，清除缓存并重新尝试")
            del _us_market_cache[cache_key]
    
    # 美股市场代码列表（按优先级）
    # 106: 纽约证券交易所 (NYSE) - 大公司优先
    # 105: 纳斯达克 (NASDAQ) - 科技股优先
    # 107: 美国其他交易所
    market_codes = [106, 105, 107]
    
    # 生成股票代码格式（与 v0.6 保持一致，提高兼容性）
    symbol_formats = [
        symbol,                      # 原始格式（优先）
        symbol.replace('.', '_'),    # BRK.B → BRK_B（东方财富常用）
        symbol.replace('.', '-'),    # BRK.B → BRK-B
        symbol.replace('.', ''),     # BRK.B → BRKB（去掉点号）
    ]
    # 去重
    symbol_formats = list(dict.fromkeys(symbol_formats))
    
    # 最大尝试次数：4种格式 × 3个市场代码 = 12次（与 v0.6 一致）
    max_attempts = len(symbol_formats) * len(market_codes)
    
    # 依次尝试
    tried_combinations = []
    last_error = None
    
    attempt = 0
    for symbol_format in symbol_formats:
        for market_code in market_codes:
            if attempt >= max_attempts:
                break
            
            attempt += 1
            secid = f'{market_code}.{symbol_format}'
            tried_combinations.append(secid)
            
            print(f"    🔍 尝试 [{attempt}/{max_attempts}]: secid={secid}")
            
            result_df = _try_us_request(
                secid=secid,
                symbol=symbol,
                period=period,
                start_date=start_date,
                end_date=end_date,
                is_cached=False
            )
            
            if result_df is not None and not result_df.empty:
                # 成功！保存到缓存
                _us_market_cache[cache_key] = (market_code, symbol_format)
                print(f"    💾 成功组合已缓存: {symbol} → {symbol_format} (市场: {market_code})")
                
                if symbol_format != symbol:
                    print(f"    ℹ️  代码格式转换: {symbol} → {symbol_format}")
                
                return result_df
        
        if attempt >= max_attempts:
            break
    
    # 所有尝试都失败
    error = RequestError(
        error_type='ALL_ATTEMPTS_FAILED',
        message=f'美股 {symbol} 所有尝试均失败',
        details={
            'stock_code': symbol,
            'total_attempts': attempt,
            'tried_combinations': ', '.join(tried_combinations),
            'symbol_formats_tried': symbol_formats,
            'market_codes_tried': market_codes,
        }
    )
    _print_error_log(error, prefix="    ")
    
    return pd.DataFrame()


def _try_us_request(secid, symbol, period, start_date, end_date, is_cached=False):
    """
    尝试单次美股K线请求
    
    Args:
        secid: 证券ID，如 '106.AAPL'
        symbol: 原始股票代码
        period: 周期
        start_date: 开始日期
        end_date: 结束日期
        is_cached: 是否使用缓存的secid
    
    Returns:
        pd.DataFrame or None: 成功返回DataFrame，失败返回None
    """
    # 周期映射
    klt_map = {
        'daily': 101,
        'weekly': 102,
    }
    klt = klt_map.get(period, 101)
    
    # 构建API URL
    url = 'https://push2his.eastmoney.com/api/qt/stock/kline/get'
    
    # 请求参数
    params = {
        'secid': secid,
        'ut': 'fa5fd1943c7b386f172d6893dbfba10b',
        'fields1': 'f1,f2,f3,f4,f5,f6',
        'fields2': 'f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61',
        'klt': klt,
        'fqt': 0,
        'beg': start_date,
        'end': end_date,
        '_': str(int(datetime.now().timestamp() * 1000)),
    }
    
    context = {
        'stock_code': symbol,
        'market': '美股',
        'period': period,
        'secid': secid,
        'start_date': start_date,
        'end_date': end_date,
        'is_cached': is_cached,
    }
    
    try:
        headers = _get_random_headers()
        
        # 根据配置选择使用 Session 或直接请求（v0.7修复：禁用Session以避免被识别）
        if USE_SESSION:
            session = _get_session()
            response = session.get(url, params=params, headers=headers, timeout=30)
        else:
            response = requests.get(url, params=params, headers=headers, timeout=30)
        
        # 检查HTTP状态码
        if response.status_code != 200:
            if not is_cached:  # 只在非缓存请求时打印详细错误
                error = _diagnose_error(response=response, context=context)
                _print_error_log(error, prefix="      ")
            return None
        
        if not response.text:
            return None
        
        try:
            data = response.json()
        except (ValueError, json.JSONDecodeError) as json_err:
            if not is_cached:
                context['response_text'] = response.text
                error = _diagnose_error(exception=json_err, context=context)
                _print_error_log(error, prefix="      ")
            return None
        
        if 'data' not in data or not data['data']:
            return None
        
        klines = data['data'].get('klines', [])
        if not klines:
            return None
        
        # 解析K线数据
        records = []
        parse_errors = 0
        for kline in klines:
            parts = kline.split(',')
            if len(parts) >= 11:
                try:
                    records.append({
                        '日期': parts[0],
                        '开盘': float(parts[1]),
                        '收盘': float(parts[2]),
                        '最高': float(parts[3]),
                        '最低': float(parts[4]),
                        '成交量': float(parts[5]),
                        '成交额': float(parts[6]),
                        '振幅': float(parts[7]),
                        '涨跌幅': float(parts[8]),
                        '涨跌额': float(parts[9]),
                        '换手率': float(parts[10]),
                    })
                except (ValueError, IndexError):
                    parse_errors += 1
            else:
                parse_errors += 1
        
        if parse_errors > 0:
            print(f"      ⚠️  解析警告: {parse_errors}/{len(klines)} 条数据格式异常")
        
        if not records:
            return None
        
        print(f"      ✅ 成功获取 {len(records)} 条K线数据")
        return pd.DataFrame(records)
        
    except requests.RequestException as req_err:
        if not is_cached:
            error = _diagnose_error(exception=req_err, context=context)
            _print_error_log(error, prefix="      ")
        return None
    except Exception as exc:
        if not is_cached:
            error = _diagnose_error(exception=exc, context=context)
            _print_error_log(error, prefix="      ")
        return None


# ==================== 统一接口函数 ====================

def get_historical_data(stock_code, market='A', period='daily', days=365):
    """
    获取股票历史K线数据（v0.7 增强版）
    
    从东方财富网获取指定股票的历史价格数据。
    
    v0.7 改进：
    - 扩充User-Agent池（30个）
    - 简化请求头（仅2个字段，避免过度伪装）
    - 美股智能缓存，避免重复尝试
    - 详细的错误日志，便于诊断问题
    
    Args:
        stock_code (str): 股票代码
            - A股：'000001', '600519' 等
            - 港股：'00700', '01810' 等
            - 美股：'AAPL', 'TSLA' 等
        market (str): 市场类型。可选值：
            - 'A': A股市场
            - 'HK': 港股市场
            - 'US': 美股市场
        period (str): 数据周期。可选值：
            - 'daily': 日K线
            - 'weekly': 周K线
        days (int): 获取天数，默认365天
    
    Returns:
        pd.DataFrame: 历史K线数据，包含以下列：
            - 日期, 开盘, 收盘, 最高, 最低, 成交量, 成交额
            - 振幅, 涨跌幅, 涨跌额, 换手率
    
    示例：
        >>> df = get_historical_data('600519', market='A', period='daily', days=365)
        >>> df = get_historical_data('00700', market='HK', period='weekly', days=730)
        >>> df = get_historical_data('AAPL', market='US', period='daily', days=180)
    """
    # 计算日期范围
    current_date = datetime.now()
    start_date = current_date - timedelta(days=days)
    start_date_str = start_date.strftime("%Y%m%d")
    end_date_str = current_date.strftime("%Y%m%d")
    
    # 根据市场类型调用相应函数
    if market == 'A':
        return _get_a_historical_data(stock_code, period, start_date_str, end_date_str)
    elif market == 'HK':
        return _get_hk_historical_data(stock_code, period, start_date_str, end_date_str)
    elif market == 'US':
        return _get_us_historical_data(stock_code, period, start_date_str, end_date_str)
    else:
        error = RequestError(
            error_type='INVALID_MARKET',
            message=f'不支持的市场类型: {market}',
            details={
                'provided_market': market,
                'supported_markets': ['A', 'HK', 'US'],
            }
        )
        _print_error_log(error)
        return pd.DataFrame()


def clear_us_cache():
    """
    清除美股市场代码缓存
    
    使用场景：
    - 发现缓存的市场代码不再有效
    - 需要重新尝试所有市场代码组合
    """
    global _us_market_cache
    cache_size = len(_us_market_cache)
    _us_market_cache.clear()
    print(f"✓ 已清除 {cache_size} 条美股市场代码缓存")


def get_cache_info():
    """
    获取当前缓存信息
    
    Returns:
        dict: 缓存统计信息
    """
    return {
        'cache_size': len(_us_market_cache),
        'cached_symbols': list(_us_market_cache.keys()),
    }

