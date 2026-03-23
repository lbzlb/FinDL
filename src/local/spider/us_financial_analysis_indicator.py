"""
美股财务分析指标获取模块
从东方财富API获取美股公司的财务分析主要指标数据
"""

import random
import time

import pandas as pd
import requests


def stock_financial_us_report_query_market_em(symbol):
    """
    根据股票代码获取SECUCODE（证券统一代码）
    
    支持多种代码格式尝试：
    - 原始格式（如 BRK.B）
    - 短横线格式（如 BRK-B）
    - 无分隔符格式（如 BRKB）
    
    :param symbol: 股票代码，如 "AAPL", "TSLA", "BRK.B"
    :return: SECUCODE，失败返回None
    """
    url = "https://datacenter.eastmoney.com/securities/api/data/v1/get"
    
    # 生成多种可能的代码格式
    test_formats = [
        symbol,                      # 原始格式
        symbol.replace('.', '_'),    # 点号→下划线（东方财富美股常用格式）
        symbol.replace('.', '-'),    # 点号→短横线
        symbol.replace('.', ''),     # 去掉点号
    ]
    
    # 去重
    test_formats = list(dict.fromkeys(test_formats))
    
    for test_code in test_formats:
        params = {
            "reportName": "RPT_USF10_INFO_ORGPROFILE",
            "columns": "SECUCODE,SECURITY_CODE,ORG_CODE,SECURITY_INNER_CODE,ORG_NAME,ORG_EN_ABBR,BELONG_INDUSTRY,"
                       "FOUND_DATE,CHAIRMAN,REG_PLACE,ADDRESS,EMP_NUM,ORG_TEL,ORG_FAX,ORG_EMAIL,ORG_WEB,ORG_PROFILE",
            "quoteColumns": "",
            "filter": f'(SECURITY_CODE="{test_code}")',
            "pageNumber": "1",
            "pageSize": "200",
            "sortTypes": "",
            "sortColumns": "",
            "source": "SECURITIES",
            "client": "PC",
        }
        
        try:
            r = requests.get(url, params=params, timeout=10)
            r.raise_for_status()

            data_json = r.json()
            if not data_json.get("success", False):
                continue  # 尝试下一个格式
            
            data_list = data_json.get("result", {}).get("data", [])
            if not data_list:
                continue  # 尝试下一个格式
            
            # 成功找到数据
            secucode = data_list[0]["SECUCODE"]
            if test_code != symbol:
                print(f"  ℹ️  股票代码格式转换: {symbol} → {test_code}")
            return secucode
            
        except requests.exceptions.RequestException:
            continue  # 尝试下一个格式
        except (KeyError, IndexError):
            continue  # 尝试下一个格式
    
    # 所有格式都失败
    print(f"  ✗ 未找到股票代码 {symbol} 的数据（已尝试格式: {', '.join(test_formats)}）")
    return None


def stock_financial_us_company_type(symbol):
    """
    获取公司类型（一般企业、银行、保险）
    
    :param symbol: 股票代码
    :return: (SECUCODE, ORG_TYPE) 元组，失败返回 (None, None)
    """
    url = "https://datacenter.eastmoney.com/securities/api/data/v1/get"
    # 先获取SECUCODE
    if (stock_code := stock_financial_us_report_query_market_em(symbol)) is None: 
        return None, None
    
    param = {
        "reportName": "RPT_USF10_DATA_MAININDICATOR",
        "columns": "SECUCODE,SECURITY_CODE,SECURITY_NAME_ABBR,ORG_TYPE",
        "quoteColumns": "",
        "filter": f'(SECUCODE="{stock_code}")',
        "pageNumber": "",
        "pageSize": "",
        "sortTypes": "",
        "sortColumns": "",
        "source": "SECURITIES",
        "client": "PC",
    }
    
    try:
        r = requests.get(url, params=param, timeout=10)
        r.raise_for_status()
        
        data_json = r.json()
        if not data_json.get("success", False):
            print(f"API返回错误 (SECUCODE: {stock_code}): {data_json.get('message', 'Unknown error')}")
            return None, None
        
        data_list = data_json.get('result', {}).get('data', [])
        if not data_list:
            print(f"未找到公司类型数据 (SECUCODE: {stock_code})")
            return None, None
        
        data_item = data_list[0]
        return data_item['SECUCODE'], data_item['ORG_TYPE']
        
    except requests.exceptions.RequestException as e:
        print(f"请求失败 (SECUCODE: {stock_code}): {str(e)}")
        return None, None
    except (KeyError, IndexError) as e:
        print(f"数据解析失败 (SECUCODE: {stock_code}): {str(e)}")
        return None, None


def stock_financial_us_analysis_indicator_em(symbol: str = "TSLA", indicator: str = "年报") -> pd.DataFrame:
    """
    东方财富-美股-财务分析-主要指标
    根据公司类型（一般企业、银行、保险）自动选择对应的API接口获取财务数据
    https://emweb.eastmoney.com/PC_USF10/pages/index.html?code=TSLA&type=web&color=w#/cwfx
    :param symbol: 股票代码
    :type symbol: str
    :param indicator: 报表类型，可选值: {"全部", "年报", "单季报", "累计季报"}
    :type indicator: str
    :return: 东方财富-美股-财务分析-主要指标，如果公司类型无法识别则返回空DataFrame
    :rtype: pandas.DataFrame
    """
    url = "https://datacenter.eastmoney.com/securities/api/data/v1/get"
    # 获取SECUCODE和公司类型
    symbol, group = stock_financial_us_company_type(symbol)
    if symbol is None or group is None:
        return pd.DataFrame()
    
    # 基础请求参数
    params = {
        "quoteColumns": "",
        "pageNumber": "",
        "pageSize": "",
        "sortTypes": "-1",
        "sortColumns": "REPORT_DATE",
        "source": "SECURITIES",
        "client": "PC",
    }
    
    # 根据公司类型选择对应的API接口和字段
    match group:
        case "一般企业":
            params.update({
                "reportName": "RPT_USF10_FN_GMAININDICATOR",
                "columns":"USF10_FN_GMAININDICATOR"
            })
        case "银行":
            params.update({
                "reportName": "RPT_USF10_FN_BMAININDICATOR",
                "columns": "ORG_CODE,SECURITY_CODE,SECURITY_NAME_ABBR,SECUCODE,SECURITY_INNER_CODE,REPORT_DATE,STD_REPORT_DATE,START_DATE,NOTICE_DATE,DATE_TYPE,DATE_TYPE_CODE,REPORT_TYPE,REPORT_DATA_TYPE,FINANCIAL_DATE,CURRENCY,CURRENCY_NAME,ACCOUNT_STANDARD,ACCOUNT_STANDARD_NAME,ORGTYPE,TOTAL_INCOME,TOTAL_INCOME_YOY,NET_INTEREST_INCOME,NET_INTEREST_INCOME_YOY,PARENT_HOLDER_NETPROFIT,PARENT_HOLDER_NETPROFIT_YOY,BASIC_EPS_CS,BASIC_EPS_CS_YOY,DILUTED_EPS_CS,DILUTED_EPS_CS_YOY,LOAN_LOSS_PROVISION,LOAN_LOSS_PROVISION_YOY,LOAN_DEPOSIT,LOAN_EQUITY,LOAN_ASSETS,DEPOSIT_EQUITY,DEPOSIT_ASSETS,ROL,ROD,ROE,ROE_YOY,ROA,ROA_YOY"
            })
        case "保险":
            params.update({
                "reportName": "RPT_USF10_FN_IMAININDICATOR",
                "columns": "ORG_CODE,SECURITY_CODE,SECUCODE,SECURITY_NAME_ABBR,SECURITY_INNER_CODE,STD_REPORT_DATE,REPORT_DATE,DATE_TYPE,DATE_TYPE_CODE,REPORT_TYPE,REPORT_DATA_TYPE,FISCAL_YEAR,START_DATE,NOTICE_DATE,ACCOUNT_STANDARD,ACCOUNT_STANDARD_NAME,CURRENCY,CURRENCY_NAME,ORGTYPE,TOTAL_INCOME,TOTAL_INCOME_YOY,PREMIUM_INCOME,PREMIUM_INCOME_YOY,PARENT_HOLDER_NETPROFIT,PARENT_HOLDER_NETPROFIT_YOY,BASIC_EPS_CS,BASIC_EPS_CS_YOY,DILUTED_EPS_CS,PAYOUT_RATIO,CAPITIAL_RATIO,ROE,ROE_YOY,ROA,ROA_YOY,DEBT_RATIO,DEBT_RATIO_YOY,EQUITY_RATIO"
            })
    
    # 根据报表类型设置过滤条件
    match indicator:
        case "全部":
            params.update({"filter": f'(SECUCODE="{symbol}")(DATE_TYPE_CODE in ("001","002","003","004"))'})
        case "年报":
            params.update({"filter": f'(SECUCODE="{symbol}")(DATE_TYPE_CODE="001")'})
        case "单季报":
            params.update({"filter": f'(SECUCODE="{symbol}")(DATE_TYPE_CODE in ("003","006","007","008"))'})
        case "累计季报":
            params.update({"filter": f'(SECUCODE="{symbol}")(DATE_TYPE_CODE in ("002","004"))'})
        case _:
            raise ValueError("请输入正确的 indicator 参数")
    
    # 请求API并解析数据
    try:
        r = requests.get(url, params=params)
        r.raise_for_status()

        data_json = r.json()
        if not data_json.get("success", False):
            print(f"API返回错误 (symbol: {symbol}): {data_json.get('message', 'Unknown error')}")
            return pd.DataFrame()
        # 转换为DataFrame
        temp_df = pd.DataFrame(data_json["result"]["data"])
        return temp_df
    except requests.exceptions.RequestException as e:
        print(f"请求失败 (symbol: {symbol}): {str(e)}")
        return pd.DataFrame()

if __name__ == "__main__":
    # 批量获取指定股票的财务指标数据
    df = pd.read_excel("docs/股票代码汇总-陈俊同-20251118.xlsx")
    code_list = df[df['市场']=="美股"]['股票代码'].tolist()

    for stock_code in ["AXP", "AIG", "ACGL", "FRT", "HES", "MTB", "MKTX", "PFG", "UNM", "VTR"]:
        print(f"正在获取{stock_code}-财务主要指标")
        # 随机延迟，避免请求过快
        time.sleep(random.uniform(1,10))
        df = stock_financial_us_analysis_indicator_em(stock_code,"单季报")
        if not df.empty:
            print(f"成功获取到{stock_code}的数据")
        # 保存到Excel文件
        df.to_excel(f"demo/{stock_code}.xlsx", index=False)

