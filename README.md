# FinDL 项目架构说明

## 环境要求

- Python **>= 3.12**（见 `pyproject.toml`）
- 依赖管理与运行推荐使用 **[uv](https://github.com/astral-sh/uv)**，安装依赖：`uv sync`（或在各目录下用 `uv run` 按项目环境执行脚本）
- 建议在项目根目录激活虚拟环境后再运行（Windows：`.\.venv\Scripts\Activate.ps1`）

## 目录结构

```text
FinDL/
├─ script/
│  ├─ local.py
│  └─ server.py
├─ src/
│  ├─ local/
│  │  ├─ code/
│  │  │  ├─ crawl_index_data.py
│  │  │  ├─ crawl_stock_data.py
│  │  │  ├─ merge_stock_index.py
│  │  │  └─ spider/
│  │  │     ├─ eastmoney_financial.py
│  │  │     ├─ eastmoney_kline.py
│  │  │     ├─ financial_data_mapper.py
│  │  │     └─ us_financial_analysis_indicator.py
│  │  └─ data/
│  │     ├─ macro_indices.xlsx
│  │     ├─ 股票代码汇总-陈俊同-20251118.xlsx
│  │     ├─ 东方财富财务数据API映射最终版-陈俊同-20251030.xlsx
│  │     ├─ stock/
│  │     └─ processed/
│  └─ server/
│     ├─ code/
│     │  ├─ roll_generate_index_v0.7_20260111154500.py
│     │  ├─ process_data_NaNto-1000_20260213.py
│     │  ├─ extract_company_data_v0.3_20260126155425.py
│     │  ├─ parquet_to_predata_v0.01_20260214.py
│     │  ├─ train_timexer.py
│     │  ├─ predict_and_evaluate_v0.1_20260212123505.py
│     │  ├─ predict_with_predata_v0.1_20260120133643.py
│     │  └─ ...（trainer、数据集、特征与指标等辅助模块）
│     ├─ config/
│     │  ├─ dataset_config.yaml
│     │  ├─ timexer_config.yaml
│     │  └─ training_config.yaml
│     └─ data/
│        ├─ roll_generate_index/
│        ├─ preprocess_data_NaNto-1000/
│        ├─ data_v0.01/
│        ├─ experiments/
│        └─ predict/
└─ README.md
```

## 文件作用说明

### 流程入口

- `script/local.py`
  - 通过脚本内 **`STEPS`** 列表按顺序执行子脚本（可按需注释或增删某步，例如仅重跑合并）。
  - 典型完整流程：抓指数 → 抓个股财务与 K 线 → 将个股与宏观指数合并并输出 parquet。
  - 全部步骤执行完毕后，将 `src/local/data/processed` 打包到**项目根目录**，文件名为 **`YYYYMMDD.tar.gz`**（gzip 压缩的 tar）。

- `script/server.py`
  - 通过 **`STEPS`** 串联服务器侧脚本：索引生成 → 数据预处理 → 抽样导出检查 → 预测数据构建 → 模型训练 → 评估与报告输出（顺序见脚本内列表）。

### 本地脚本（`src/local/code`）

- `src/local/code/crawl_index_data.py`
  - 抓取宏观指数，生成 `src/local/data/macro_indices.xlsx`。

- `src/local/code/crawl_stock_data.py`
  - 抓取公司财务、日K、周K，并输出到 `src/local/data/stock`。
  - 同时写入进度与日志（`progress.json`、`company_index.csv`、`kline_failure_log.csv`）。

- `src/local/code/merge_stock_index.py`
  - 将公司数据与宏观指数合并，输出公司级 parquet 到 `src/local/data/processed`。

### 本地抓取依赖（`src/local/code/spider`）

- `src/local/code/spider/eastmoney_financial.py`
  - 财务数据抓取实现。

- `src/local/code/spider/eastmoney_kline.py`
  - 日K/周K数据抓取实现。

- `src/local/code/spider/financial_data_mapper.py`
  - 财务字段映射与标准化处理。

- `src/local/code/spider/us_financial_analysis_indicator.py`
  - 美股财务指标相关实现。

### 服务器脚本（`src/server/code`）

- `src/server/code/roll_generate_index_v0.7_20260111154500.py`
  - 从 `src/server/data` 的 parquet 生成滚动窗口样本索引，输出到 `src/server/data/roll_generate_index`。

- `src/server/code/process_data_NaNto-1000_20260213.py`
  - 基于索引生成训练/验证 tensor（NaN/0 -> -1000），输出到 `src/server/data/preprocess_data_NaNto-1000`。

- `src/server/code/extract_company_data_v0.3_20260126155425.py`
  - 从预处理数据中按公司抽样导出 Excel，用于数据检查。

- `src/server/code/parquet_to_predata_v0.01_20260214.py`
  - 将 `src/server/data` 的 parquet 构建为预测输入，输出到 `src/server/data/data_v0.01/data.pt`。

- `src/server/code/train_timexer.py`
  - 训练模型并输出到 `src/server/data/experiments/timexer_latest`。

- `src/server/code/predict_and_evaluate_v0.1_20260212123505.py`
  - 读取模型、预处理数据、预测数据，输出预测和评估报告到 `src/server/data/predict`。

- `src/server/code/predict_with_predata_v0.1_20260120133643.py`
  - 基于预测数据（`.pt`）的独立推理脚本：加载模型与预数据，逐公司预测并输出 Excel/Parquet（路径在脚本内配置，未接入 `script/server.py` 流水线）。

- **辅助模块**（`src/server/code`，训练与数据管道复用）
  - `metrics_utils.py`：回归类任务的评估指标（有效值过滤、`compute_metrics` / `format_metrics` 等）。
  - `trainer.py`：训练与验证循环、TensorBoard 日志、早停、模型保存；可与带 mask 的数据集配合。
  - `stock_dataset.py`：`torch.utils.data.Dataset` 实现，按索引 parquet 加载样本与目标，支持 **mask**（空白位置 `-1000`）。
  - `preprocessed_dataset.py`：直接读取预处理生成的 `.pt` tensor，支持内存映射（mmap）与 mask，用于加速训练。
  - `data_utils.py`：Parquet 加载、按行取数、LRU **文件缓存**，降低重复读盘开销。
  - `feature_utils.py`：特征列识别、统计量计算/加载、标准化与归一化等特征侧处理。
  - `feature_selector.py`：从 DataFrame 中划分特征列与目标列，供数据集与训练流程使用。
  - `loss_utils.py`：自定义损失（如 MAPE、SMAPE 等），用于训练目标函数。

- `src/server/config/dataset_config.yaml`
  - 数据集、特征和索引配置。

- `src/server/config/timexer_config.yaml`
  - 模型结构参数配置。

- `src/server/config/training_config.yaml`
  - 训练超参数、早停、日志相关配置。

## 运行方式

### 本地（Windows PowerShell）

```powershell
.\.venv\Scripts\Activate.ps1
uv run script/local.py
```

### 服务器

```powershell
.\.venv\Scripts\Activate.ps1
uv run script/server.py
```
