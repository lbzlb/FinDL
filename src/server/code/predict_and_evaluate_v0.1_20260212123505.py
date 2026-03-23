import importlib.util
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from openpyxl.utils import get_column_letter
from tqdm import tqdm

# 添加项目路径
current_path = Path(__file__).resolve()
project_root = current_path
while project_root.name and project_root != project_root.parent:
    if (project_root / "src").exists():
        break
    project_root = project_root.parent

if not (project_root / "src").exists():
    raise RuntimeError(
        f"无法找到项目根目录（应包含src文件夹），当前脚本位置: {Path(__file__)}"
    )

sys.path.insert(0, str(project_root))


# ===== 配置区域（修改此处） =====
# 模型目录
MODEL_DIR = "src/server/data/experiments/timexer_latest"

# 预处理数据目录（训练集/验证集，用于阶段一）
PREPROCESSED_DATA_DIR = "src/server/data/preprocess_data_NaNto-1000"

# 预测数据文件（.pt文件路径，用于阶段二）
PREDATA_FILE = "src/server/data/data_v0.01/data.pt"

# 输出目录（结果将保存在此目录下的子文件夹中）
OUTPUT_DIR = "src/server/data/predict"

# 计算设备
DEVICE = "cuda"  # 或 'cpu'

# 推理批次大小
BATCH_SIZE = 256


# ============================================================================
# 通用工具函数
# ============================================================================


def _load_module(module_path: Path, module_name: str):
    """动态加载模块"""
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _adjust_excel_column_widths(writer, sheet_name, df):
    """自动调整Excel列宽"""
    worksheet = writer.sheets[sheet_name]
    for idx, col in enumerate(df.columns):
        max_len = (
            max(df[col].astype(str).map(len).max() if len(df) > 0 else 0, len(str(col)))
            + 2
        )
        col_letter = get_column_letter(idx + 1)
        worksheet.column_dimensions[col_letter].width = min(max_len, 50)


# ============================================================================
# 模型检测与加载
# ============================================================================


def detect_model_type(model_dir: Path) -> tuple:
    """
    检测模型类型和版本

    Returns:
        (model_type, model_version): 模型类型和版本
    """
    config_path = model_dir / "configs" / "model_config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"模型配置文件不存在: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model_name = config.get("model", {}).get("name", "").lower()
    dir_name = model_dir.name.lower()

    # 优先从模型名称识别TimeXer-Official v0.6
    if "timexer_official" in model_name or "timexer_official" in dir_name:
        return "timexer_official", "v0.6_20260206"

    # 优先从目录名精确识别TimeXer版本
    if "timexer" in model_name or "timexer" in dir_name:
        version_pattern = r"v0\.\d+_\d{8}"
        match = re.search(version_pattern, dir_name)
        if match:
            version_str = match.group(0)
            if version_str.startswith("v0.53_"):
                return "timexer_mlp", "v0.53_20260207"
            elif version_str.startswith("v0.45_"):
                return "timexer", "v0.45_20260207"
            elif version_str.startswith("v0.44_"):
                return "timexer", "v0.44_20260126"
            elif version_str.startswith("v0.43_"):
                return "timexer", "v0.43_20260119"
            elif version_str.startswith("v0.42_"):
                return "timexer", "v0.42_20260118"
            elif version_str.startswith("v0.41_"):
                return "timexer", "v0.41_20260116"
            elif version_str.startswith("v0.5_"):
                return "timexer_mlp", "v0.5_20260107"
            elif version_str.startswith("v0.4_"):
                return "timexer", "v0.4_20260106"

        if "timexer_mlp" in model_name:
            return "timexer_mlp", "v0.5_20260107"
        else:
            return "timexer", "v0.4_20260106"

    if "itransformer" in model_name or "itransformer_decoder" in model_name:
        return "itransformer", "v0.1_20251212"
    elif "crossformer" in model_name:
        return "crossformer", "v0.3_20251230"
    elif "tsmixer" in model_name:
        return "tsmixer", "v0.2_20251226"
    else:
        if "itransformer" in dir_name:
            return "itransformer", "v0.1_20251212"
        elif "crossformer" in dir_name:
            return "crossformer", "v0.3_20251230"
        elif "tsmixer" in dir_name:
            return "tsmixer", "v0.2_20251226"
        else:
            raise ValueError(f"无法识别模型类型: {model_name} (目录: {dir_name})")


def load_model_dynamically(model_dir: Path, device: str = "cuda"):
    """
    动态加载模型

    Returns:
        (model, model_version, log_offset): 加载好的模型、模型版本和对数变换offset
    """
    print(f"\n加载模型: {model_dir}")

    model_type, model_version = detect_model_type(model_dir)
    print(f"检测到模型类型: {model_type}, 版本: {model_version}")

    config_path = model_dir / "configs" / "model_config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model_config = config["model"]

    checkpoint_path = model_dir / "checkpoints" / "best_model.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"模型检查点不存在: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    log_offset = checkpoint.get("log_offset", None)
    if log_offset is not None:
        print(f"从checkpoint读取log_offset: {log_offset}")

    # 智能版本检测
    checkpoint_keys = set(checkpoint["model_state_dict"].keys())
    has_missing_embedding = any("missing_embedding" in key for key in checkpoint_keys)
    has_norm_mask = any("norm_mask" in key for key in checkpoint_keys)

    if has_norm_mask and model_type == "timexer_mlp":
        actual_model_version = "v0.53_20260207"
        models_path = project_root / "src" / "models" / actual_model_version
        print(f"检测到norm_mask参数，使用模型版本: {actual_model_version}")
    elif (
        has_missing_embedding
        and model_type == "timexer_mlp"
        and model_version not in ["v0.53_20260207"]
    ):
        actual_model_version = "v0.51_20260128"
        models_path = project_root / "src" / "models" / actual_model_version
        print(f"检测到missing_embedding参数，使用模型版本: {actual_model_version}")
    elif model_version == "v0.44_20260126":
        models_path = project_root / "src" / "models" / "v0.43_20260119"
    elif model_type == "timexer_official":
        models_path = project_root / "src" / "models" / "v0.6_20260206"
    elif model_version == "v0.53_20260207":
        models_path = project_root / "src" / "models" / "v0.53_20260207"
        print("使用TimeXerMLP v0.53模型（Instance Normalization + 反归一化）")
    elif model_version == "v0.45_20260207":
        models_path = project_root / "src/server/code"
        print("使用TimeXer v0.45模型（Instance Normalization + 反归一化）")
    else:
        models_path = project_root / "src/server/code"

    if model_type == "itransformer":
        itransformer_module = _load_module(
            models_path / "itransformer_decoder.py", "itransformer_decoder"
        )
        ModelClass = itransformer_module.iTransformerDecoder
        model = ModelClass(
            input_features=model_config["input_features"],
            seq_len=model_config["seq_len"],
            d_model=model_config["d_model"],
            n_layers=model_config["n_layers"],
            n_heads=model_config["n_heads"],
            d_ff=model_config["d_ff"],
            dropout=model_config["dropout"],
            activation=model_config["activation"],
            decoder_config=model_config.get("decoder", {}),
            input_resnet_config=model_config.get("input_resnet", {}),
            output_resnet_config=model_config.get("output_resnet", {}),
            final_output_config=model_config.get("final_output", {}),
        )

    elif model_type == "crossformer":
        crossformer_module = _load_module(models_path / "crossformer.py", "crossformer")
        ModelClass = crossformer_module.Crossformer
        model = ModelClass(
            seq_len=model_config["seq_len"],
            n_features=model_config["n_features"],
            d_model=model_config["d_model"],
            n_blocks=model_config["n_blocks"],
            n_heads=model_config["n_heads"],
            n_segments=model_config["n_segments"],
            n_feature_groups=model_config["n_feature_groups"],
            dropout=model_config["dropout"],
            activation=model_config["activation"],
            prediction_len=model_config.get("prediction_len", 1),
            router_topk_ratio=model_config.get("router_topk_ratio", 0.5),
            positional_encoding=model_config.get("positional_encoding", {}),
            temporal_aggregation=model_config.get("temporal_aggregation", {}),
            output_projection=model_config.get("output_projection", {}),
        )

    elif model_type == "timexer_mlp":
        timexer_mlp_module = _load_module(models_path / "timexer_mlp.py", "timexer_mlp")
        ModelClass = timexer_mlp_module.TimeXerMLP
        model = ModelClass(
            seq_len=model_config["seq_len"],
            n_features=model_config["n_features"],
            endogenous_features=model_config.get("endogenous_features", 44),
            exogenous_features=model_config.get("exogenous_features", 20),
            prediction_len=model_config.get("prediction_len", 1),
            endogenous_indices=model_config.get("endogenous_indices"),
            exogenous_indices=model_config.get("exogenous_indices"),
            endogenous_blocks=model_config.get("endogenous_blocks", 3),
            endogenous_hidden_dim=model_config.get("endogenous_hidden_dim", 256),
            exogenous_blocks=model_config.get("exogenous_blocks", 2),
            exogenous_hidden_dim=model_config.get("exogenous_hidden_dim", 256),
            shared_time_mixing=model_config.get("shared_time_mixing", True),
            mlp_fusion_ff_dim=model_config.get("mlp_fusion_ff_dim", 512),
            dropout=model_config.get("dropout", 0.1),
            activation=model_config.get("activation", "gelu"),
            use_layernorm=model_config.get("use_layernorm", True),
            use_residual=model_config.get("use_residual", True),
            norm_type=model_config.get("norm_type", "layer"),
            n_blocks=model_config.get("n_blocks", None),
            ff_dim=model_config.get("ff_dim", None),
            temporal_aggregation_config=model_config.get("temporal_aggregation", {}),
            output_projection_config=model_config.get("output_projection", {}),
            use_norm=model_config.get("use_norm", True),
            norm_feature_indices=model_config.get("norm_feature_indices"),
            output_feature_index=model_config.get("output_feature_index", 2),
        )

    elif model_type == "timexer_official":
        timexer_official_module = _load_module(
            models_path / "timexer_official_adapter.py", "timexer_official_adapter"
        )
        ModelClass = timexer_official_module.TimeXerOfficialAdapter
        model = ModelClass(
            seq_len=model_config["seq_len"],
            n_features=model_config["n_features"],
            endogenous_index=model_config.get("endogenous_index", 1),
            prediction_len=model_config.get("prediction_len", 1),
            patch_len=model_config.get("patch_len", 25),
            d_model=model_config.get("d_model", 64),
            n_heads=model_config.get("n_heads", 8),
            e_layers=model_config.get("e_layers", 2),
            d_ff=model_config.get("d_ff", 256),
            dropout=model_config.get("dropout", 0.1),
            activation=model_config.get("activation", "gelu"),
            use_norm=model_config.get("use_norm", True),
            missing_value_flag=model_config.get("missing_value_flag", -1000.0),
        )

    elif model_type == "timexer":
        timexer_module = _load_module(models_path / "timexer.py", "timexer")
        ModelClass = timexer_module.TimeXer
        model_params = {
            "seq_len": model_config["seq_len"],
            "n_features": model_config["n_features"],
            "prediction_len": model_config.get("prediction_len", 1),
            "endogenous_features": model_config.get("endogenous_features", 44),
            "exogenous_features": model_config.get("exogenous_features", 20),
            "endogenous_indices": model_config.get("endogenous_indices"),
            "exogenous_indices": model_config.get("exogenous_indices"),
            "endogenous_blocks": model_config.get("endogenous_blocks", 3),
            "endogenous_hidden_dim": model_config.get("endogenous_hidden_dim", 256),
            "exogenous_blocks": model_config.get("exogenous_blocks", 2),
            "exogenous_hidden_dim": model_config.get("exogenous_hidden_dim", 256),
            "shared_time_mixing": model_config.get("shared_time_mixing", True),
            "time_mixing_type": model_config.get("time_mixing_type", "attention"),
            "time_attn_n_heads": model_config.get("time_attn_n_heads", 8),
            "use_rope": model_config.get("use_rope", True),
            "cross_attn_n_heads": model_config.get("cross_attn_n_heads", 8),
            "cross_attn_ff_dim": model_config.get("cross_attn_ff_dim", 1024),
            "dropout": model_config.get("dropout", 0.1),
            "activation": model_config.get("activation", "gelu"),
            "use_layernorm": model_config.get("use_layernorm", True),
            "use_residual": model_config.get("use_residual", True),
            "temporal_aggregation_config": model_config.get("temporal_aggregation", {}),
            "output_projection_config": model_config.get("output_projection", {}),
        }
        if model_version in [
            "v0.41_20260116",
            "v0.42_20260118",
            "v0.43_20260119",
            "v0.44_20260126",
            "v0.45_20260207",
        ]:
            model_params["use_layernorm_in_tsmixer"] = model_config.get(
                "use_layernorm_in_tsmixer"
            )
            model_params["use_layernorm_in_attention"] = model_config.get(
                "use_layernorm_in_attention"
            )
            model_params["use_layernorm_before_pooling"] = model_config.get(
                "use_layernorm_before_pooling"
            )
        if model_version == "v0.45_20260207":
            model_params["use_norm"] = model_config.get("use_norm", True)
            model_params["norm_feature_indices"] = model_config.get(
                "norm_feature_indices"
            )
            model_params["output_feature_index"] = model_config.get(
                "output_feature_index", 2
            )
        model = ModelClass(**model_params)

    elif model_type == "tsmixer":
        tsmixer_module = _load_module(models_path / "tsmixer.py", "tsmixer")
        ModelClass = tsmixer_module.TSMixer
        model = ModelClass(
            seq_len=model_config["seq_len"],
            n_features=model_config["n_features"],
            prediction_len=model_config.get("prediction_len", 1),
            n_blocks=model_config.get("n_blocks", 4),
            ff_dim=model_config.get("ff_dim", 2048),
            dropout=model_config.get("dropout", 0.1),
            activation=model_config.get("activation", "gelu"),
            norm_type=model_config.get("norm_type", "layer"),
            use_residual=model_config.get("use_residual", True),
            temporal_aggregation_config=model_config.get("temporal_aggregation", {}),
            output_projection_config=model_config.get("output_projection", {}),
        )

    else:
        raise ValueError(f"不支持的模型类型: {model_type}")

    # 弹性加载权重
    missing_keys, unexpected_keys = model.load_state_dict(
        checkpoint["model_state_dict"], strict=False
    )
    if missing_keys:
        print("警告: 模型中缺失以下参数:")
        for key in missing_keys:
            print(f"  - {key}")
    if unexpected_keys:
        print("警告: checkpoint中包含以下额外参数:")
        for key in unexpected_keys:
            print(f"  - {key}")
    if not missing_keys and not unexpected_keys:
        print("✓ 所有参数完全匹配，加载成功")

    model.to(device)
    model.eval()
    print(f"模型参数数量: {model.get_num_parameters():,}")
    print(f"模型加载完成，使用设备: {device}")

    return model, model_version, log_offset


# ============================================================================
# 数据加载函数
# ============================================================================


def load_preprocessed_data(preprocessed_dir: Path):
    """
    加载预处理的训练集和验证集数据

    Returns:
        (train_data, val_data): 训练集和验证集数据，不存在则返回None
    """
    print(f"\n加载预处理数据: {preprocessed_dir}")
    train_data = None
    val_data = None

    train_files = sorted(preprocessed_dir.glob("train_*.pt"))
    if train_files:
        train_pt = train_files[-1]
        print(f"训练集: {train_pt.name}")
        train_data = torch.load(train_pt, map_location="cpu", weights_only=False)
        print(f"训练集样本数: {train_data['metadata']['num_samples']}")
    else:
        print("警告: 未找到训练集文件，将跳过训练集推理")

    val_files = sorted(preprocessed_dir.glob("val_*.pt"))
    if val_files:
        val_pt = val_files[-1]
        print(f"验证集: {val_pt.name}")
        val_data = torch.load(val_pt, map_location="cpu", weights_only=False)
        print(f"验证集样本数: {val_data['metadata']['num_samples']}")
    else:
        print("警告: 未找到验证集文件，将跳过验证集推理")

    if train_data is None and val_data is None:
        raise FileNotFoundError(f"在 {preprocessed_dir} 中未找到任何数据文件")

    return train_data, val_data


def load_index_files(train_data_metadata: dict = None, val_data_metadata: dict = None):
    """
    加载索引文件

    Returns:
        (train_index_df, val_index_df): 索引DataFrame，不存在则返回None
    """
    print("\n加载索引文件:")
    train_index_df = None
    val_index_df = None

    if train_data_metadata is not None:
        train_index_path = Path(train_data_metadata["index_file"])
        print(f"训练集索引: {train_index_path}")
        if train_index_path.exists():
            train_index_df = pd.read_parquet(train_index_path)
            print(f"训练集索引记录数: {len(train_index_df)}")
        else:
            print(f"警告: 训练集索引文件不存在: {train_index_path}")

    if val_data_metadata is not None:
        val_index_path = Path(val_data_metadata["index_file"])
        print(f"验证集索引: {val_index_path}")
        if val_index_path.exists():
            val_index_df = pd.read_parquet(val_index_path)
            print(f"验证集索引记录数: {len(val_index_df)}")
        else:
            print(f"警告: 验证集索引文件不存在: {val_index_path}")

    return train_index_df, val_index_df


def get_close_price_index(feature_names: list) -> int:
    """从特征名列表中找到收盘价的索引"""
    close_price_patterns = ["收盘", "close", "Close", "CLOSE", "收盘价", "close_price"]
    for pattern in close_price_patterns:
        for idx, name in enumerate(feature_names):
            if pattern in name:
                print(f"找到收盘价特征: {name} (索引: {idx})")
                return idx
    raise ValueError(f"无法在特征列表中找到收盘价特征。特征列表: {feature_names}")


def load_predata(predata_file: Path):
    """
    加载预测数据文件

    Returns:
        data_dict: 包含X tensor和metadata的字典
    """
    print(f"\n加载预测数据: {predata_file}")
    if not predata_file.exists():
        raise FileNotFoundError(f"预测数据文件不存在: {predata_file}")

    data = torch.load(predata_file, map_location="cpu", weights_only=False)
    print(f"数据形状: {data['X'].shape}")
    print(f"  - 样本数（公司数）: {data['X'].shape[0]}")
    print(f"  - 序列长度: {data['X'].shape[1]}")
    print(f"  - 特征数: {data['X'].shape[2]}")
    print(f"公司数: {data['metadata']['num_companies']}")
    return data


# ============================================================================
# 阶段一：历史推理
# ============================================================================


def inference_all_samples(
    model: nn.Module,
    model_version: str,
    log_offset: float = None,
    train_data: dict = None,
    val_data: dict = None,
    train_index_df: pd.DataFrame = None,
    val_index_df: pd.DataFrame = None,
    device: str = "cuda",
    batch_size: int = 64,
):
    """
    对所有历史样本（训练集+验证集）进行推理

    Returns:
        results: 包含所有推理结果的列表
    """
    print("\n" + "=" * 80)
    print("阶段一：历史数据推理")
    print("=" * 80)

    # 判断是否需要对数还原
    needs_log_inverse = model_version == "v0.44_20260126"
    if needs_log_inverse:
        if log_offset is None:
            raise ValueError("v0.44模型需要log_offset进行还原")
        print(f"\n*** v0.44模型: exp(pred) - {log_offset:.4f} ***")

    if model_version in ["v0.45_20260207", "v0.53_20260207"]:
        print(
            f"\n*** {model_version}模型: Instance Normalization，输出已在原始尺度 ***"
        )

    # 获取收盘价特征索引
    close_price_idx = None
    if train_data is not None:
        feature_names = train_data["metadata"].get("feature_columns_example", [])
        if feature_names:
            close_price_idx = get_close_price_index(feature_names)
    elif val_data is not None:
        feature_names = val_data["metadata"].get("feature_columns_example", [])
        if feature_names:
            close_price_idx = get_close_price_index(feature_names)

    if close_price_idx is None:
        raise ValueError("无法获取收盘价特征索引！")

    results = []
    model.eval()

    # 推理训练集
    if train_data is not None and train_index_df is not None:
        print(f"\n推理训练集... (样本数: {len(train_data['X'])})")
        train_X = train_data["X"]
        train_y = train_data["y"]

        train_predictions = []
        train_last_close_prices = []
        train_indices = []

        with torch.no_grad():
            for i in tqdm(range(0, len(train_X), batch_size), desc="训练集推理"):
                batch_X = train_X[i : i + batch_size].to(device)
                batch_pred = model(batch_X)
                if needs_log_inverse:
                    batch_pred = torch.exp(batch_pred) - log_offset
                batch_last_close = (
                    batch_X[:, -1, close_price_idx].cpu().numpy().flatten().tolist()
                )
                train_predictions.extend(batch_pred.cpu().numpy().flatten().tolist())
                train_last_close_prices.extend(batch_last_close)
                train_indices.extend(range(i, min(i + batch_size, len(train_X))))

        for idx, pred_value, last_close_price in zip(
            train_indices, train_predictions, train_last_close_prices
        ):
            if idx < len(train_index_df):
                sample_info = train_index_df.iloc[idx]
                true_value = train_y[idx].item()

                abs_relative_error = (
                    abs(pred_value - true_value) / abs(true_value) * 100
                    if true_value != 0
                    else float("inf")
                )
                relative_error_signed = (
                    (pred_value - true_value) / true_value * 100
                    if true_value != 0
                    else float("inf")
                )
                return_rate = (
                    (pred_value - last_close_price) / last_close_price * 100
                    if last_close_price != 0
                    else float("inf")
                )

                target_date = sample_info.get("target_date", "")
                if pd.notna(target_date) and isinstance(target_date, pd.Timestamp):
                    target_date = target_date.strftime("%Y-%m-%d")
                elif pd.isna(target_date):
                    target_date = ""

                results.append(
                    {
                        "split": "train",
                        "sample_id": sample_info["sample_id"],
                        "company_id": sample_info.get("company_id", ""),
                        "company_name": sample_info.get("company_name", ""),
                        "stock_code": sample_info.get("stock_code", ""),
                        "target_date": target_date,
                        "last_input_close": last_close_price,
                        "pred_value": pred_value,
                        "true_value": true_value,
                        "abs_relative_error": abs_relative_error,
                        "relative_error_signed": relative_error_signed,
                        "return_rate": return_rate,
                    }
                )
    else:
        print("\n跳过训练集推理")

    # 推理验证集
    if val_data is not None and val_index_df is not None:
        print(f"\n推理验证集... (样本数: {len(val_data['X'])})")
        val_X = val_data["X"]
        val_y = val_data["y"]

        val_predictions = []
        val_last_close_prices = []
        val_indices = []

        with torch.no_grad():
            for i in tqdm(range(0, len(val_X), batch_size), desc="验证集推理"):
                batch_X = val_X[i : i + batch_size].to(device)
                batch_pred = model(batch_X)
                if needs_log_inverse:
                    batch_pred = torch.exp(batch_pred) - log_offset
                batch_last_close = (
                    batch_X[:, -1, close_price_idx].cpu().numpy().flatten().tolist()
                )
                val_predictions.extend(batch_pred.cpu().numpy().flatten().tolist())
                val_last_close_prices.extend(batch_last_close)
                val_indices.extend(range(i, min(i + batch_size, len(val_X))))

        for idx, pred_value, last_close_price in zip(
            val_indices, val_predictions, val_last_close_prices
        ):
            if idx < len(val_index_df):
                sample_info = val_index_df.iloc[idx]
                true_value = val_y[idx].item()

                abs_relative_error = (
                    abs(pred_value - true_value) / abs(true_value) * 100
                    if true_value != 0
                    else float("inf")
                )
                relative_error_signed = (
                    (pred_value - true_value) / true_value * 100
                    if true_value != 0
                    else float("inf")
                )
                return_rate = (
                    (pred_value - last_close_price) / last_close_price * 100
                    if last_close_price != 0
                    else float("inf")
                )

                target_date = sample_info.get("target_date", "")
                if pd.notna(target_date) and isinstance(target_date, pd.Timestamp):
                    target_date = target_date.strftime("%Y-%m-%d")
                elif pd.isna(target_date):
                    target_date = ""

                results.append(
                    {
                        "split": "val",
                        "sample_id": sample_info["sample_id"],
                        "company_id": sample_info.get("company_id", ""),
                        "company_name": sample_info.get("company_name", ""),
                        "stock_code": sample_info.get("stock_code", ""),
                        "target_date": target_date,
                        "last_input_close": last_close_price,
                        "pred_value": pred_value,
                        "true_value": true_value,
                        "abs_relative_error": abs_relative_error,
                        "relative_error_signed": relative_error_signed,
                        "return_rate": return_rate,
                    }
                )
    else:
        print("\n跳过验证集推理")

    print(f"\n推理完成！共处理 {len(results)} 个样本")
    return results


# ============================================================================
# 阶段一：按公司统计（含新增指标）
# ============================================================================


def _compute_split_stats(split_df: pd.DataFrame, prefix: str) -> dict:
    """
    计算单个数据集（训练集或验证集）的统计指标

    Args:
        split_df: 某个公司某个split的数据
        prefix: 列名前缀，如 'train_' 或 'val_'

    Returns:
        stats: 统计指标字典
    """
    stats = {}

    if len(split_df) == 0:
        for key in [
            "相对误差_均值",
            "相对误差_标准差",
            "绝对相对误差_均值",
            "绝对相对误差_标准差",
            "绝对相对误差_中位数",
            "绝对相对误差_最大值",
            "收益率_均值",
            "±10%命中率",
            "±20%命中率",
            "方向准确率",
            "1σ校准度",
            "2σ校准度",
            "ci95_带宽比",
        ]:
            stats[prefix + key] = np.nan
        stats[prefix + "样本数"] = 0
        return stats

    # 清洗数据
    rel_errors = (
        split_df["relative_error_signed"]
        .replace([float("inf"), -float("inf")], np.nan)
        .dropna()
    )
    abs_rel_errors = (
        split_df["abs_relative_error"]
        .replace([float("inf"), -float("inf")], np.nan)
        .dropna()
    )
    return_rates = (
        split_df["return_rate"].replace([float("inf"), -float("inf")], np.nan).dropna()
    )

    # 基础统计
    mu = rel_errors.mean() if len(rel_errors) > 0 else np.nan
    sigma = rel_errors.std() if len(rel_errors) > 1 else np.nan

    stats[prefix + "相对误差_均值"] = mu
    stats[prefix + "相对误差_标准差"] = sigma
    stats[prefix + "绝对相对误差_均值"] = (
        abs_rel_errors.mean() if len(abs_rel_errors) > 0 else np.nan
    )
    stats[prefix + "绝对相对误差_标准差"] = (
        abs_rel_errors.std() if len(abs_rel_errors) > 1 else np.nan
    )
    stats[prefix + "绝对相对误差_中位数"] = (
        abs_rel_errors.median() if len(abs_rel_errors) > 0 else np.nan
    )
    stats[prefix + "绝对相对误差_最大值"] = (
        abs_rel_errors.max() if len(abs_rel_errors) > 0 else np.nan
    )
    stats[prefix + "收益率_均值"] = (
        return_rates.mean() if len(return_rates) > 0 else np.nan
    )

    # ±10%命中率 和 ±20%命中率
    valid_for_hit = split_df[split_df["true_value"] != 0].copy()
    valid_abs_errors = (
        valid_for_hit["abs_relative_error"]
        .replace([float("inf"), -float("inf")], np.nan)
        .dropna()
    )
    if len(valid_abs_errors) > 0:
        within_10pct = (valid_abs_errors <= 10.0).sum()
        stats[prefix + "±10%命中率"] = within_10pct / len(valid_abs_errors) * 100
        within_20pct = (valid_abs_errors <= 20.0).sum()
        stats[prefix + "±20%命中率"] = within_20pct / len(valid_abs_errors) * 100
    else:
        stats[prefix + "±10%命中率"] = np.nan
        stats[prefix + "±20%命中率"] = np.nan

    # 方向准确率
    valid_dir = split_df[
        (split_df["last_input_close"] != 0)
        & split_df["true_value"].notna()
        & split_df["pred_value"].notna()
    ].copy()
    if len(valid_dir) > 0:
        pred_direction = np.sign(
            valid_dir["pred_value"].values - valid_dir["last_input_close"].values
        )
        true_direction = np.sign(
            valid_dir["true_value"].values - valid_dir["last_input_close"].values
        )
        correct = (pred_direction == true_direction).sum()
        stats[prefix + "方向准确率"] = correct / len(valid_dir) * 100
    else:
        stats[prefix + "方向准确率"] = np.nan

    # 校准度（误差分布正态性检验）
    if len(rel_errors) > 1 and pd.notna(mu) and pd.notna(sigma) and sigma > 0:
        within_1sigma = ((rel_errors >= mu - sigma) & (rel_errors <= mu + sigma)).sum()
        within_2sigma = (
            (rel_errors >= mu - 2 * sigma) & (rel_errors <= mu + 2 * sigma)
        ).sum()
        stats[prefix + "1σ校准度"] = within_1sigma / len(rel_errors) * 100
        stats[prefix + "2σ校准度"] = within_2sigma / len(rel_errors) * 100
    else:
        stats[prefix + "1σ校准度"] = np.nan
        stats[prefix + "2σ校准度"] = np.nan

    # CI95带宽比（方式B）
    if pd.notna(mu) and pd.notna(sigma) and sigma > 0:
        denom_upper = 1 + (mu - 2 * sigma) / 100
        denom_lower = 1 + (mu + 2 * sigma) / 100
        if denom_upper > 0 and denom_lower > 0:
            stats[prefix + "ci95_带宽比"] = (1 / denom_upper - 1 / denom_lower) * 100
        else:
            stats[prefix + "ci95_带宽比"] = np.nan
    else:
        stats[prefix + "ci95_带宽比"] = np.nan

    stats[prefix + "样本数"] = len(split_df)

    return stats


def compute_company_statistics(inference_results: list) -> pd.DataFrame:
    """
    按公司分组计算所有统计指标

    Args:
        inference_results: 推理结果列表

    Returns:
        company_stats_df: 每公司一行的统计DataFrame
    """
    print("\n" + "=" * 80)
    print("计算按公司的统计指标")
    print("=" * 80)

    results_df = pd.DataFrame(inference_results)
    company_groups = results_df.groupby(["company_id", "company_name", "stock_code"])
    print(f"共 {len(company_groups)} 个公司")

    all_stats = []
    for (company_id, company_name, stock_code), group_df in tqdm(
        company_groups, desc="统计公司指标"
    ):
        stats = {
            "company_id": company_id,
            "company_name": company_name,
            "stock_code": stock_code,
        }

        # 训练集统计
        train_df = group_df[group_df["split"] == "train"]
        stats.update(_compute_split_stats(train_df, "train_"))

        # 验证集统计
        val_df = group_df[group_df["split"] == "val"]
        stats.update(_compute_split_stats(val_df, "val_"))

        all_stats.append(stats)

    company_stats_df = pd.DataFrame(all_stats)

    # 按company_id数字大小排序（兼容字符串类型的company_id）
    company_stats_df = company_stats_df.sort_values(
        "company_id", key=lambda x: pd.to_numeric(x, errors="coerce")
    ).reset_index(drop=True)

    print(f"统计完成: {len(company_stats_df)} 个公司")

    return company_stats_df


# ============================================================================
# 阶段二：预测与误差修正
# ============================================================================


def _compute_error_corrected_prices(
    P: float, mu: float, sigma: float, last_close: float
):
    """
    基于方式B计算误差修正价格和收益率

    方式B公式:
        上界_kσ = P / (1 + (μ - k×σ) / 100)
        下界_kσ = P / (1 + (μ + k×σ) / 100)

    Returns:
        dict: 修正价格和收益率
    """
    result = {}

    if pd.notna(mu) and pd.notna(sigma) and sigma > 0:
        for k, label in [(1, "1σ"), (2, "2σ")]:
            denom_upper = 1 + (mu - k * sigma) / 100
            denom_lower = 1 + (mu + k * sigma) / 100

            upper_price = P / denom_upper if denom_upper > 0 else np.nan
            lower_price = P / denom_lower if denom_lower > 0 else np.nan

            result[f"price_upper_{label}"] = upper_price
            result[f"price_lower_{label}"] = lower_price

            # 收益率
            if last_close != 0 and not np.isnan(last_close) and last_close != -1000:
                result[f"return_upper_{label}%"] = (
                    (upper_price - last_close) / last_close * 100
                    if pd.notna(upper_price)
                    else np.nan
                )
                result[f"return_lower_{label}%"] = (
                    (lower_price - last_close) / last_close * 100
                    if pd.notna(lower_price)
                    else np.nan
                )
            else:
                result[f"return_upper_{label}%"] = np.nan
                result[f"return_lower_{label}%"] = np.nan

        # CI95带宽比
        upper_2 = result.get("price_upper_2σ", np.nan)
        lower_2 = result.get("price_lower_2σ", np.nan)
        if pd.notna(upper_2) and pd.notna(lower_2) and P != 0:
            result["ci95_带宽比%"] = (upper_2 - lower_2) / P * 100
        else:
            result["ci95_带宽比%"] = np.nan
    else:
        for label in ["1σ", "2σ"]:
            result[f"price_upper_{label}"] = np.nan
            result[f"price_lower_{label}"] = np.nan
            result[f"return_upper_{label}%"] = np.nan
            result[f"return_lower_{label}%"] = np.nan
        result["ci95_带宽比%"] = np.nan

    return result


def predict_with_error_correction(
    model: nn.Module,
    model_version: str,
    log_offset: float,
    predata: dict,
    company_stats_df: pd.DataFrame,
    device: str = "cuda",
    batch_size: int = 256,
):
    """
    对预测数据进行推理，并基于历史误差统计计算误差修正价格

    Returns:
        prediction_df: 预测结果DataFrame
    """
    print("\n" + "=" * 80)
    print("阶段二：预测与误差修正")
    print("=" * 80)

    needs_log_inverse = model_version == "v0.44_20260126"
    if needs_log_inverse:
        if log_offset is None:
            raise ValueError(
                "v0.44模型需要log_offset进行还原，但未找到log_offset信息！"
            )
        print(f"\n*** v0.44模型: exp(pred) - {log_offset:.4f} ***")
    if model_version in ["v0.45_20260207", "v0.53_20260207"]:
        print(
            f"\n*** {model_version}模型: Instance Normalization，输出已在原始尺度 ***"
        )

    X = predata["X"]
    metadata = predata["metadata"]
    company_info_list = metadata["company_info"]
    feature_columns = metadata["feature_columns_example"]

    # 查找收盘价列索引
    close_idx = None
    for idx, col in enumerate(feature_columns):
        if "收盘" in col:
            close_idx = idx
            break
    if close_idx is None:
        raise ValueError("在特征列中未找到'收盘'列")
    print(f"'收盘'列索引: {close_idx}")

    # 批量推理
    model.eval()
    predictions = []
    print("\n执行批量预测...")
    with torch.no_grad():
        for i in tqdm(range(0, len(X), batch_size), desc="预测进度"):
            batch_X = X[i : i + batch_size].to(device)
            batch_pred = model(batch_X)
            if needs_log_inverse:
                batch_pred = torch.exp(batch_pred) - log_offset
            predictions.extend(batch_pred.cpu().numpy().flatten().tolist())

    print(f"预测完成，共 {len(predictions)} 个样本")

    # 准备company_stats的快速查找（统一company_id为字符串）
    stats_lookup = {}
    for _, row in company_stats_df.iterrows():
        stats_lookup[str(row["company_id"])] = row

    # 构建结果
    results = []
    for idx, company_info in enumerate(company_info_list):
        if idx >= len(predictions):
            break

        company_id = company_info.get("sequence_id", idx)
        company_name = company_info.get("company_name", "Unknown")
        stock_code = company_info.get("stock_code", "Unknown")
        dates = company_info.get("dates", [])
        last_trade_date = dates[-1] if dates else None
        last_close = X[idx, -1, close_idx].item()
        predicted_close = predictions[idx]

        # 预测收益率
        if last_close != 0 and not np.isnan(last_close) and last_close != -1000:
            predicted_return = (predicted_close - last_close) / last_close * 100
        else:
            predicted_return = np.nan

        row = {
            "company_id": company_id,
            "company_name": company_name,
            "stock_code": stock_code,
            "last_trade_date": last_trade_date,
            "last_close_price": last_close,
            "predicted_close": predicted_close,
            "predicted_return_rate%": predicted_return,
        }

        # 查找该公司的统计数据（用字符串匹配）
        company_row = stats_lookup.get(str(company_id), None)

        # 分别用验证集和训练集误差统计计算修正价格
        for stats_prefix, output_prefix in [("val_", "val_"), ("train_", "train_")]:
            if company_row is not None:
                mu = company_row.get(stats_prefix + "相对误差_均值", np.nan)
                sigma = company_row.get(stats_prefix + "相对误差_标准差", np.nan)
            else:
                mu = np.nan
                sigma = np.nan

            row[output_prefix + "μ%"] = mu
            row[output_prefix + "σ%"] = sigma

            # 获取±10%命中率和±20%命中率
            if company_row is not None:
                row[output_prefix + "±10%命中率"] = company_row.get(
                    stats_prefix + "±10%命中率", np.nan
                )
                row[output_prefix + "±20%命中率"] = company_row.get(
                    stats_prefix + "±20%命中率", np.nan
                )
            else:
                row[output_prefix + "±10%命中率"] = np.nan
                row[output_prefix + "±20%命中率"] = np.nan

            # 修正收益率 = 预测收益率 - 相对误差均值μ（一阶偏差修正）
            if pd.notna(predicted_return) and pd.notna(mu):
                row[output_prefix + "修正收益率%"] = predicted_return - mu
            else:
                row[output_prefix + "修正收益率%"] = np.nan

            # 计算误差修正价格
            corrected = _compute_error_corrected_prices(
                predicted_close, mu, sigma, last_close
            )

            for key, value in corrected.items():
                row[output_prefix + key] = value

        results.append(row)

    prediction_df = pd.DataFrame(results)

    # 统计匹配情况
    matched = prediction_df["val_μ%"].notna().sum()
    print(f"\n成功匹配到误差统计的公司数: {matched}/{len(prediction_df)}")

    return prediction_df


# ============================================================================
# 输出函数
# ============================================================================


def save_prediction_report(
    prediction_df: pd.DataFrame, company_stats_df: pd.DataFrame, output_dir: Path
):
    """
    保存预测结果完整报告

    File: 预测结果_完整报告.xlsx (3 sheets) + .parquet
    """
    print("\n" + "-" * 60)
    print("保存: 预测结果_完整报告")
    print("-" * 60)

    # ===== Sheet 1: 预测与误差修正（验证集） =====
    val_columns = [
        "company_id",
        "company_name",
        "stock_code",
        "last_trade_date",
        "last_close_price",
        "predicted_close",
        "predicted_return_rate%",
        "val_修正收益率%",
        "val_μ%",
        "val_σ%",
        "val_price_upper_1σ",
        "val_price_lower_1σ",
        "val_price_upper_2σ",
        "val_price_lower_2σ",
        "val_return_upper_1σ%",
        "val_return_lower_1σ%",
        "val_return_upper_2σ%",
        "val_return_lower_2σ%",
        "val_ci95_带宽比%",
        "val_±10%命中率",
        "val_±20%命中率",
    ]
    # 仅选择存在的列
    val_cols_exist = [c for c in val_columns if c in prediction_df.columns]
    sheet1_df = prediction_df[val_cols_exist].copy()
    # 按company_id数字大小排序
    if "company_id" in sheet1_df.columns:
        sheet1_df = sheet1_df.sort_values(
            "company_id", key=lambda x: pd.to_numeric(x, errors="coerce")
        ).reset_index(drop=True)
    sheet1_df.columns = [
        c.replace("val_", "验证集_") if c.startswith("val_") else c
        for c in sheet1_df.columns
    ]

    # ===== Sheet 2: 预测与误差修正（训练集参考） =====
    train_columns = [
        "company_id",
        "company_name",
        "stock_code",
        "last_trade_date",
        "last_close_price",
        "predicted_close",
        "predicted_return_rate%",
        "train_修正收益率%",
        "train_μ%",
        "train_σ%",
        "train_price_upper_1σ",
        "train_price_lower_1σ",
        "train_price_upper_2σ",
        "train_price_lower_2σ",
        "train_return_upper_1σ%",
        "train_return_lower_1σ%",
        "train_return_upper_2σ%",
        "train_return_lower_2σ%",
        "train_ci95_带宽比%",
        "train_±10%命中率",
        "train_±20%命中率",
    ]
    train_cols_exist = [c for c in train_columns if c in prediction_df.columns]
    sheet2_df = prediction_df[train_cols_exist].copy()
    # 按company_id数字大小排序
    if "company_id" in sheet2_df.columns:
        sheet2_df = sheet2_df.sort_values(
            "company_id", key=lambda x: pd.to_numeric(x, errors="coerce")
        ).reset_index(drop=True)
    sheet2_df.columns = [
        c.replace("train_", "训练集_") if c.startswith("train_") else c
        for c in sheet2_df.columns
    ]

    # ===== Sheet 3: 模型评估指标 =====
    eval_columns_order = [
        "company_id",
        "company_name",
        "stock_code",
        "val_相对误差_均值",
        "val_相对误差_标准差",
        "val_绝对相对误差_均值",
        "val_绝对相对误差_标准差",
        "val_绝对相对误差_中位数",
        "val_绝对相对误差_最大值",
        "val_收益率_均值",
        "val_±10%命中率",
        "val_±20%命中率",
        "val_方向准确率",
        "val_1σ校准度",
        "val_2σ校准度",
        "val_ci95_带宽比",
        "val_样本数",
        "train_相对误差_均值",
        "train_相对误差_标准差",
        "train_绝对相对误差_均值",
        "train_绝对相对误差_标准差",
        "train_绝对相对误差_中位数",
        "train_绝对相对误差_最大值",
        "train_收益率_均值",
        "train_±10%命中率",
        "train_±20%命中率",
        "train_方向准确率",
        "train_1σ校准度",
        "train_2σ校准度",
        "train_ci95_带宽比",
        "train_样本数",
    ]
    eval_cols_exist = [c for c in eval_columns_order if c in company_stats_df.columns]
    sheet3_df = company_stats_df[eval_cols_exist].copy()
    # 按company_id数字大小排序
    if "company_id" in sheet3_df.columns:
        sheet3_df = sheet3_df.sort_values(
            "company_id", key=lambda x: pd.to_numeric(x, errors="coerce")
        ).reset_index(drop=True)

    # 保存Excel
    excel_path = output_dir / "预测结果_完整报告.xlsx"
    print(f"保存Excel: {excel_path}")
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        sheet1_df.to_excel(writer, sheet_name="预测与误差修正(验证集)", index=False)
        _adjust_excel_column_widths(writer, "预测与误差修正(验证集)", sheet1_df)

        sheet2_df.to_excel(writer, sheet_name="预测与误差修正(训练集参考)", index=False)
        _adjust_excel_column_widths(writer, "预测与误差修正(训练集参考)", sheet2_df)

        sheet3_df.to_excel(writer, sheet_name="模型评估指标", index=False)
        _adjust_excel_column_widths(writer, "模型评估指标", sheet3_df)

    # 保存Parquet（完整数据，包含所有列）
    parquet_path = output_dir / "预测结果_完整报告.parquet"
    print(f"保存Parquet: {parquet_path}")
    # 合并prediction_df和company_stats_df
    prediction_df_copy = prediction_df.copy()
    stats_copy = company_stats_df.copy()
    # 统一company_id类型以确保merge匹配（都转为str进行merge，但不影响排序）
    prediction_df_copy["_merge_key"] = prediction_df_copy["company_id"].astype(str)
    stats_copy["_merge_key"] = stats_copy["company_id"].astype(str)
    merged = prediction_df_copy.merge(
        stats_copy.drop(columns=["company_name", "stock_code"], errors="ignore"),
        on="_merge_key",
        how="left",
        suffixes=("", "_stats"),
    )
    # 清理merge辅助列
    merged = merged.drop(columns=["_merge_key"], errors="ignore")
    if "company_id_stats" in merged.columns:
        merged = merged.drop(columns=["company_id_stats"], errors="ignore")
    merged = merged.sort_values(
        "company_id", key=lambda x: pd.to_numeric(x, errors="coerce")
    ).reset_index(drop=True)
    merged.to_parquet(parquet_path, index=False)

    print(f"完成: 预测结果_完整报告 ({len(prediction_df)} 条记录)")


def save_company_ranking(company_stats_df: pd.DataFrame, output_dir: Path):
    """
    保存公司排名文件

    File: 公司排名.xlsx (11 sheets) + .parquet
    """
    print("\n" + "-" * 60)
    print("保存: 公司排名")
    print("-" * 60)

    # ===== 按指标值排名的配置 =====
    # (Sheet名, 指标列名, 升序/降序, 说明)
    ranked_configs = [
        ("验证集_绝对相对误差均值", "val_绝对相对误差_均值", True, "越小越好"),
        ("验证集_绝对相对误差中位数", "val_绝对相对误差_中位数", True, "越小越好"),
        ("验证集_绝对相对误差最大值", "val_绝对相对误差_最大值", True, "越小越好"),
        ("验证集_±10%命中率", "val_±10%命中率", False, "越大越好"),
        ("验证集_±20%命中率", "val_±20%命中率", False, "越大越好"),
        ("验证集_方向准确率", "val_方向准确率", False, "越大越好"),
        ("验证集_95%CI带宽比", "val_ci95_带宽比", True, "越小越精确"),
        ("验证集_平均收益率", "val_收益率_均值", False, "降序"),
        ("训练集_绝对相对误差均值", "train_绝对相对误差_均值", True, "越小越好"),
        ("训练集_±10%命中率", "train_±10%命中率", False, "越大越好"),
        ("训练集_±20%命中率", "train_±20%命中率", False, "越大越好"),
        ("训练集_方向准确率", "train_方向准确率", False, "越大越好"),
        ("训练集_平均收益率", "train_收益率_均值", False, "降序"),
    ]

    # ===== 按company_id排序的参考表配置 =====
    # (Sheet名, 指标列名)
    reference_configs = [
        ("验证集_相对误差均值", "val_相对误差_均值"),
        ("验证集_相对误差标准差", "val_相对误差_标准差"),
        ("训练集_相对误差均值", "train_相对误差_均值"),
        ("训练集_相对误差标准差", "train_相对误差_标准差"),
    ]

    excel_sheets = {}
    parquet_data = []

    # 处理按指标值排名的Sheet
    for sheet_name, value_col, ascending, desc in ranked_configs:
        if value_col not in company_stats_df.columns:
            print(f"警告: 列 {value_col} 不存在，跳过排名: {sheet_name}")
            continue

        ranking_df = company_stats_df[
            ["company_id", "company_name", "stock_code", value_col]
        ].copy()
        ranking_df = ranking_df.rename(columns={value_col: "指标值"})
        ranking_df["排名方式"] = sheet_name

        # 排序（NaN排在最后）
        ranking_df = ranking_df.sort_values(
            "指标值", ascending=ascending, na_position="last"
        ).reset_index(drop=True)

        # 添加排名
        ranking_df["排名"] = np.nan
        valid_mask = ranking_df["指标值"].notna()
        ranking_df.loc[valid_mask, "排名"] = range(1, valid_mask.sum() + 1)

        result_df = ranking_df[
            ["排名", "company_id", "company_name", "stock_code", "指标值"]
        ]
        excel_sheets[sheet_name] = result_df

        parquet_row = ranking_df[
            ["排名方式", "排名", "company_id", "company_name", "stock_code", "指标值"]
        ]
        parquet_data.append(parquet_row)

    # 处理按company_id排序的参考表Sheet
    for sheet_name, value_col in reference_configs:
        if value_col not in company_stats_df.columns:
            print(f"警告: 列 {value_col} 不存在，跳过参考表: {sheet_name}")
            continue

        ref_df = company_stats_df[
            ["company_id", "company_name", "stock_code", value_col]
        ].copy()
        ref_df = ref_df.rename(columns={value_col: "指标值"})
        ref_df["排名方式"] = sheet_name

        # 按company_id数字大小排序
        ref_df = ref_df.sort_values(
            "company_id", key=lambda x: pd.to_numeric(x, errors="coerce")
        ).reset_index(drop=True)

        # 按company_id顺序编号
        ref_df["排名"] = range(1, len(ref_df) + 1)

        result_df = ref_df[
            ["排名", "company_id", "company_name", "stock_code", "指标值"]
        ]
        excel_sheets[sheet_name] = result_df

        parquet_row = ref_df[
            ["排名方式", "排名", "company_id", "company_name", "stock_code", "指标值"]
        ]
        parquet_data.append(parquet_row)

    # 保存Excel
    excel_path = output_dir / "公司排名.xlsx"
    print(f"保存Excel: {excel_path}")
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        for sname, df in excel_sheets.items():
            # Excel sheet名最多31字符
            safe_name = sname[:31]
            df.to_excel(writer, sheet_name=safe_name, index=False)
            _adjust_excel_column_widths(writer, safe_name, df)

    # 保存Parquet
    parquet_path = output_dir / "公司排名.parquet"
    print(f"保存Parquet: {parquet_path}")
    if parquet_data:
        all_ranking_df = pd.concat(parquet_data, ignore_index=True)
        all_ranking_df.to_parquet(parquet_path, index=False)

    print(f"完成: 公司排名 ({len(excel_sheets)} 个排名维度)")


def save_historical_summary(company_stats_df: pd.DataFrame, output_dir: Path):
    """
    保存模型历史推理汇总

    File: 模型历史推理汇总.xlsx (1 sheet) + .parquet
    """
    print("\n" + "-" * 60)
    print("保存: 模型历史推理汇总")
    print("-" * 60)

    # 按company_id数字大小排序
    output_stats_df = company_stats_df.copy()
    output_stats_df = output_stats_df.sort_values(
        "company_id", key=lambda x: pd.to_numeric(x, errors="coerce")
    ).reset_index(drop=True)

    excel_path = output_dir / "模型历史推理汇总.xlsx"
    print(f"保存Excel: {excel_path}")
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        output_stats_df.to_excel(writer, sheet_name="公司统计汇总", index=False)
        _adjust_excel_column_widths(writer, "公司统计汇总", output_stats_df)

    parquet_path = output_dir / "模型历史推理汇总.parquet"
    print(f"保存Parquet: {parquet_path}")
    output_stats_df.to_parquet(parquet_path, index=False)

    print(f"完成: 模型历史推理汇总 ({len(company_stats_df)} 个公司)")


def save_company_details(
    inference_results: list, company_stats_df: pd.DataFrame, output_dir: Path
):
    """
    保存按公司的详细推理明细

    Folder: company_details/ 下每个公司一个 .xlsx + .parquet
    """
    print("\n" + "-" * 60)
    print("保存: 按公司推理明细 (company_details)")
    print("-" * 60)

    details_dir = output_dir / "company_details"
    details_dir.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(inference_results)
    # 先按company_id数字大小排序，再用sort=False保持该顺序进行groupby
    results_df = results_df.sort_values(
        "company_id", key=lambda x: pd.to_numeric(x, errors="coerce")
    ).reset_index(drop=True)
    company_groups = results_df.groupby(
        ["company_id", "company_name", "stock_code"], sort=False
    )

    # 构建company_stats快速查找（统一用字符串key）
    stats_lookup = {}
    for _, row in company_stats_df.iterrows():
        stats_lookup[str(row["company_id"])] = row

    for (company_id, company_name, stock_code), group_df in tqdm(
        company_groups, desc="保存公司明细"
    ):
        train_df = group_df[group_df["split"] == "train"].copy()
        val_df = group_df[group_df["split"] == "val"].copy()

        # 按日期排序
        if "target_date" in train_df.columns and len(train_df) > 0:
            train_df = train_df.sort_values("target_date").reset_index(drop=True)
        if "target_date" in val_df.columns and len(val_df) > 0:
            val_df = val_df.sort_values("target_date").reset_index(drop=True)

        # Sheet 1: 推理结果（训练集和验证集并排）
        output_data = []
        max_len = max(len(train_df), len(val_df))

        for i in range(max_len):
            row = {}
            if i < len(train_df):
                tr = train_df.iloc[i]
                row["训练_数据时间"] = tr["target_date"]
                row["训练_输入最后收盘价"] = tr["last_input_close"]
                row["训练_预测收盘价"] = tr["pred_value"]
                row["训练_真实收盘价"] = tr["true_value"]
                row["训练_收益率%"] = tr["return_rate"]
                row["训练_绝对相对误差"] = tr["abs_relative_error"]
                row["训练_相对误差"] = tr["relative_error_signed"]
            else:
                for k in [
                    "训练_数据时间",
                    "训练_输入最后收盘价",
                    "训练_预测收盘价",
                    "训练_真实收盘价",
                    "训练_收益率%",
                    "训练_绝对相对误差",
                    "训练_相对误差",
                ]:
                    row[k] = None

            if i < len(val_df):
                vl = val_df.iloc[i]
                row["验证_数据时间"] = vl["target_date"]
                row["验证_输入最后收盘价"] = vl["last_input_close"]
                row["验证_预测收盘价"] = vl["pred_value"]
                row["验证_真实收盘价"] = vl["true_value"]
                row["验证_收益率%"] = vl["return_rate"]
                row["验证_绝对相对误差"] = vl["abs_relative_error"]
                row["验证_相对误差"] = vl["relative_error_signed"]
            else:
                for k in [
                    "验证_数据时间",
                    "验证_输入最后收盘价",
                    "验证_预测收盘价",
                    "验证_真实收盘价",
                    "验证_收益率%",
                    "验证_绝对相对误差",
                    "验证_相对误差",
                ]:
                    row[k] = None

            output_data.append(row)

        output_df = pd.DataFrame(output_data)

        # Sheet 2: 统计汇总
        stats_row = stats_lookup.get(str(company_id), None)
        if stats_row is not None:
            stats_df = pd.DataFrame([stats_row.to_dict()])
        else:
            stats_df = pd.DataFrame(
                [
                    {
                        "company_id": company_id,
                        "company_name": company_name,
                        "stock_code": stock_code,
                    }
                ]
            )

        # 文件名
        safe_name = re.sub(r'[<>:"/\\|?*]', "_", str(company_name))
        safe_code = re.sub(r'[<>:"/\\|?*]', "_", str(stock_code))
        filename_base = f"{company_id}_{safe_name}_{safe_code}"

        # 保存Excel
        excel_path = details_dir / f"{filename_base}.xlsx"
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            output_df.to_excel(writer, sheet_name="推理结果", index=False)
            _adjust_excel_column_widths(writer, "推理结果", output_df)
            stats_df.to_excel(writer, sheet_name="统计汇总", index=False)
            _adjust_excel_column_widths(writer, "统计汇总", stats_df)

        # 保存Parquet
        parquet_path = details_dir / f"{filename_base}.parquet"
        # 合并数据和统计
        stats_tag = stats_df.copy()
        stats_tag.insert(0, "统计标记", "统计汇总")
        combined = pd.concat([output_df, stats_tag], ignore_index=True, sort=False)
        combined.to_parquet(parquet_path, index=False)

    print(f"完成: 按公司推理明细 ({len(company_groups)} 个公司保存到 {details_dir})")


# ============================================================================
# 主函数
# ============================================================================


def main():
    """主函数：整合阶段一和阶段二的全流程"""
    model_dir = Path(MODEL_DIR)
    preprocessed_dir = Path(PREPROCESSED_DATA_DIR)
    predata_file = Path(PREDATA_FILE)
    output_base_dir = Path(OUTPUT_DIR)

    # 检查路径
    if not model_dir.exists():
        raise FileNotFoundError(f"模型目录不存在: {model_dir}")
    if not preprocessed_dir.exists():
        raise FileNotFoundError(f"预处理数据目录不存在: {preprocessed_dir}")
    if not predata_file.exists():
        raise FileNotFoundError(f"预测数据文件不存在: {predata_file}")

    # 检查设备
    if DEVICE == "cuda" and not torch.cuda.is_available():
        print("警告: CUDA不可用，使用CPU")
        device = "cpu"
    else:
        device = DEVICE

    # 输出目录
    model_name = model_dir.name
    predata_name = predata_file.parent.name
    run_time = datetime.now().strftime("%Y%m%d%H%M%S")
    output_dir = output_base_dir / f"{model_name}_{predata_name}_{run_time}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("模型预测与评估综合脚本 v0.1")
    print("=" * 80)
    print(f"模型目录: {model_dir}")
    print(f"预处理数据目录: {preprocessed_dir}")
    print(f"预测数据文件: {predata_file}")
    print(f"输出目录: {output_dir}")
    print(f"计算设备: {device}")
    print(f"批次大小: {BATCH_SIZE}")
    print("=" * 80)

    # ===== 加载模型（只需一次） =====
    model, model_version, log_offset = load_model_dynamically(model_dir, device=device)

    # ===== 阶段一：历史推理评估 =====
    print("\n\n" + "█" * 80)
    print("  阶段一：历史推理评估")
    print("█" * 80)

    # 加载训练/验证数据
    train_data, val_data = load_preprocessed_data(preprocessed_dir)
    train_index_df, val_index_df = load_index_files(
        train_data["metadata"] if train_data is not None else None,
        val_data["metadata"] if val_data is not None else None,
    )

    # 推理所有历史样本
    inference_results = inference_all_samples(
        model=model,
        model_version=model_version,
        log_offset=log_offset,
        train_data=train_data,
        val_data=val_data,
        train_index_df=train_index_df,
        val_index_df=val_index_df,
        device=device,
        batch_size=BATCH_SIZE,
    )

    # 计算按公司的统计指标
    company_stats_df = compute_company_statistics(inference_results)

    # ===== 阶段二：预测与误差修正 =====
    print("\n\n" + "█" * 80)
    print("  阶段二：预测与误差修正")
    print("█" * 80)

    # 加载预测数据
    predata = load_predata(predata_file)

    # 预测并计算误差修正价格
    prediction_df = predict_with_error_correction(
        model=model,
        model_version=model_version,
        log_offset=log_offset,
        predata=predata,
        company_stats_df=company_stats_df,
        device=device,
        batch_size=BATCH_SIZE,
    )

    # ===== 保存所有结果 =====
    print("\n\n" + "█" * 80)
    print("  保存结果")
    print("█" * 80)

    save_prediction_report(prediction_df, company_stats_df, output_dir)
    save_company_ranking(company_stats_df, output_dir)
    save_historical_summary(company_stats_df, output_dir)
    save_company_details(inference_results, company_stats_df, output_dir)

    print("\n" + "=" * 80)
    print(f"全部任务完成！结果保存在: {output_dir}")
    print("=" * 80)
    print("  - 预测结果_完整报告.xlsx/.parquet  (3 sheets)")
    print("  - 公司排名.xlsx/.parquet           (11 sheets)")
    print("  - 模型历史推理汇总.xlsx/.parquet    (1 sheet)")
    print(f"  - company_details/                  ({len(company_stats_df)} 个公司)")
    print("=" * 80)


if __name__ == "__main__":
    main()
    
