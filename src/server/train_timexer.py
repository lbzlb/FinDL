import rootutils
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import yaml
from datetime import datetime

rootutils.setup_root(__file__, indicator=".python-version", pythonpath=True)
from src.server.models.timexer import TimeXer
from src.server.preprocessed_dataset import PreprocessedStockDataset
from src.server.trainer import EarlyStopping, Trainer
from src.server.utils.feature_utils import save_feature_stats
from src.server.utils.loss_utils import MAPELoss, SMAPELoss

DEFAULT_PREPROCESSED_DIR = Path("data/preprocess_data_NaNto-1000")

DATA_DEVICE = "cpu"  # 修改此处: 'cpu' 或 'cuda'


def load_config(config_path: str) -> dict:
    """加载YAML配置文件"""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def create_optimizer(model: nn.Module, optimizer_config: dict) -> torch.optim.Optimizer:
    """创建优化器"""
    
    lr = float(optimizer_config["lr"])
    weight_decay = float(optimizer_config.get("weight_decay", 0.0))

    return torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    


def create_scheduler(
    optimizer: torch.optim.Optimizer, scheduler_config: dict, num_epochs: int
):

    T_max = int(scheduler_config.get("T_max", num_epochs))
    eta_min = float(scheduler_config.get("eta_min", 0))
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=T_max, eta_min=eta_min
    )


def create_criterion(loss_config: dict) -> nn.Module:
    """创建损失函数"""
    loss_type = loss_config["type"].lower()
    reduction = loss_config.get("reduction", "mean")

    if loss_type == "mse":
        return nn.MSELoss(reduction=reduction)
    elif loss_type == "mae":
        return nn.L1Loss(reduction=reduction)
    elif loss_type == "huber":
        delta = float(loss_config.get("delta", 1.0))
        return nn.HuberLoss(reduction=reduction, delta=delta)
    elif loss_type == "mape":
        epsilon = float(loss_config.get("epsilon", 1e-8))
        max_relative_error = loss_config.get("max_relative_error", None)
        if max_relative_error is not None:
            max_relative_error = float(max_relative_error)
        return MAPELoss(
            reduction=reduction, epsilon=epsilon, max_relative_error=max_relative_error
        )
    elif loss_type == "smape":
        epsilon = float(loss_config.get("epsilon", 1e-8))
        max_relative_error = loss_config.get("max_relative_error", None)
        if max_relative_error is not None:
            max_relative_error = float(max_relative_error)
        return SMAPELoss(
            reduction=reduction, epsilon=epsilon, max_relative_error=max_relative_error
        )
    else:
        raise ValueError(
            f"Unknown loss type: {loss_type}. Available: 'mse', 'mae', 'huber', 'mape', 'smape'"
        )


def main():
    model_config_path = "src/server/config/timexer_config.yaml"

    training_config_path = "src/server/config/training_config.yaml"

    dataset_config_path = "src/server/config/dataset_config.yaml"

    # 加载配置文件
    print("\n加载配置文件...")
    model_config = load_config(model_config_path)
    training_config = load_config(training_config_path)
    dataset_config = load_config(dataset_config_path)

    model_cfg = model_config["model"]
    train_cfg = training_config["training"]

    # 读取TensorBoard配置
    tb_config = training_config.get("tensorboard", {})
    tb_enabled = tb_config.get("enabled", True)

    # 获取代码运行时间（用于目录命名，确保每次运行都是独立的）
    print("\n获取代码运行时间...")
    run_timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    print(f"代码运行时间: {run_timestamp}")
    
    # ===== 使用预处理数据（快速模式，支持mask + 内存映射） =====
    print("使用预处理数据（快速模式，支持mask + 内存映射优化）")

    # 获取数据生成时间
    print("\n获取数据生成时间...")
    data_timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    print(f"数据生成时间: {data_timestamp}")

    # 自动检测预处理文件
    print("\n自动检测预处理文件...")
    train_pt_file, val_pt_file = DEFAULT_PREPROCESSED_DIR / Path("train_NaNto-1000.pt"), DEFAULT_PREPROCESSED_DIR / Path("val_NaNto-1000.pt")

    train_dataset = PreprocessedStockDataset(
        pt_file_path=str(train_pt_file),
        device=None,  # 内存映射模式必须使用CPU
        blank_value=-1000.0,  # 空白数据标记值
        return_mask=True,  # 启用mask
        mmap_mode=True,  # 启用内存映射（关键！）
        precompute_mask=True,  # 预计算mask，提升训练速度，充分利用CPU内存
    )

    val_dataset = PreprocessedStockDataset(
        pt_file_path=str(val_pt_file),
        device=None,
        blank_value=-1000.0,
        return_mask=True,
        mmap_mode=True,  # 启用内存映射
        precompute_mask=True,  # 预计算mask
    )

    # 从元数据中获取特征统计量
    feature_stats = train_dataset.get_feature_stats()

    
    
    output_dir = Path("data/experiments/timexer_latest")

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n输出目录: {output_dir}")
    print(f"  运行时间戳: {run_timestamp}")
    print(f"  数据时间戳: {data_timestamp}")

    # 保存特征统计量（如果需要）
    if feature_stats is not None:
        stats_file = output_dir / "feature_stats.json"
        save_feature_stats(feature_stats, stats_file)
        print(f"特征统计量已保存到: {stats_file}")

    # 获取实际的特征数量和序列长度
    num_features = train_dataset.get_num_features()
    seq_len = train_dataset.get_seq_len()

    print(f"训练集样本数: {len(train_dataset)}")
    print(f"验证集样本数: {len(val_dataset)}")
    print(f"特征数量: {num_features}")
    print(f"序列长度: {seq_len}")

    # 更新模型配置中的特征数量和序列长度（如果不同）
    if model_cfg.get("n_features", 0) != num_features:
        print(
            f"\n警告: 配置中的特征数量 ({model_cfg.get('n_features', 0)}) 与实际特征数量 ({num_features}) 不同"
        )
        print(f"更新模型配置中的特征数量为: {num_features}")
        model_cfg["n_features"] = num_features

    if model_cfg.get("seq_len", 0) != seq_len:
        print(
            f"\n警告: 配置中的序列长度 ({model_cfg.get('seq_len', 0)}) 与实际序列长度 ({seq_len}) 不同"
        )
        print(f"更新模型配置中的序列长度为: {seq_len}")
        model_cfg["seq_len"] = seq_len

    # 创建数据加载器
    print("\n创建数据加载器...")
    # 如果数据在GPU上，必须设置num_workers=0（多进程无法访问GPU数据）
    num_workers = 0 if DATA_DEVICE == "cuda" else train_cfg["num_workers"]
    # 如果数据在GPU上，pin_memory无效（数据已在GPU）
    pin_memory = False if DATA_DEVICE == "cuda" else train_cfg["pin_memory"]

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=train_cfg.get("prefetch_factor", 2)
        if num_workers > 0
        else None,  # 多进程时才需要prefetch
        persistent_workers=train_cfg.get("persistent_workers", True)
        if num_workers > 0
        else False,  # 多进程时才需要persistent
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=train_cfg.get("prefetch_factor", 2)
        if num_workers > 0
        else None,
        persistent_workers=train_cfg.get("persistent_workers", True)
        if num_workers > 0
        else False,
    )

    # 初始化模型（使用更新后的配置）
    print("\n初始化TimeXer模型（v0.45，学习型Missing Embedding）...")

    # 应用特征分离预设方案（如果配置中指定了）
    preset = model_cfg.get("feature_split_preset", None)
    if preset is not None:
        # 根据预设方案生成索引列表，覆盖配置中的索引
        endogenous_indices = list(range(0, 44))
        exogenous_indices = list(range(44, 64))
        # 更新配置
        model_cfg["endogenous_indices"] = endogenous_indices
        model_cfg["exogenous_indices"] = exogenous_indices
    else:
        # 读取配置中的索引列表
        endogenous_indices = model_cfg.get("endogenous_indices", None)
        exogenous_indices = model_cfg.get("exogenous_indices", None)

        # 如果提供了索引列表，打印信息
        if endogenous_indices is not None and exogenous_indices is not None:
            print("\n使用索引列表方式分离特征:")
            print(
                f"  内生数据位置索引: {endogenous_indices[:5]}...{endogenous_indices[-5:]} (共{len(endogenous_indices)}个)"
            )
            print(
                f"  宏观数据位置索引: {exogenous_indices[:5]}...{exogenous_indices[-5:]} (共{len(exogenous_indices)}个)"
            )
        else:
            print("\n使用特征数量方式分离特征:")
            print(f"  内生特征数量: {model_cfg.get('endogenous_features', 44)}")
            print(f"  宏观特征数量: {model_cfg.get('exogenous_features', 20)}")

    model = TimeXer(
        seq_len=model_cfg["seq_len"],  # 使用自动检测的序列长度
        n_features=model_cfg["n_features"],  # 使用自动检测的特征数量
        endogenous_features=model_cfg.get("endogenous_features", 44),
        exogenous_features=model_cfg.get("exogenous_features", 20),
        prediction_len=model_cfg.get("prediction_len", 1),
        endogenous_indices=endogenous_indices,  # 索引列表（如果提供）
        exogenous_indices=exogenous_indices,  # 索引列表（如果提供）
        endogenous_blocks=model_cfg.get("endogenous_blocks", 3),
        endogenous_hidden_dim=model_cfg.get("endogenous_hidden_dim", 256),
        exogenous_blocks=model_cfg.get("exogenous_blocks", 2),
        exogenous_hidden_dim=model_cfg.get("exogenous_hidden_dim", 256),
        shared_time_mixing=model_cfg.get("shared_time_mixing", True),
        cross_attn_n_heads=model_cfg.get("cross_attn_n_heads", 8),
        cross_attn_ff_dim=model_cfg.get("cross_attn_ff_dim", 1024),
        dropout=model_cfg.get("dropout", 0.1),
        activation=model_cfg.get("activation", "gelu"),
        use_layernorm=model_cfg.get("use_layernorm", True),
        use_residual=model_cfg.get("use_residual", True),
        norm_type=model_cfg.get("norm_type", "layer"),
        n_blocks=model_cfg.get("n_blocks", None),  # 保留以兼容，不使用
        ff_dim=model_cfg.get("ff_dim", None),  # 保留以兼容，不使用
        temporal_aggregation_config=model_cfg.get("temporal_aggregation", {}),
        output_projection_config=model_cfg.get("output_projection", {}),
        # v0.45新增：Instance Normalization参数
        use_norm=model_cfg.get("use_norm", True),
        norm_feature_indices=model_cfg.get("norm_feature_indices", None),
        output_feature_index=model_cfg.get("output_feature_index", 2),
    )
    print(f"模型参数数量: {model.get_num_parameters():,}")
    print(f"LayerNorm: {'启用' if model.use_layernorm else '禁用'}")
    print(f"残差连接: {'启用' if model.use_residual else '禁用'}")
    print("学习型Missing Embedding: 启用 (自动处理-1000缺失值)")

    # v0.45新增：显示归一化配置
    print(
        f"Instance Normalization: {'启用' if model_cfg.get('use_norm', True) else '禁用'}"
    )
    if model_cfg.get("use_norm", True):
        norm_indices = model_cfg.get("norm_feature_indices", None)
        if norm_indices is None:
            print(f"  归一化特征: 全部 (0-{num_features - 1})")
        else:
            print(f"  归一化特征: {len(norm_indices)}个特征")
            print(
                f"  归一化索引: {norm_indices[:10]}{'...' if len(norm_indices) > 10 else ''}"
            )
        print(
            f"  输出特征索引: {model_cfg.get('output_feature_index', 2)} (用于反归一化)"
        )
        print("  反归一化: 启用 (输出还原到原始尺度)")
        print("  损失计算: 原始尺度 (更符合业务含义)")

    # 创建优化器
    print("\n创建优化器...")
    optimizer = create_optimizer(model, train_cfg["optimizer"])

    # 创建学习率调度器
    print("创建学习率调度器...")
    scheduler = create_scheduler(
        optimizer, train_cfg["scheduler"], train_cfg["num_epochs"]
    )

    # 创建损失函数
    print("创建损失函数...")
    criterion = create_criterion(train_cfg["loss"])

    # 创建早停机制
    early_stopping = None
    if train_cfg["early_stopping"]["enabled"]:
        early_stop_mode = train_cfg["early_stopping"].get(
            "mode", train_cfg.get("best_metric_mode", "min")
        )
        early_stopping = EarlyStopping(
            patience=train_cfg["early_stopping"]["patience"],
            min_delta=train_cfg["early_stopping"]["min_delta"],
            mode=early_stop_mode,
        )

    # 创建子目录
    configs_dir = output_dir / "configs"
    logs_dir = output_dir / "logs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # TensorBoard目录
    tensorboard_dir = None
    if tb_enabled:
        tb_log_dir = tb_config.get("log_dir", "tensorboard")
        # 使用完整的模型文件夹名称作为运行名称
        run_name = output_dir.name
        tensorboard_dir = output_dir / tb_log_dir / run_name
        print("\nTensorBoard已启用")
        print(f"  日志目录: {tensorboard_dir}")
        print(f"  运行名称: {run_name}")
        print(f"  记录间隔: 每 {tb_config.get('log_interval', 50)} batch")
        print(
            f"  直方图: {'启用' if tb_config.get('log_histograms', True) else '禁用'}"
        )

    # 创建训练器（带TensorBoard支持，v0.3支持mask + 内存映射数据）
    print("\n创建训练器（v0.3，支持mask + 内存映射数据）...")

    # 获取最佳模型选择配置
    best_metric = train_cfg.get("best_metric", "loss")
    best_metric_mode = train_cfg.get("best_metric_mode", "min")
    print(f"最佳模型选择指标: {best_metric} ({best_metric_mode})")

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        device=train_cfg["device"],
        save_dir=str(output_dir / "checkpoints"),
        save_best=train_cfg["save_best"],
        best_metric=best_metric,
        best_metric_mode=best_metric_mode,
        early_stopping=early_stopping,
        mixed_precision=train_cfg["mixed_precision"],
        val_interval=train_cfg["val_interval"],
        save_interval=train_cfg["save_interval"],
        log_file=str(logs_dir / "training_log.jsonl"),
        history_file=str(logs_dir / "training_history.json"),
        # TensorBoard参数
        tensorboard_enabled=tb_enabled,
        tensorboard_dir=str(tensorboard_dir) if tensorboard_dir else None,
        tb_log_interval=tb_config.get("log_interval", 50),
        tb_histogram_interval=tb_config.get("histogram_interval", 500),
        tb_log_histograms=tb_config.get("log_histograms", True),
    )

    # 保存配置到configs子目录
    print("\n保存配置...")

    # 获取模型的详细信息
    model_info = model.get_model_info()

    # 合并模型配置和详细信息，并添加运行时元数据
    detailed_model_config = {
        "metadata": {
            **model_config.get("metadata", {}),
            "run_timestamp": run_timestamp,  # 代码运行时间
            "data_timestamp": data_timestamp,  # 数据生成时间
            "output_directory": str(output_dir),  # 输出目录
        },
        "model": {
            **model_config.get("model", {}),
            **model_info,  # 添加详细的模型信息
        },
    }

    # 保存详细的模型配置
    with open(configs_dir / "model_config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(
            detailed_model_config,
            f,
            allow_unicode=True,
            default_flow_style=False,
            sort_keys=False,
        )

    with open(configs_dir / "training_config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(training_config, f, allow_unicode=True)
    with open(configs_dir / "dataset_config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(dataset_config, f, allow_unicode=True)

    # 开始训练
    print("\n" + "=" * 80)
    print("开始训练")
    print("=" * 80)
    if tb_enabled:
        # 提示使用父目录启动，这样可以对比多个实验
        tb_parent_dir = output_dir / tb_config.get("log_dir", "tensorboard")
        print("\n启动TensorBoard查看训练:")
        print(f"  单次运行: tensorboard --logdir={tensorboard_dir}")
        print(
            f"  对比多次: tensorboard --logdir={tb_parent_dir.parent.parent / 'tensorboard'}"
        )
        print("=" * 80)

    trainer.train(train_cfg["num_epochs"])



if __name__ == "__main__":
    main()
