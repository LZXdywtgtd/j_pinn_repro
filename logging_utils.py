"""
D3 训练日志 writer（TensorBoard / W&B，可选）

用法（train.py）：
    writer = make_writer(args.logger, log_dir)
    if writer is not None:
        writer.add_scalars("loss", comps_subset, epoch)

可选依赖：
    tensorboard  →  pip install tensorboard
    wandb        →  pip install wandb（需 wandb login）
两者都未装时 logger 返回 None（不崩溃）。
"""
from __future__ import annotations

from typing import Optional


def make_writer(logger_type: str, log_dir: str):
    """
    创建训练日志 writer。

    Args:
        logger_type: "none" / "tensorboard" / "wandb"
        log_dir: tensorboard event 文件目录（wandb 忽略）

    Returns:
        SummaryWriter / wandb.run / None
    """
    if logger_type == "none":
        return None
    if logger_type == "tensorboard":
        try:
            from torch.utils.tensorboard import SummaryWriter
            return SummaryWriter(log_dir=log_dir)
        except ImportError:
            print("[WARN] tensorboard 未安装；跳过（pip install tensorboard）")
            return None
    if logger_type == "wandb":
        try:
            import wandb
            return wandb.init(project="jpinn-repro")
        except ImportError:
            print("[WARN] wandb 未安装；跳过（pip install wandb && wandb login）")
            return None
    raise ValueError(f"未知 logger_type: {logger_type}")


def log_metrics(writer, epoch: int, comps: dict) -> None:
    """
    发射每 epoch 指标到 writer（tensorboard / wandb 都支持 dict）。

    Args:
        writer: make_writer 返回的 writer（None 则跳过）
        epoch: 当前 epoch
        comps: LossAggregator 返回的 components dict
    """
    if writer is None:
        return
    metrics = {
        "loss/total": comps.get("total", 0.0),
        "loss/pde": comps.get("pde", 0.0),
        "loss/iface": comps.get("iface", 0.0),
        "loss/bc": comps.get("bc", 0.0),
        "loss/neumann": comps.get("neumann", 0.0),
        "loss/smooth": comps.get("smooth", 0.0),
    }
    if "bc_active_frac" in comps:
        metrics["loss/bc_active_frac"] = comps["bc_active_frac"]
    if hasattr(writer, "add_scalars"):
        # tensorboard SummaryWriter
        writer.add_scalars("loss", metrics, epoch)
    elif hasattr(writer, "log"):
        # wandb
        writer.log(metrics, step=epoch)