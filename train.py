"""
训练入口（仿 v4 run_train.py 风格）

特性（沿用 projects/pe_mmnet/project_v4/run_train.py）：
- Adam + CosineAnnealingLR(eta_min=1e-6)  (line 1494-1500)
- 干跑验证 + NaN 回滚  (line 1532-1680)
- 梯度裁剪 max_norm=1.0  (line 1662)
- CSV 日志 (line 1429-1434)

J-PINN 特有：
- Adam（论文原选，非 AdamW）
- float64 默认（论文 §4.5）
- CPU 默认（论文注：CPU ~10× faster than GPU for ~72K 参数）
- 支持 --ablation {full, two, single}
- 每 epoch 重新采样配点（论文 §2.3）
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from typing import Optional

import numpy as np
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from data.dataset import ThermalDataset
from losses import LossAggregator, LossWeights
from models.pinn_core import build_model
from utils_console import (
    print_info, print_warning, print_error, print_success,
    print_title, print_result, print_header,
)
from utils_tee_eta import Tee, ETAEstimator


def datetime_now() -> str:
    """ISO 8601 时间戳（CSV 用）"""
    from datetime import datetime
    return datetime.now().isoformat()


# ============================================================
# 命令行参数
# ============================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="J-PINN 训练入口（2D 热力场降维版）")
    p.add_argument("--epochs", type=int, default=5000, help="训练总 epoch")
    p.add_argument("--lr", type=float, default=1e-3, help="初始学习率")
    p.add_argument("--device", type=str, default="cpu", help="cpu / cuda")
    p.add_argument("--data", type=str, default="data/synthetic_thermal.npz", help="数据 .npz 路径")
    p.add_argument("--N_int", type=int, default=2500, help="每区域内部配点数")
    p.add_argument("--N_bc", type=int, default=100, help="每条外边配点数")
    p.add_argument("--N_iface", type=int, default=50, help="每条缝合边配点数")
    p.add_argument("--N_crack", type=int, default=50, help="裂纹段每侧配点数")
    p.add_argument(
        "--ablation",
        type=str,
        choices=["full", "two", "single"],
        default="full",
        help="消融模式：full=4区域(J-PINN), two=2区域, single=1区域",
    )
    p.add_argument("--out", type=str, default="checkpoints/jpinn.pt", help="checkpoint 保存路径")
    p.add_argument("--log", type=str, default="logs/train_history.csv", help="训练历史 CSV")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--print_every", type=int, default=500, help="每 N 个 epoch 打印一行")
    # v0.3 友好化日志
    p.add_argument("--log_plain", action="store_true",
                   help="禁用 ANSI 颜色 + Tee 日志文件（用于 CI / 重定向）")
    # 损失权重（默认值与 losses.LossWeights dataclass 保持一致；不一致会导致 CLI 静默覆盖）
    p.add_argument("--lambda_pde", type=float, default=100.0)
    p.add_argument("--lambda_interface", type=float, default=10.0)
    p.add_argument("--lambda_interface_normal", type=float, default=1.0,
                   help="L_traction（缝合边法向连续）权重；0=关闭")
    p.add_argument("--lambda_bc", type=float, default=1.0)
    p.add_argument("--lambda_neumann_crack", type=float, default=0.05)
    p.add_argument("--lambda_smooth", type=float, default=0.0)
    return p.parse_args()


# ============================================================
# 主入口
# ============================================================
def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 精度与设备
    dtype = torch.float64
    torch.set_default_dtype(dtype)
    device = torch.device(args.device)

    print("=" * 70)
    print(f"J-PINN 训练  消融={args.ablation}  设备={args.device}  精度=float64")
    print(f"epochs={args.epochs}  lr={args.lr}  配点: int/region={args.N_int} "
          f"bc/edge={args.N_bc} iface/seam={args.N_iface} crack/side={args.N_crack}")
    print("=" * 70)

    # 数据
    ds = ThermalDataset(npz_path=args.data, device=device, dtype=dtype)
    print_info(f"数据加载完成：T ∈ [{ds.T_min:.4f}, {ds.T_max:.4f}]")
    print_info(f"  域: [{ds.spec.x_min}, {ds.spec.x_max}] × [{ds.spec.y_min}, {ds.spec.y_max}]")
    print_info(f"  裂纹段: x ∈ [{ds.spec.crack_x_min}, {ds.spec.crack_x_max}]")

    # 模型
    model = build_model(ablation=args.ablation, dtype=dtype).to(device)
    n_params = model.count_parameters()
    print_info(f"模型参数量: {n_params}")
    if args.ablation == "full":
        # 论文 §2.3 报告 71,712；本实现因 LayerNorm 略有差异，宽松判定
        assert 60_000 < n_params < 100_000, f"4 区域 JPINN 应约 7-9 万参数，实际 {n_params}"

    # 优化器与调度器（沿用 v4:1494-1500；论文原选 Adam 而非 AdamW）
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # 损失聚合器
    weights = LossWeights(
        lambda_pde=args.lambda_pde,
        lambda_interface=args.lambda_interface,
        lambda_interface_normal=args.lambda_interface_normal,
        lambda_bc=args.lambda_bc,
        lambda_neumann_crack=args.lambda_neumann_crack,
        lambda_smooth=args.lambda_smooth,
    )
    agg = LossAggregator(weights)

    # v0.3 友好化：Tee（实时落盘）+ ETA 估算器
    tee: Tee | None = None
    if not args.log_plain:
        log_dir = os.path.dirname(args.log) or "."
        os.makedirs(log_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(log_dir, f"train_{args.ablation}_{ts}.log")
        tee = Tee(log_path)
        sys.stdout = tee
        sys.stderr = tee
        atexit.register(tee.close)

    # CSV 日志（沿用 v4:1429-1434 + v0.3 扩展 14 列）
    os.makedirs(os.path.dirname(args.log) or ".", exist_ok=True)
    task_id = f"jpinn_{args.ablation}_{time.strftime('%Y%m%d_%H%M%S')}"
    log_f = open(args.log, "w", newline="", encoding="utf-8")
    writer = csv.writer(log_f)
    writer.writerow([
        "timestamp", "task_id", "ablation", "epoch",
        "total", "pde", "iface", "tnormal", "bc", "neumann", "smooth",
        "lr", "seconds", "eta_seconds", "ema_s",
    ])

    # ============================================================
    # 干跑验证（沿用 v4:1532-1680）
    # ============================================================
    print_info("[干跑] 验证一次 forward+backward ...")
    try:
        batch = ds.get_collocation_batch(
            n_int_per_region=128,
            n_bc_per_edge=16,
            n_iface_per_seam=8,
            n_crack_per_side=8,
            seed=args.seed,
        )
        optimizer.zero_grad()
        total, comps = agg(model, batch)
        if torch.isnan(total):
            raise RuntimeError("干跑损失出现 NaN，请检查数据/模型初始化")
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        print_info(f"[干跑] 成功：total={total.item():.4e} pde={comps['pde']:.2e}")
    except RuntimeError as e:
        print_error(f"[干跑] 失败：{e}")
        log_f.close()
        raise

    # ============================================================
    # 主训练循环
    # ============================================================
    print_info("开始训练 ...")
    start = time.time()
    best_loss = float("inf")
    best_state: Optional[dict] = None
    eta = ETAEstimator(total=args.epochs, alpha=0.3)

    for epoch in range(1, args.epochs + 1):
        eta.start_epoch()
        epoch_t0 = time.time()

        # 重新采样（论文 §2.3：每 epoch 重新生成配点）
        batch = ds.get_collocation_batch(
            n_int_per_region=args.N_int,
            n_bc_per_edge=args.N_bc,
            n_iface_per_seam=args.N_iface,
            n_crack_per_side=args.N_crack,
            seed=args.seed + epoch,
        )

        # forward + loss
        optimizer.zero_grad()
        total, comps = agg(model, batch)

        if torch.isnan(total):
            print_warning(f"epoch {epoch} 出现 NaN，跳过（保留上一参数）")
            continue

        # backward + 梯度裁剪（v4:1662）
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        elapsed = time.time() - epoch_t0
        cur_lr = optimizer.param_groups[0]["lr"]
        eta_info = eta.get_eta(epoch)

        # 写入 CSV（14 列）
        writer.writerow([
            datetime_now().isoformat(),
            task_id,
            args.ablation,
            epoch,
            f"{comps['total']:.6e}",
            f"{comps['pde']:.6e}",
            f"{comps['iface']:.6e}",
            f"{comps['tnormal']:.6e}",
            f"{comps['bc']:.6e}",
            f"{comps['neumann']:.6e}",
            f"{comps['smooth']:.6e}",
            f"{cur_lr:.6e}",
            f"{elapsed:.4f}",
            f"{eta_info['eta_seconds']:.1f}",
            f"{eta_info['ema']:.2f}",
        ])

        # 控制台日志（带颜色 + ETA）
        if epoch == 1 or epoch % args.print_every == 0 or epoch == args.epochs:
            print_info(
                f"[{epoch:>5d}/{args.epochs}] total={comps['total']:.4e}  "
                f"pde={comps['pde']:.2e}  iface={comps['iface']:.2e}  "
                f"bc={comps['bc']:.2e}  neum={comps['neumann']:.2e}  "
                f"lr={cur_lr:.2e}  ({elapsed:.2f}s)  "
                f"ETA={eta_info['eta_str']}{eta_info['confidence']} "
                f"finish={eta_info['finish_time']}"
            )

        # 最佳模型跟踪
        if comps["total"] < best_loss:
            best_loss = comps["total"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    total_minutes = (time.time() - start) / 60.0
    print_info(f"\n训练完成：{args.epochs} epoch，共 {total_minutes:.2f} 分钟")
    print_result("最佳 total loss", best_loss, fmt=".4e")

    # ============================================================
    # 保存 checkpoint（仿 v4:1071-1084 格式）
    # ============================================================
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(
        {
            "model_state_dict": best_state if best_state is not None else model.state_dict(),
            "epoch": args.epochs,
            "ablation": args.ablation,
            "loss_weights": weights.__dict__,
            "n_params": n_params,
            "best_loss": best_loss,
            "ds_meta": {
                "T_min": ds.T_min,
                "T_max": ds.T_max,
                "spec": ds.spec.__dict__,
            },
            "args": vars(args),
        },
        args.out,
    )
    print(f"Checkpoint 保存到: {args.out}")
    log_f.close()


if __name__ == "__main__":
    main()