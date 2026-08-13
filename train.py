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
import copy
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
from outlier import OutlierConfig
from utils_console import (
    print_info, print_warning, print_error, print_success,
    print_title, print_result, print_header,
)
from utils_tee_eta import Tee, ETAEstimator, estimate_training_time
from schedulers import LossProportionalLR


def datetime_now() -> str:
    """ISO 8601 时间戳（CSV 用）"""
    from datetime import datetime
    return datetime.now().isoformat()


# ============================================================
# 命令行参数
# ============================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="J-PINN 训练入口（2D 热力场降维版）")
    p.add_argument("--epochs", type=int, default=None,
                   help="训练总 epoch（默认 5000；--resume 未传时沿用 checkpoint 的 target）")
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
    # 续训
    p.add_argument("--resume", type=str, default=None,
                   help="从 checkpoint 续训；恢复 model + optimizer + scheduler + RNG state")
    # 损失权重（默认值与 losses.LossWeights dataclass 保持一致；不一致会导致 CLI 静默覆盖）
    p.add_argument("--lambda_pde", type=float, default=100.0)
    p.add_argument("--lambda_interface", type=float, default=10.0)
    p.add_argument("--lambda_interface_normal", type=float, default=1.0,
                   help="L_traction（缝合边法向连续）权重；0=关闭")
    p.add_argument("--lambda_bc", type=float, default=1.0)
    p.add_argument("--lambda_neumann_crack", type=float, default=0.05)
    p.add_argument("--lambda_smooth", type=float, default=0.0)
    # P2 Z-score 边界去噪（v0.5）
    p.add_argument("--outlier_enabled", action="store_true",
                   help="启用 Z-score 边界去噪（论文 §3.3 Eq.19-20）")
    p.add_argument("--outlier_burnin", type=int, default=100,
                   help="P2 burn-in epochs（宽限期）")
    p.add_argument("--outlier_delta", type=float, default=3.0,
                   help="P2 Z-score 阈值 δ")
    p.add_argument("--outlier_ema_alpha", type=float, default=0.1,
                   help="P2 残差平方 EMA 平滑系数 α")
    p.add_argument("--boundary_strategy", type=str, choices=["resample", "fixed"],
                   default="resample",
                   help="外边界采样策略：resample（每 epoch 重新采样）/ fixed（固定点，P2 用）")
    # P17 损失比例 LR 调度器
    p.add_argument("--scheduler", type=str, choices=["cosine", "loss_prop"],
                   default="cosine", help="LR 调度器：cosine / loss_prop（论文 §3.4）")
    p.add_argument("--lr_min", type=float, default=1e-6,
                   help="LR 下限（loss_prop 调度器用）")
    return p.parse_args()


# ============================================================
# 主入口
# ============================================================
def main() -> None:
    args = parse_args()

    # 续训时优先用 checkpoint 内的 args/seed（避免 silent override）
    resume_state = None
    if args.resume:
        if not os.path.exists(args.resume):
            raise FileNotFoundError(f"--resume 路径不存在: {args.resume}")
        resume_state = torch.load(args.resume, map_location="cpu", weights_only=False)
        # 用 checkpoint 内的 seed/epochs 覆盖 CLI（保证续训可复现 + scheduler 对齐）
        ckpt_args = resume_state.get("args", {})
        if "seed" in ckpt_args:
            args.seed = ckpt_args["seed"]
        if args.epochs is None:
            # 用户未显式传 --epochs：沿用 checkpoint 的 target（scheduler 对齐）
            if "epochs" in ckpt_args and int(ckpt_args["epochs"]) > 0:
                args.epochs = int(ckpt_args["epochs"])
        # 用户显式传 --epochs：优先 CLI（允许续训延长目标）
        if args.epochs is None:
            args.epochs = 5000  # 默认兜底
        print_info(f"[RESUME] 从 {args.resume} 恢复（completed_epoch={resume_state.get('epoch', '?')}，"
                   f"target_epochs={args.epochs}，best_loss={resume_state.get('best_loss', '?'):.4e}）")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 非 resume 或 resume 未传 --epochs：默认 5000
    if args.epochs is None:
        args.epochs = 5000

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
    # Fix A2/A3：scheduler.T_max 用 args.epochs（resume 时会被 checkpoint 内 target_epochs 覆盖）；
    # scheduler.last_epoch 通过 load_state_dict 从 checkpoint 恢复（自动设置正确）
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if args.scheduler == "loss_prop":
        scheduler = LossProportionalLR(optimizer, loss_ref=None, lr_min=args.lr_min)
    else:
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
    # P2：Z-score 边界去噪（可选）
    outlier_cfg = None
    if args.outlier_enabled:
        outlier_cfg = OutlierConfig(
            enabled=True,
            burnin_epochs=args.outlier_burnin,
            delta=args.outlier_delta,
            ema_alpha=args.outlier_ema_alpha,
        )
        print_info(
            f"[P2] Z-score 边界去噪启用：burnin={args.outlier_burnin} "
            f"δ={args.outlier_delta} α={args.outlier_ema_alpha}"
        )
    agg = LossAggregator(weights, outlier_cfg=outlier_cfg, device=device)

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
        "bc_active_frac", "bc_n_outliers",
        "lr", "seconds", "eta_seconds", "ema_s",
    ])

    # ============================================================
    # 干跑验证（沿用 v4:1532-1680 + D1-fix：无条件回滚状态）
    # ============================================================
    print_info("[干跑] 验证一次 forward+backward ...")
    # D1-fix（CODE_BUGS N-5）：干跑前快照 model/optimizer/scheduler，
    # 干跑是冒烟测试，参数更新不持久；无条件回滚避免 Adam 动量被首次 batch 污染
    dry_model_state = {k: v.clone() for k, v in model.state_dict().items()}
    dry_optimizer_state = copy.deepcopy(optimizer.state_dict())
    dry_scheduler_state = copy.deepcopy(scheduler.state_dict())
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
    finally:
        # D1-fix：无条件回滚（成功/失败都恢复干净状态）
        model.load_state_dict(dry_model_state)
        optimizer.load_state_dict(dry_optimizer_state)
        scheduler.load_state_dict(dry_scheduler_state)

    # ============================================================
    # 预运行时间估算（P7；仿 v4:954-1063）
    # ============================================================
    if args.epochs >= 10:
        try:
            est_avg, est_total_min = estimate_training_time(
                model, ds, agg, optimizer, device, args, scheduler=scheduler
            )
            print_info(f"[预估] 单 epoch ~{est_avg:.2f}s，"
                       f"总训练时间 ~{est_total_min:.1f} 分钟（~{est_total_min/60:.1f} 小时）")
        except Exception as e:
            print_warning(f"[预估] 估算失败（继续训练）：{e}")

    # ============================================================
    # 主训练循环
    # ============================================================
    print_info("开始训练 ...")
    start = time.time()
    best_loss = float("inf")
    best_state: Optional[dict] = None
    eta = ETAEstimator(total=args.epochs, alpha=0.3)

    # 续训：恢复 model / optimizer / scheduler / RNG state / 起点
    start_epoch = 1
    if resume_state is not None:
        model.load_state_dict(resume_state["model_state_dict"])
        # 兼容旧版 checkpoint（无 optimizer/scheduler state）
        if "optimizer_state_dict" in resume_state and resume_state["optimizer_state_dict"] is not None:
            optimizer.load_state_dict(resume_state["optimizer_state_dict"])
        if "scheduler_state_dict" in resume_state and resume_state["scheduler_state_dict"] is not None:
            scheduler.load_state_dict(resume_state["scheduler_state_dict"])
        if "rng_state" in resume_state and resume_state["rng_state"] is not None:
            torch.set_rng_state(resume_state["rng_state"])
            if "numpy_rng_state" in resume_state:
                np.random.set_state(resume_state["numpy_rng_state"])
        # P2：恢复 outlier tracker 状态（续训）
        if agg.outlier is not None and "outlier_state_dict" in resume_state:
            if resume_state["outlier_state_dict"] is not None:
                agg.outlier.load_state_dict(resume_state["outlier_state_dict"])
        # Fix A1：保存时存的是 completed_epochs 而非 args.epochs
        # 兼容旧版（存 args.epochs）：若 completed_epoch > target_epochs 视为旧版，跳过
        completed_epoch = int(resume_state.get("epoch", 0))
        if completed_epoch >= args.epochs:
            print_warning(f"[RESUME] checkpoint epoch ({completed_epoch}) >= target ({args.epochs})，"
                          f"可能是旧版格式（存的是 target）。尝试按 completed_epoch = {args.epochs} - 1 处理。")
            completed_epoch = args.epochs - 1
        start_epoch = completed_epoch + 1
        # 续训 best_loss 沿用
        if "best_loss" in resume_state and resume_state["best_loss"] is not None:
            best_loss = float(resume_state["best_loss"])
            best_state = resume_state["model_state_dict"]
        print_info(f"[RESUME] 从 epoch {start_epoch} 继续训练（剩余 {args.epochs - start_epoch + 1} 轮）")
        if start_epoch > args.epochs:
            print_warning(f"[RESUME] start_epoch ({start_epoch}) > args.epochs ({args.epochs})，"
                          f"无需训练；直接保存 checkpoint。")
            # 保存并退出
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": args.epochs,  # 此时已"完成" target_epochs
                    "target_epochs": args.epochs,
                    "ablation": args.ablation,
                    "loss_weights": weights.__dict__,
                    "n_params": n_params,
                    "best_loss": best_loss,
                    "ds_meta": {"T_min": ds.T_min, "T_max": ds.T_max, "spec": ds.spec.__dict__},
                    "args": vars(args),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "rng_state": torch.get_rng_state(),
                    "numpy_rng_state": np.random.get_state(),
                    "outlier_state_dict": (
                        agg.outlier.state_dict() if agg.outlier is not None else None
                    ),
                },
                args.out,
            )
            print_info(f"Checkpoint 已保存: {args.out}")
            log_f.close()
            return

    for epoch in range(start_epoch, args.epochs + 1):
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

        # forward + loss（P2：传 current_epoch 用于 burn-in）
        optimizer.zero_grad()
        total, comps = agg(model, batch, current_epoch=epoch)

        if torch.isnan(total):
            print_warning(f"epoch {epoch} 出现 NaN，跳过（保留上一参数）")
            continue

        # backward + 梯度裁剪（v4:1662）
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if args.scheduler == "loss_prop":
            scheduler.step(comps["total"])  # P17：损失比例 LR，需传 loss
        else:
            scheduler.step()

        elapsed = time.time() - epoch_t0
        eta.update(elapsed)  # P7 修复：之前从未调用，ETA 恒为 ?/0
        cur_lr = optimizer.param_groups[0]["lr"]
        eta_info = eta.get_eta(epoch)

        # 写入 CSV（16 列）
        writer.writerow([
            datetime_now(),  # 已返回 ISO 字符串（勿再 .isoformat()）
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
            f"{comps['bc_active_frac']:.4f}",
            f"{comps['bc_n_outliers']:d}",
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

    # Fix A1：epoch 是 for 循环最后一次跑的数（completed_epoch）
    # 不再用 args.epochs（语义：目标 epoch 数）
    completed_epoch = epoch
    total_minutes = (time.time() - start) / 60.0
    print_info(f"\n训练完成：completed_epoch={completed_epoch} / target={args.epochs}，共 {total_minutes:.2f} 分钟")
    print_result("最佳 total loss", best_loss, fmt=".4e")

    # ============================================================
    # 保存 checkpoint（仿 v4:1071-1084 格式）
    # ============================================================
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(
        {
            "model_state_dict": best_state if best_state is not None else model.state_dict(),
            "epoch": completed_epoch,
            "target_epochs": args.epochs,
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
            # 续训所需
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "rng_state": torch.get_rng_state(),
            "numpy_rng_state": np.random.get_state(),
            # P2：outlier tracker 状态
            "outlier_state_dict": (
                agg.outlier.state_dict() if agg.outlier is not None else None
            ),
        },
        args.out,
    )
    print_info(f"Checkpoint 保存到: {args.out}")
    log_f.close()


if __name__ == "__main__":
    main()