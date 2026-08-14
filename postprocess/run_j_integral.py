"""
J-integral 后处理 CLI 入口（论文 §4.4 + §5.3 头条）

用法：
    python -m postprocess.run_j_integral \\
        --checkpoint checkpoints/jpinn.pt \\
        --data data/synthetic_thermal.npz \\
        --out_dir logs/j_integral \\
        --n_per_side 200 \\
        --x_lig_values -0.9 -0.7 -0.5 -0.3 -0.1 \\
        --x_wake_values 0.1 0.3 0.5 0.7 0.9

输出：
- logs/j_integral/J_raw.png — PINN 原始 J 曲面（可能含线性漂移）
- logs/j_integral/J_corrected.png — 远场锚定补偿后 J 曲面（应平坦）
- logs/j_integral/J_exact.png — 解析场 J 曲面（参考）
- logs/j_integral/relative_error.png — PINN corrected vs exact 误差
- logs/j_integral/metrics.json — 路径无关度 + 相对误差
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset import ThermalDataset
from postprocess.analytic_J import j_integral_exact_surface
from postprocess.contour_sampling import sweep_contours
from postprocess.far_field_anchoring import (
    compensate_j_surface,
    fit_linear_drift,
    path_independence_metric,
    relative_error,
    select_far_field_anchor,
)
from postprocess.j_integral import j_integral_surface

# matplotlib headless
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (3D 投影)


def load_model_forckp(ckpt_path: Path, device: torch.device, dtype=torch.float64):
    """从 checkpoint 重建模型（复用 visualize.load_model 模式）"""
    from models.pinn_core import build_model
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    ablation = state.get("ablation", "full")
    model = build_model(ablation=ablation, dtype=dtype).to(device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return model, state


def compute_and_save(
    checkpoint: str = "checkpoints/jpinn.pt",
    data: str = "data/synthetic_thermal.npz",
    out_dir: str = "logs/j_integral",
    n_per_side: int = 200,
    x_lig_values: tuple[float, ...] = (-0.9, -0.7, -0.5, -0.3, -0.1),
    x_wake_values: tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 0.9),
    y_range: tuple[float, float] = (-1.0, 1.0),
    anchor_mode: str = "extremes",
) -> dict:
    """主入口：从 checkpoint 计算 J 曲面 → 补偿 → 画图 + 写指标

    Args:
        anchor_mode: B7 (v0.7 阶段 5) 远场锚定选择
            - "extremes"（默认）：min(x_lig) + max(x_wake)
            - "residual_min"：|J_raw - J_fit| 最小 contour（论文 §4.4）

    Returns: dict with all metrics
    """
    device = torch.device("cpu")
    dtype = torch.float64
    torch.set_default_dtype(dtype)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 加载数据集（拿 T_min/T_max/spec）
    ds = ThermalDataset(npz_path=data, device=device, dtype=dtype)

    # 加载模型
    print(f"加载 checkpoint: {checkpoint}")
    model, state = load_model_forckp(Path(checkpoint), device, dtype)

    # 生成 contour 网格
    contours = sweep_contours(
        x_lig_values=x_lig_values,
        x_wake_values=x_wake_values,
        y_range=y_range,
        n_per_side=n_per_side,
    )
    print(f"生成 {len(contours)} 个 contour（{len(x_lig_values)}×{len(x_wake_values)}）")

    # 计算 PINN J 曲面
    print("计算 PINN J 曲面...")
    x_lig_arr, x_wake_arr, J_pinn_grid = j_integral_surface(
        model, contours, ds.T_min, ds.T_max, ds.spec,
    )
    # v0.8 修复：扁平化 J_pinn_grid；不论 J_pinn_grid 是 1D 还是 2D 都正确对应 25 个 contour
    # 旧版嵌套循环 (i, j) 索引错位会传 list 给 np.linalg.lstsq 导致 "Incompatible dimensions"
    J_pinn_flat = np.asarray(J_pinn_grid).ravel()
    # 同样把 x_lig_arr / x_wake_arr 扩展到 25 长度，与 J_pinn_flat 索引对齐
    # 旧版只取 unique 长度（5）但 J 是 (5,5) → 维度不匹配
    if J_pinn_grid.ndim == 2:
        x_lig_flat = np.tile(x_lig_arr, len(x_wake_arr))      # (5,5) 行优先 → (25,)
        x_wake_flat = np.repeat(x_wake_arr, len(x_lig_arr))    # (5,5) 列重复 → (25,)
    else:
        x_lig_flat = x_lig_arr
        x_wake_flat = x_wake_arr

    # 计算解析 J 曲面（参考）
    print("计算解析 J 曲面（参考）...")
    J_exact_flat = j_integral_exact_surface(contours, include_crack=False)

    # 远场锚定
    # B7 (v0.7 阶段 5): anchor_mode
    #   - extremes（默认）：min(x_lig) + max(x_wake)（原行为）
    #   - residual_min：|J_raw - J_fit| 最小 contour（论文 §4.4 更严谨）
    x_lig_far, x_wake_far = select_far_field_anchor(
        x_lig_flat, x_wake_flat, J_pinn_flat, mode=anchor_mode,
    )
    J_pinn_corrected = compensate_j_surface(
        x_lig_flat, x_wake_flat, J_pinn_flat,
        x_lig_far=x_lig_far, x_wake_far=x_wake_far,
    )

    # 解析 J 也补偿（用同样的锚定点，便于比较）
    J_exact_corrected = compensate_j_surface(
        x_lig_flat, x_wake_flat, J_exact_flat,
        x_lig_far=x_lig_far, x_wake_far=x_wake_far,
    )

    # 路径无关度
    metrics = {
        "n_contours": len(contours),
        "n_per_side": n_per_side,
        "x_lig_values": list(x_lig_values),
        "x_wake_values": list(x_wake_values),
        "J_pinn_path_indep_raw": path_independence_metric(J_pinn_flat),
        "J_pinn_path_indep_corrected": path_independence_metric(J_pinn_corrected),
        "J_exact_path_indep_raw": path_independence_metric(J_exact_flat),
        "J_exact_path_indep_corrected": path_independence_metric(J_exact_corrected),
        "J_pinn_far_field": float(J_pinn_flat[0]),  # 第一 contour 的值
        "J_exact_far_field": float(J_exact_flat[0]),
        "relative_error_corrected": relative_error(J_pinn_corrected, J_exact_corrected),
    }

    # 画 3D surface（论文 Fig.12 风格）
    print("画图...")
    unique_lig = sorted(set(x_lig_values))
    unique_wake = sorted(set(x_wake_values))
    if len(unique_lig) * len(unique_wake) == len(contours):
        plot_3d_surface(J_pinn_flat.reshape(len(unique_lig), len(unique_wake)),
                        unique_lig, unique_wake,
                        "J_raw (PINN)", out_path / "J_raw.png")
        plot_3d_surface(J_pinn_corrected.reshape(len(unique_lig), len(unique_wake)),
                        unique_lig, unique_wake,
                        "J_corrected (PINN, 远场锚定)", out_path / "J_corrected.png")
        plot_3d_surface(J_exact_flat.reshape(len(unique_lig), len(unique_wake)),
                        unique_lig, unique_wake,
                        "J_exact (解析场)", out_path / "J_exact.png")
        err = np.abs(J_pinn_corrected.reshape(len(unique_lig), len(unique_wake)) -
                     J_exact_corrected.reshape(len(unique_lig), len(unique_wake)))
        rel_err = err / (np.abs(J_exact_corrected).max() + 1e-12)
        plot_3d_surface(rel_err, unique_lig, unique_wake,
                        "相对误差 (corrected)", out_path / "relative_error.png")

    # 保存 metrics
    metrics_path = out_path / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"\n指标已保存到: {metrics_path}")
    print(f"\n=== J-integral 指标 ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4e}")
        else:
            print(f"  {k}: {v}")

    return metrics


def plot_3d_surface(Z: np.ndarray, x_lig: list, x_wake: list, title: str, save_path: Path):
    """3D surface 图（论文 Fig.12 风格）"""
    X, Y = np.meshgrid(x_lig, x_wake, indexing="ij")
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="none", alpha=0.9)
    ax.set_xlabel("x_lig (韧带侧)")
    ax.set_ylabel("x_wake (尾迹侧)")
    ax.set_zlabel("J")
    ax.set_title(title)
    fig.colorbar(surf, ax=ax, shrink=0.6)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="J-PINN J-integral 后处理（论文 §4.4 头条）")
    # v0.9：默认 None → latest.json 解析（复用 visualize 的解析函数）
    p.add_argument("--checkpoint", default=None,
                   help="checkpoint 路径（默认 outputs/latest.json 中 full 的最新任务）")
    p.add_argument("--data", default="data/synthetic_thermal.npz")
    p.add_argument("--out_dir", default=None,
                   help="输出目录（默认 checkpoint 同目录 j_integral/）")
    p.add_argument("--n_per_side", type=int, default=200)
    p.add_argument("--x_lig_values", type=float, nargs="+",
                   default=[-0.9, -0.7, -0.5, -0.3, -0.1])
    p.add_argument("--x_wake_values", type=float, nargs="+",
                   default=[0.1, 0.3, 0.5, 0.7, 0.9])
    p.add_argument("--y_min", type=float, default=-1.0)
    p.add_argument("--y_max", type=float, default=1.0)
    # B7 (v0.7 阶段 5): 远场锚定模式
    p.add_argument("--anchor_mode", choices=["extremes", "residual_min"],
                   default="extremes",
                   help="远场锚定选择：extremes=min(x_lig)+max(x_wake)；residual_min=|J_raw-J_fit|最小")
    args = p.parse_args()
    # v0.9：checkpoint 解析：显式 > latest > 报错
    if args.checkpoint is None:
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from visualize import resolve_checkpoint_from_latest
        args.checkpoint = resolve_checkpoint_from_latest()
        if args.checkpoint is None:
            p.error("未指定 --checkpoint 且 outputs/latest.json 无 full 任务。"
                    "请先训练（python train.py）或显式传 --checkpoint。")
    # v0.9：out_dir 解析：显式 > checkpoint 同目录 j_integral/ > 旧默认
    if args.out_dir is None:
        args.out_dir = os.path.join(os.path.dirname(args.checkpoint) or ".", "j_integral")

    metrics = compute_and_save(
        checkpoint=args.checkpoint,
        data=args.data,
        out_dir=args.out_dir,
        n_per_side=args.n_per_side,
        x_lig_values=tuple(args.x_lig_values),
        x_wake_values=tuple(args.x_wake_values),
        y_range=(args.y_min, args.y_max),
        anchor_mode=args.anchor_mode,
    )

    # 质量门控
    if metrics["J_pinn_path_indep_corrected"] > 0.10:
        print(f"\n[WARNING] 路径无关性偏大 (>10%)，可能未收敛或模型有 bug")
        return 1
    if metrics["relative_error_corrected"] > 0.5:
        print(f"\n[WARNING] PINN vs 解析 相对误差 > 50%，可能损失权重未调好")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())