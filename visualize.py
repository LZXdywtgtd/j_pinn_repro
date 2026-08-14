"""
可视化脚本：加载 checkpoint 画图

输出到 logs/figures/：
- pred_vs_true_heatmap.png : 三联图（预测 / 真值 / |误差|）
- loss_curves.png         : 损失分量曲线（来自 train_history.csv）
- per_region_2x2.png      : 4 区域子图（每个 MLP 在其域的预测）

支持：
- 单 checkpoint 模式：默认
- 对比模式（--compare）：多 checkpoint 同框对比 E₂ 误差
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import List

import numpy as np
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from data.dataset import ThermalDataset
from models.pinn_core import build_model
from jpinn_core.utils import (
    DEFAULT_DOMAIN,
    T_exact_torch,
    denormalize_from_unit,
    normalize_to_unit,
    region_id,
)

# matplotlib headless
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# ============================================================
# v0.9 结果管理：latest 解析
# ============================================================
def resolve_checkpoint_from_latest(output_root: str = "outputs", ablation: str = "full") -> str | None:
    """从 outputs/latest.json 解析最新 checkpoint；无则返回 None"""
    latest_path = os.path.join(output_root, "latest.json")
    if not os.path.exists(latest_path):
        return None
    import json
    try:
        with open(latest_path, "r", encoding="utf-8") as f:
            latest = json.load(f)
    except Exception:
        return None
    task_id = latest.get(ablation)
    if not task_id:
        return None
    ckpt = os.path.join(output_root, ablation, task_id, "model_best.pt")
    return ckpt if os.path.exists(ckpt) else None


# 命令行
# ============================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="J-PINN 可视化")
    # v0.9：默认 None → latest.json 解析 → 报错提示
    p.add_argument("--checkpoint", type=str, default=None,
                   help="checkpoint 路径（默认 outputs/latest.json 中 full 的最新任务）")
    p.add_argument("--data", type=str, default="data/synthetic_thermal.npz")
    p.add_argument("--out_dir", type=str, default=None,
                   help="输出目录（默认 checkpoint 同目录 figures/）")
    p.add_argument("--compare", type=str, nargs="*", default=None,
                   help="消融对比：传多个 checkpoint 路径，如 --compare a.pt b.pt c.pt")
    p.add_argument("--compare_labels", type=str, nargs="*", default=None)
    args = p.parse_args()
    # checkpoint 解析：显式 > latest > 报错
    if args.checkpoint is None and args.compare is None:
        args.checkpoint = resolve_checkpoint_from_latest()
        if args.checkpoint is None:
            p.error("未指定 --checkpoint 且 outputs/latest.json 无 full 任务。"
                    "请先训练（python train.py）或显式传 --checkpoint。")
    # out_dir 解析：显式 > checkpoint 同目录 figures/ > 旧默认
    if args.out_dir is None:
        if args.checkpoint:
            args.out_dir = os.path.join(os.path.dirname(args.checkpoint) or ".", "figures")
        else:
            args.out_dir = "logs/figures"
    return args


# ============================================================
# 加载模型
# ============================================================
def load_model(ckpt_path: str, device: torch.device, dtype=torch.float64):
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    ablation = state.get("ablation", "full")
    model = build_model(ablation=ablation, dtype=dtype).to(device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return model, state


# ============================================================
# 在网格上预测
# ============================================================
@torch.no_grad()
def predict_on_grid(model, ds: ThermalDataset, N: int = 200, device="cpu"):
    spec = ds.spec
    xs = torch.linspace(spec.x_min, spec.x_max, N, dtype=torch.float64, device=device)
    ys = torch.linspace(spec.y_min, spec.y_max, N, dtype=torch.float64, device=device)
    X, Y = torch.meshgrid(xs, ys, indexing="xy")
    rid = region_id(X.flatten(), Y.flatten()).to(device)
    x_f, y_f = X.flatten(), Y.flatten()

    T_pred_norm = model(x_f, y_f, rid)
    T_pred_phys = denormalize_from_unit(T_pred_norm, ds.T_min, ds.T_max)
    T_pred_phys = T_pred_phys.reshape(N, N).cpu().numpy()

    T_true_phys = T_exact_torch(X, Y, include_crack=True,
                                crack_x_max=spec.crack_x_max).cpu().numpy()

    return X.cpu().numpy(), Y.cpu().numpy(), T_pred_phys, T_true_phys


# ============================================================
# 图 1：预测 vs 真值热图
# ============================================================
def plot_pred_vs_true(X, Y, T_pred, T_true, ds: ThermalDataset, out_path: str):
    err = np.abs(T_pred - T_true)
    rel_err = err / (np.abs(T_true).max() + 1e-9)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    extent = [ds.spec.x_min, ds.spec.x_max, ds.spec.y_min, ds.spec.y_max]
    vmin, vmax = T_true.min(), T_true.max()

    im0 = axes[0].imshow(T_pred, extent=extent, origin="lower", cmap="inferno", vmin=vmin, vmax=vmax)
    axes[0].set_title("Predicted T(x,y)")
    axes[0].set_xlabel("x"); axes[0].set_ylabel("y")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(T_true, extent=extent, origin="lower", cmap="inferno", vmin=vmin, vmax=vmax)
    axes[1].set_title("Exact T(x,y)")
    axes[1].set_xlabel("x"); axes[1].set_ylabel("y")
    plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(rel_err, extent=extent, origin="lower", cmap="viridis", vmin=0, vmax=0.1)
    axes[2].set_title("|Rel. error|")
    axes[2].set_xlabel("x"); axes[2].set_ylabel("y")
    plt.colorbar(im2, ax=axes[2])

    # 标出裂纹段
    for ax in axes:
        ax.axhline(0, color="cyan", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.axvline(0, color="cyan", linestyle="--", linewidth=0.8, alpha=0.6)

    fig.suptitle(f"J-PINN prediction  |  max|T-Pred - T-True| = {err.max():.4f}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return err.max(), float(np.sqrt(np.mean(err ** 2)))


# ============================================================
# 图 2：损失曲线（从 CSV）
# ============================================================
def plot_loss_curves(csv_path: str, out_path: str):
    if not os.path.exists(csv_path):
        print(f"  [skip] loss_curves: {csv_path} 不存在")
        return
    epochs, total, pde, iface, bc, neum, smooth = [], [], [], [], [], [], []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            total.append(float(row["total"]))
            pde.append(float(row["pde"]))
            iface.append(float(row["iface"]))
            bc.append(float(row["bc"]))
            neum.append(float(row["neumann"]))
            smooth.append(float(row["smooth"]))
    # v0.8 阶段 9 修复：空 CSV（0 行数据）时直接跳过，避免 vals_arr.max() 对空数组崩溃
    if len(epochs) == 0:
        print(f"  [skip] loss_curves: {csv_path} 是空文件（0 行数据），可能训练未正常写盘")
        return

    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    series = [
        ("total", total, "Total loss"),
        ("pde", pde, "L_pde (∇²T)"),
        ("iface", iface, "L_interface"),
        ("bc", bc, "L_bc (Dirichlet)"),
        ("neum", neum, "L_neumann_crack"),
        ("smooth", smooth, "L_smooth"),
    ]
    for ax, (name, vals, title) in zip(axes.flat, series):
        # 处理全零/负值情况（log scale 不允许 ≤0）
        vals_arr = np.array(vals)
        vmin = max(vals_arr[vals_arr > 0].min() * 0.5 if (vals_arr > 0).any() else 1e-12, 1e-12)
        vmax = max(vals_arr.max(), vmin * 10)
        ax.plot(epochs, vals, linewidth=1.0)
        # 若有 ≤0 值则跳过 log scale
        if (vals_arr > 0).all():
            ax.set_yscale("log")
            ax.set_ylim(vmin, vmax)
        else:
            ax.set_yscale("symlog")
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.grid(True, alpha=0.3)
    fig.suptitle("Training loss components (log/symlog scale)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# v0.9 完整成长面板：损失六分量 / LR 轨迹 / P2 去噪 / 训练速度
# ============================================================
def plot_training_growth(csv_path: str, out_path: str):
    """4 行面板，复用 train_history.csv 的 16 列（无 schema 变更）：
    1. 损失六分量（log/symlog）
    2. 学习率轨迹（loss_prop 调度器随损失衰减）
    3. P2 去噪过程（bc_active_frac + bc_n_outliers 双轴）
    4. 训练速度（seconds/epoch，EMA 平滑）
    """
    if not os.path.exists(csv_path):
        print(f"  [skip] growth_panel: {csv_path} 不存在")
        return
    import csv as _csv
    epochs, lr, active_frac, n_outliers, seconds = [], [], [], [], []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = _csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            lr.append(float(row["lr"]))
            active_frac.append(float(row["bc_active_frac"]))
            n_outliers.append(int(row["bc_n_outliers"]))
            seconds.append(float(row["seconds"]))
    if len(epochs) == 0:
        print(f"  [skip] growth_panel: {csv_path} 是空文件（0 行数据）")
        return

    fig, axes = plt.subplots(4, 1, figsize=(12, 14), sharex=True)

    # 行 1：损失六分量
    ax = axes[0]
    total, pde, iface, bc, neum, smooth = [], [], [], [], [], []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = _csv.DictReader(f)
        for row in reader:
            total.append(float(row["total"]))
            pde.append(float(row["pde"]))
            iface.append(float(row["iface"]))
            bc.append(float(row["bc"]))
            neum.append(float(row["neumann"]))
            smooth.append(float(row["smooth"]))
    for vals, label, color in [
        (total, "total", "tab:blue"), (pde, "pde", "tab:orange"),
        (iface, "iface", "tab:green"), (bc, "bc", "tab:red"),
        (neum, "neumann", "tab:purple"), (smooth, "smooth", "tab:brown"),
    ]:
        ax.plot(epochs, vals, linewidth=1.0, label=label, color=color)
    ax.set_yscale("symlog")
    ax.set_ylabel("loss")
    ax.set_title("1) 损失六分量")
    ax.legend(ncol=6, fontsize=8)
    ax.grid(True, alpha=0.3)

    # 行 2：学习率轨迹
    ax = axes[1]
    ax.plot(epochs, lr, linewidth=1.0, color="tab:cyan")
    ax.set_yscale("log")
    ax.set_ylabel("lr")
    ax.set_title("2) 学习率轨迹（loss_prop 随损失衰减）")
    ax.grid(True, alpha=0.3)

    # 行 3：P2 去噪过程（双轴）
    ax = axes[2]
    ax.plot(epochs, active_frac, linewidth=1.0, color="tab:green", label="active_frac")
    ax.set_ylabel("bc_active_frac")
    ax.set_ylim(0, 1.05)
    ax.set_title("3) P2 边界去噪过程")
    ax.grid(True, alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(epochs, n_outliers, linewidth=0.8, color="tab:red", label="n_outliers", alpha=0.6)
    ax2.set_ylabel("bc_n_outliers（当轮剔除）", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper right")

    # 行 4：训练速度（EMA 平滑）
    ax = axes[3]
    ax.plot(epochs, seconds, linewidth=0.6, color="tab:gray", alpha=0.5, label="raw")
    # 简单 EMA（α=0.1）
    ema = [seconds[0]]
    for s in seconds[1:]:
        ema.append(0.1 * s + 0.9 * ema[-1])
    ax.plot(epochs, ema, linewidth=1.5, color="tab:blue", label="EMA(α=0.1)")
    ax.set_ylabel("seconds/epoch")
    ax.set_xlabel("epoch")
    ax.set_title("4) 训练速度")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle("训练成长面板（Train History 16 列全景）")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 图 3：4 区域子图（仅对 ablation=full 有意义）
# ============================================================
def plot_per_region(model, X, Y, T_pred, ds: ThermalDataset, out_path: str):
    """4 个子图：A/B/C/D 每个区域的预测"""
    rid_np = region_id(torch.as_tensor(X), torch.as_tensor(Y)).numpy()
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    titles = ["A (top-left)", "B (top-right)", "C (bottom-left)", "D (bottom-right)"]
    extent = [ds.spec.x_min, ds.spec.x_max, ds.spec.y_min, ds.spec.y_max]
    vmin, vmax = T_pred.min(), T_pred.max()
    for i, ax in enumerate(axes.flat):
        mask = (rid_np == i)
        # 把非本区域 NaN 化便于可视化
        T_show = np.where(mask, T_pred, np.nan)
        im = ax.imshow(T_show, extent=extent, origin="lower", cmap="inferno", vmin=vmin, vmax=vmax)
        ax.set_title(f"Region {titles[i]}")
        ax.set_xlabel("x"); ax.set_ylabel("y")
        plt.colorbar(im, ax=ax)
    fig.suptitle("Per-region predictions (4 MLPs)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 消融对比柱状图
# ============================================================
def plot_ablation_compare(checkpoints: List[str], labels: List[str],
                          ds: ThermalDataset, out_path: str, device):
    e2_list = []
    maxerr_list = []
    for ckpt_path in checkpoints:
        model, _ = load_model(ckpt_path, device)
        _, _, T_pred, T_true = predict_on_grid(model, ds, N=200, device=device)
        err = T_pred - T_true
        e2 = float(np.sqrt(np.mean(err ** 2)) / (np.abs(T_true).max() + 1e-9))
        e2_list.append(e2)
        maxerr_list.append(float(np.abs(err).max()))

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(labels))
    ax.bar(x, [e * 100 for e in e2_list], color=["#e74c3c", "#f39c12", "#27ae60"][:len(labels)])
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("E₂  relative error (%)")
    ax.set_title("Ablation: domain decomposition effect on accuracy")
    for i, v in enumerate(e2_list):
        ax.text(i, v * 100 + 0.05, f"{v * 100:.2f}%", ha="center", fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return e2_list, maxerr_list


# ============================================================
# 主入口
# ============================================================
def main():
    args = parse_args()
    device = torch.device("cpu")
    dtype = torch.float64
    torch.set_default_dtype(dtype)

    os.makedirs(args.out_dir, exist_ok=True)

    # 加载数据集
    ds = ThermalDataset(npz_path=args.data, device=device, dtype=dtype)

    if args.compare:
        # 对比模式
        # 过滤空字符串（Windows 下 args.compare 可能因换行+续行符解析出空 token）
        compare_paths = [p for p in args.compare if p and p.strip() and p != "\\"]
        compare_labels = (
            [l for l in (args.compare_labels or []) if l and l.strip()]
            or [os.path.basename(p).replace(".pt", "") for p in compare_paths]
        )
        if not compare_paths:
            print("[ERROR] --compare 没有有效路径")
            return
        out_compare = os.path.join(args.out_dir, "ablation_compare.png")
        e2, maxerr = plot_ablation_compare(compare_paths, compare_labels, ds, out_compare, device)
        print("\n=== 消融对比 ===")
        for lbl, e2v, mxv in zip(compare_labels, e2, maxerr):
            print(f"  {lbl:>20s}  E₂={e2v * 100:.3f}%  max|err|={mxv:.4f}")
        print(f"图已保存：{out_compare}")
        return

    # 单 checkpoint 模式
    print(f"加载 checkpoint: {args.checkpoint}")
    model, state = load_model(args.checkpoint, device, dtype)
    print(f"  ablation={state.get('ablation', '?')}  best_loss={state.get('best_loss', '?'):.4e}"
          if isinstance(state.get('best_loss'), float) else f"  ablation={state.get('ablation', '?')}")

    # 预测
    print("在 200×200 网格上推理 ...")
    X, Y, T_pred, T_true = predict_on_grid(model, ds, N=200, device=device)

    # 图 1：预测 vs 真值
    out1 = os.path.join(args.out_dir, "pred_vs_true_heatmap.png")
    max_err, e2 = plot_pred_vs_true(X, Y, T_pred, T_true, ds, out1)
    print(f"  max|err|={max_err:.4f}  RMSE={e2:.4f}")
    print(f"  保存: {out1}")

    # 图 2：损失曲线（v0.9：CSV 优先 checkpoint 同目录，回退旧路径）
    csv_path = os.path.join(os.path.dirname(args.checkpoint) or ".", "train_history.csv")
    if not os.path.exists(csv_path):
        csv_path = "logs/train_history.csv"  # 回退旧显式路径训练的历史 CSV
    out2 = os.path.join(args.out_dir, "loss_curves.png")
    plot_loss_curves(csv_path, out2)
    if os.path.exists(out2):
        print(f"  保存: {out2}")

    # 图 2b（v0.9）：完整成长面板（损失/LR/去噪/速度）
    out2b = os.path.join(args.out_dir, "training_growth_panel.png")
    plot_training_growth(csv_path, out2b)
    if os.path.exists(out2b):
        print(f"  保存: {out2b}")

    # 图 3：4 区域子图
    if state.get("ablation", "full") == "full":
        out3 = os.path.join(args.out_dir, "per_region_2x2.png")
        plot_per_region(model, X, Y, T_pred, ds, out3)
        print(f"  保存: {out3}")


if __name__ == "__main__":
    main()