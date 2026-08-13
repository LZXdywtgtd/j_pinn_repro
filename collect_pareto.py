"""
P6 Pareto 收集器：读 sweep 各 CSV 的最终 best_loss + 参数量，输出 Pareto 前沿。

用法：
    python collect_pareto.py --sweep_dir logs/sweep --out logs/sweep/pareto.csv

输出列：arch, hidden, n_hidden_layers, params, best_loss, epoch, time_s
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path


def read_final_best_loss(csv_path: Path) -> dict:
    """读 CSV 最后一行非 header 数据，返回 dict"""
    if not csv_path.exists():
        return {}
    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    last = rows[-1]
    # best_loss 是 total 列的最小值，不是最后一行；这里用最后一行的 total 作为近似
    # 更精确：遍历所有行取 min total
    best = min(rows, key=lambda r: float(r.get("total", "inf")))
    return {
        "epoch": len(rows),
        "best_loss": float(best.get("total", "inf")),
        "time_s": float(last.get("seconds", "0")),
    }


def collect_pareto(sweep_dir: str, out_csv: str, n_params_map: dict | None = None) -> None:
    """遍历 sweep_dir 下所有 CSV，输出 Pareto 表格。

    参数量优先从 n_params_map（外部传入）取；否则尝试读同目录 checkpoint 的 n_params。
    """
    sweep_path = Path(sweep_dir)
    if not sweep_path.is_dir():
        print(f"[ERROR] sweep 目录不存在: {sweep_dir}")
        return

    import torch

    rows_out = []
    for csv_file in sorted(sweep_path.glob("*.csv")):
        arch = csv_file.stem  # 如 w8_d1
        info = read_final_best_loss(csv_file)
        if not info:
            continue
        # 解析 hidden / depth
        parts = arch.split("_")
        hidden = int(parts[0].lstrip("w")) if parts[0].startswith("w") else 0
        depth = int(parts[1].lstrip("d")) if len(parts) > 1 else 0
        # 参数量：优先外部 map，否则读 checkpoint
        params = 0
        if n_params_map and arch in n_params_map:
            params = n_params_map[arch]
        else:
            ckpt = Path("checkpoints") / "sweep" / f"{arch}.pt"
            if ckpt.exists():
                try:
                    st = torch.load(str(ckpt), map_location="cpu", weights_only=False)
                    params = int(st.get("n_params", 0))
                except Exception:
                    params = 0
        rows_out.append({
            "arch": arch, "hidden": hidden, "n_hidden_layers": depth,
            "params": params, "best_loss": info["best_loss"],
            "epoch": info["epoch"], "time_s": info["time_s"],
        })

    if not rows_out:
        print("[WARN] 无 CSV 可收集")
        return

    # 写 CSV
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["arch", "hidden", "n_hidden_layers", "params", "best_loss", "epoch", "time_s"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows_out:
            writer.writerow(r)

    print(f"已收集 {len(rows_out)} 个架构 → {out_csv}")
    # 打印 top 5（按 best_loss）
    top5 = sorted(rows_out, key=lambda r: r["best_loss"])[:5]
    print("\nTop 5（best_loss 升序）：")
    for r in top5:
        print(f"  {r['arch']:>10s}  params={r['params']:>6d}  best_loss={r['best_loss']:.4e}")


def main():
    p = argparse.ArgumentParser(description="P6 Pareto 收集器")
    p.add_argument("--sweep_dir", type=str, default="logs/sweep")
    p.add_argument("--out", type=str, default="logs/sweep/pareto.csv")
    args = p.parse_args()
    collect_pareto(args.sweep_dir, args.out)


if __name__ == "__main__":
    main()