"""
P9 多 restart 集成：多个随机种子训练 + 参数平均（方差降低）

用法：
    python run_ensemble.py --n_restarts 3 --epochs 100 --average
    python run_ensemble.py --n_restarts 3 --seed-base 100 --no-average

设计：
- 复用 run_ablations.py 的 subprocess 模式（每个 seed 独立 train.py 进程）
- 固定架构 → 直接参数平均（无需 permutation 对齐）
- average_models 加载每个 model_state_dict，torch.stack 平均
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import torch

CURRENT_DIR = Path(__file__).resolve().parent


def run_restarts(
    n_seeds: int,
    base_args: dict,
    cwd: Path,
    seed_base: int = 0,
) -> list[Path]:
    """循环 seeds 跑 train.py，返回 checkpoint 路径列表"""
    ckpt_paths = []
    for i in range(n_seeds):
        seed = seed_base + i
        out = cwd / "checkpoints" / f"seed_{seed}.pt"
        log = cwd / "logs" / f"seed_{seed}.csv"
        cmd = [sys.executable, "train.py", "--seed", str(seed),
               "--out", str(out), "--log", str(log)]
        for k, v in base_args.items():
            cmd += [str(k), str(v)]
        print(f"\n[restart {i+1}/{n_seeds}] seed={seed}: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=str(cwd))
        if result.returncode != 0:
            print(f"[WARN] seed {seed} 训练失败（rc={result.returncode}）")
            continue
        ckpt_paths.append(out)
    return ckpt_paths


def average_models(ckpt_paths: list[Path], out_path: str) -> dict:
    """参数平均：加载每个 model_state_dict，torch.stack 平均"""
    if not ckpt_paths:
        raise ValueError("无可用 checkpoint 进行平均")

    states = []
    for p in ckpt_paths:
        if not p.exists():
            raise FileNotFoundError(f"checkpoint 不存在: {p}")
        state = torch.load(p, map_location="cpu", weights_only=False)
        states.append(state["model_state_dict"])

    # 平均（保持原 dtype，float64 精度不损失）
    avg_state = {}
    for key in states[0].keys():
        tensors = torch.stack([s[key] for s in states])
        avg_state[key] = tensors.mean(dim=0)

    # 保存 merged（保留第一个 checkpoint 的元数据）
    merged = dict(states[0])
    merged["model_state_dict"] = avg_state
    merged["epoch"] = states[0].get("epoch", 0)
    merged["n_restarts"] = len(states)
    merged["seed_list"] = [p.stem for p in ckpt_paths]
    out_path = str(out_path)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, out_path)
    print(f"平均 {len(states)} 个模型 → {out_path}")
    return merged


def main():
    p = argparse.ArgumentParser(description="P9 多 restart 集成")
    p.add_argument("--n_restarts", type=int, default=3, help="restart 次数")
    p.add_argument("--seed-base", type=int, default=0, help="起始 seed")
    p.add_argument("--epochs", type=int, default=100, help="每次 restart 的 epochs")
    p.add_argument("--average", action="store_true", help="训练后做参数平均")
    p.add_argument("--out", type=str, default="checkpoints/ensemble_avg.pt",
                   help="平均后 checkpoint 路径")
    p.add_argument("--ablation", type=str, default="full",
                   choices=["full", "two", "single"])
    # 透传额外 train.py 参数（如 --scheduler loss_prop）
    p.add_argument("--extra-args", nargs=argparse.REMAINDER, default=[],
                   help="追加到每个 train.py 的额外参数")
    args = p.parse_args()

    base_args = {
        "--epochs": args.epochs,
        "--ablation": args.ablation,
    }
    # 额外参数（--extra-args 后面的 k v）
    extra = list(args.extra_args)
    for i in range(0, len(extra) - 1, 2):
        base_args[extra[i]] = extra[i + 1]

    ckpts = run_restarts(args.n_restarts, base_args, CURRENT_DIR, seed_base=args.seed_base)

    if args.average and ckpts:
        average_models(ckpts, args.out)
    elif not ckpts:
        print("[ERROR] 所有 restart 都失败")
        return 1
    else:
        print(f"[INFO] 完成 {len(ckpts)} 个 restart（未平均）")
    return 0


if __name__ == "__main__":
    sys.exit(main())