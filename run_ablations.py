"""
消融实验编排器：按 JSON 配方表依次调 train.py 训练多个 ablation 模型

设计要点（仿 v4 team_train.py 简化版）：
- 读取 tasks/ablation_recipes.json 的 tasks 列表
- 按依赖关系（depends_on）拓扑排序
- 依次启动 train.py 子进程（不阻塞用户中断）
- 自动捕获每个 ablation 的 stdout/stderr 到独立 log
- 完成后自动调 visualize.py 生成 ablation_compare.png

使用：
  conda activate jpinn
  python run_ablations.py                          # 跑所有 ablation（默认 full → two → single）
  python run_ablations.py --only ablation-full-v0.2  # 只跑一个
  python run_ablations.py --dry-run                  # 只打印计划不执行
  python run_ablations.py --visualize-only          # 只对已有 checkpoint 生成对比图

设计决策：
- 用 subprocess.run 而非 os.system（v4 v4.6.10 修复：避免 Windows cmd 转义 bug）
- 单进程串行（CPU 训练，独占算力；并行意义不大）
- 子进程非阻塞 Ctrl-C（KeyboardInterrupt 透传到子进程）
- checkpoint 与 log 路径与 train.py 一致；本工具不引入新格式
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
TASKS_DIR = CURRENT_DIR / "tasks"
RECIPES_FILE = TASKS_DIR / "ablation_recipes.json"
DEFAULT_RECIPES = RECIPES_FILE


def load_recipes(path: Path = DEFAULT_RECIPES) -> dict:
    """加载 ablation 配方表 JSON"""
    if not path.exists():
        raise FileNotFoundError(
            f"配方表不存在: {path}\n"
            "请确保 tasks/ablation_recipes.json 存在"
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def topo_sort(tasks: list[dict]) -> list[dict]:
    """拓扑排序（按 depends_on）"""
    by_id = {t["id"]: t for t in tasks}
    visited: set[str] = set()
    order: list[dict] = []

    def visit(task_id: str) -> None:
        if task_id in visited:
            return
        if task_id not in by_id:
            raise KeyError(f"未知 task_id: {task_id}")
        task = by_id[task_id]
        for dep in task.get("depends_on", []):
            visit(dep)
        visited.add(task_id)
        order.append(task)

    for t in tasks:
        visit(t["id"])
    return order


def filter_tasks(tasks: list[dict], only_id: str | None) -> list[dict]:
    """按 --only 过滤"""
    if only_id is None:
        return tasks
    matched = [t for t in tasks if t["id"] == only_id]
    if not matched:
        raise ValueError(f"--only={only_id} 未匹配任何 task；可选: {[t['id'] for t in tasks]}")
    return matched


def run_one_ablation(task: dict, extra_args: list[str], cwd: Path) -> int:
    """启动 train.py 子进程跑一个 ablation；返回 returncode"""
    args_dict = task["args"]
    # 构造命令行：python train.py <flag1 val1> <flag2 val2> ...
    cmd = [sys.executable, "train.py"]
    for k, v in args_dict.items():
        cmd += [str(k), str(v)]
    cmd += extra_args

    print(f"\n{'=' * 70}")
    print(f"[{task['id']}] {task['name']}")
    print(f"  expected params: {task.get('expected_n_params', '?')}")
    print(f"  command: {' '.join(cmd)}")
    print(f"{'=' * 70}\n")

    t0 = time.time()
    try:
        # 实时透传 stdout/stderr（非阻塞 Ctrl-C）
        result = subprocess.run(cmd, cwd=str(cwd))
    except KeyboardInterrupt:
        print(f"\n[INTERRUPTED] {task['id']} 被用户中断（Ctrl-C）")
        return 130  # SIGINT
    elapsed = time.time() - t0
    print(f"\n[{task['id']}] 训练完成，耗时 {elapsed / 60:.2f} 分钟；returncode={result.returncode}")
    return result.returncode


def run_visualize(checkpoints: list[Path], labels: list[str], out_dir: Path) -> int:
    """对所有 checkpoint 跑 visualize.py --compare 生成对比图"""
    cmd = [
        sys.executable, "visualize.py",
        "--compare", *[str(p) for p in checkpoints],
        "--compare_labels", *labels,
    ]
    print(f"\n{'=' * 70}")
    print(f"[VISUALIZE] 生成 ablation 对比图")
    print(f"  command: {' '.join(cmd)}")
    print(f"{'=' * 70}\n")
    return subprocess.run(cmd, cwd=str(CURRENT_DIR)).returncode


def main() -> int:
    p = argparse.ArgumentParser(description="J-PINN 消融实验编排器")
    p.add_argument(
        "--recipes",
        type=str,
        default=str(DEFAULT_RECIPES),
        help="配方表 JSON 路径（默认 tasks/ablation_recipes.json）",
    )
    p.add_argument(
        "--only",
        type=str,
        default=None,
        help="只跑指定 ablation（如 ablation-full-v0.2）",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印计划不执行",
    )
    p.add_argument(
        "--visualize-only",
        action="store_true",
        help="只对已有 checkpoint 生成 ablation_compare.png，跳过训练",
    )
    p.add_argument(
        "--extra-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="追加到每个 train.py 的额外参数（放在 ablation args 之后）",
    )
    args = p.parse_args()

    recipes = load_recipes(Path(args.recipes))
    all_tasks = recipes["tasks"]
    tasks_to_run = filter_tasks(topo_sort(all_tasks), args.only)

    if args.dry_run:
        print(f"将依次执行 {len(tasks_to_run)} 个 ablation：")
        for i, t in enumerate(tasks_to_run, 1):
            print(f"  {i}. [{t['id']}] {t['name']} → {t['args'].get('--out', '?')}")
        print(f"\n总预计时长（CPU）：单 ~155 min + 双 ~55 min + 4 区域 ~65 min ≈ 4.5 小时")
        return 0

    if not args.visualize_only:
        print(f"\n{'#' * 70}")
        print(f"# J-PINN 消融编排器（共 {len(tasks_to_run)} 个 ablation）")
        print(f"# 配方表: {args.recipes}")
        print(f"{'#' * 70}")
        all_failed = []
        for task in tasks_to_run:
            rc = run_one_ablation(task, args.extra_args, CURRENT_DIR)
            if rc != 0:
                all_failed.append((task["id"], rc))
                print(f"[ERROR] {task['id']} 失败（rc={rc}），是否继续下一个？默认：继续")
                # 不中断，让后续 ablation 也能跑
        if all_failed:
            print(f"\n[WARNING] {len(all_failed)} 个 ablation 失败：{all_failed}")

    # 收集 checkpoint 用于可视化
    checkpoints = []
    labels = []
    for t in tasks_to_run:
        ckpt = Path(t["args"].get("--out", ""))
        if ckpt.exists():
            checkpoints.append(ckpt)
            labels.append(t["args"]["--ablation"])

    if len(checkpoints) >= 2:
        rc = run_visualize(checkpoints, labels, CURRENT_DIR / "logs" / "figures")
        if rc != 0:
            print(f"[WARNING] visualize.py 返回非零 ({rc})")
    elif len(checkpoints) < 2:
        print(f"[INFO] 仅 {len(checkpoints)} 个 checkpoint，跳过 ablation_compare")

    print(f"\n{'=' * 70}")
    print(f"# 消融编排完成")
    print(f"{'=' * 70}")
    return 0


if __name__ == "__main__":
    sys.exit(main())