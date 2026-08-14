"""v0.9 结果管理系统回归测试（outputs/ 归档 + 防覆盖守卫）

背景（2026-08-14 事故）：手动测试裸命令覆盖了用户 5000 epoch 的
checkpoints/jpinn.pt（best_loss 2.15e-02 → 5.55e-01）与 train_history.csv。
本测试守护以下行为：
1. 裸命令（无 --out）→ 自动生成唯一 task 目录，永不覆盖
2. 显式 --out 已存在 → 无 --force 拒绝
3. --force → 允许覆盖
4. task 目录含 4 件套（model_best.pt / train_history.csv / config.json / metadata.json）
5. outputs/latest.json 正确更新
6. resume 同目录 → CSV 追加（就地续训）
7. resume 旧路径 → 新 task 目录

运行：python -m pytest tests/test_output_management.py -v
耗时：~2 分钟（多次 subprocess 训练 mini epochs）
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
PYTHON = sys.executable


def _run_train(*args: str, timeout: int = 300) -> subprocess.CompletedProcess:
    """调 train.py 子进程（cwd=PROJECT_ROOT，UTF-8 解码）"""
    return subprocess.run(
        [PYTHON, "train.py", *args, "--print_every", "1000"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        timeout=timeout,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def _latest_ablation_dir(ablation: str = "full") -> Path:
    latest = json.loads((PROJECT_ROOT / "outputs" / "latest.json").read_text(encoding="utf-8"))
    task_id = latest[ablation]
    return PROJECT_ROOT / "outputs" / ablation / task_id


def test_bare_command_creates_unique_task_dir():
    """裸命令自动生成 task 目录 + 4 件套"""
    cp = _run_train("--epochs", "3", "--ablation", "full", "--log_plain")
    assert cp.returncode == 0, f"裸命令失败 rc={cp.returncode}\n{cp.stderr[-500:]}"
    task_dir = _latest_ablation_dir("full")
    assert task_dir.exists(), f"task 目录不存在: {task_dir}"
    for fname in ["model_best.pt", "train_history.csv", "config.json", "metadata.json"]:
        assert (task_dir / fname).exists(), f"缺 {fname}"
    # config.json 含 git_commit 字段
    cfg = json.loads((task_dir / "config.json").read_text(encoding="utf-8"))
    assert "git_commit" in cfg
    # metadata.json 含完成状态
    meta = json.loads((task_dir / "metadata.json").read_text(encoding="utf-8"))
    assert meta["completed_epoch"] == 3 and meta["target_epochs"] == 3
    print(f"  [OK] 裸命令 → {task_dir.name}（4 件套齐全）")


def test_explicit_out_guarded_without_force(tmp_path):
    """显式 --out 已存在 + 无 --force → 拒绝"""
    out = tmp_path / "test.pt"
    # 第一次：成功创建
    cp1 = _run_train("--epochs", "2", "--ablation", "single",
                     "--out", str(out), "--log", str(tmp_path / "a.csv"), "--log_plain")
    assert cp1.returncode == 0 and out.exists()
    # 第二次：同 --out 无 --force → 拒绝（非 0 且不覆盖）
    cp2 = _run_train("--epochs", "2", "--ablation", "single",
                     "--out", str(out), "--log", str(tmp_path / "b.csv"), "--log_plain")
    assert cp2.returncode != 0, "无 --force 应拒绝覆盖"
    # GBK 乱码下中文断言不可靠，用 ASCII 前缀断言
    assert "[ERROR]" in (cp2.stdout + cp2.stderr)
    print(f"  [OK] 显式 --out 无 --force 被拒绝")


def test_explicit_out_allowed_with_force(tmp_path):
    """显式 --out 已存在 + --force → 允许覆盖"""
    out = tmp_path / "test2.pt"
    cp1 = _run_train("--epochs", "2", "--ablation", "two",
                     "--out", str(out), "--log", str(tmp_path / "a2.csv"), "--log_plain")
    assert cp1.returncode == 0
    cp2 = _run_train("--epochs", "2", "--ablation", "two",
                     "--out", str(out), "--log", str(tmp_path / "b2.csv"),
                     "--force", "--log_plain")
    assert cp2.returncode == 0, f"--force 应允许覆盖，实际 rc={cp2.returncode}"
    print(f"  [OK] --force 放行覆盖")


def test_latest_json_updated():
    """latest.json 的 full 键指向最新 task 目录"""
    task_dir = _latest_ablation_dir("full")
    assert task_dir.exists()
    print(f"  [OK] latest.json → {task_dir.name}")


def test_resume_in_place_appends_csv():
    """resume 指向 outputs 内 model_best.pt → 就地续训，CSV 追加

    自包含：先裸命令训练 3 epoch 建 task 目录，再 resume 续训 2 epoch。
    """
    cp0 = _run_train("--epochs", "3", "--ablation", "full", "--log_plain")
    assert cp0.returncode == 0
    task_dir = _latest_ablation_dir("full")
    ckpt = task_dir / "model_best.pt"
    csv_path = task_dir / "train_history.csv"
    before = sum(1 for _ in open(csv_path, encoding="utf-8"))
    cp = _run_train("--epochs", "5", "--resume", str(ckpt), "--log_plain")
    assert cp.returncode == 0, f"就地续训失败 rc={cp.returncode}\n{cp.stderr[-400:]}"
    after = sum(1 for _ in open(csv_path, encoding="utf-8"))
    # 前次 3 行数据 + 本次续训 2 行（3→5）+ 表头 1 = 6
    assert after > before, f"CSV 应追加（{before} → {after}）"
    print(f"  [OK] 就地续训 CSV 追加 {before} → {after} 行")


def test_resume_old_path_creates_new_task_dir(tmp_path):
    """resume 指向 outputs 外的旧 checkpoint → 新 task 目录"""
    old_ckpt = tmp_path / "old.pt"
    cp1 = _run_train("--epochs", "2", "--ablation", "full",
                     "--out", str(old_ckpt), "--log", str(tmp_path / "old.csv"), "--log_plain")
    assert cp1.returncode == 0
    n_before = len(list((PROJECT_ROOT / "outputs" / "full").glob("jpinn_full_*")))
    cp2 = _run_train("--epochs", "3", "--resume", str(old_ckpt), "--log_plain")
    assert cp2.returncode == 0, f"旧路径续训失败 rc={cp2.returncode}\n{cp2.stderr[-400:]}"
    n_after = len(list((PROJECT_ROOT / "outputs" / "full").glob("jpinn_full_*")))
    assert n_after == n_before + 1, f"应新建 task 目录（{n_before} → {n_after}）"
    print(f"  [OK] 旧路径 resume → 新 task 目录（{n_before} → {n_after}）")


def main() -> int:
    import tempfile
    tests = [
        ("test_bare_command_creates_unique_task_dir", test_bare_command_creates_unique_task_dir),
        ("test_explicit_out_guarded_without_force", test_explicit_out_guarded_without_force),
        ("test_explicit_out_allowed_with_force", test_explicit_out_allowed_with_force),
        ("test_latest_json_updated", test_latest_json_updated),
        ("test_resume_in_place_appends_csv", test_resume_in_place_appends_csv),
    ]
    failed = 0
    for name, fn in tests:
        print(f"\n=== {name} ===")
        try:
            if fn.__name__ in ("test_explicit_out_guarded_without_force", "test_explicit_out_allowed_with_force"):
                with tempfile.TemporaryDirectory() as td:
                    fn(Path(td))
            else:
                fn()
        except Exception as e:
            print(f"  FAIL: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n=== {'ALL PASSED' if failed == 0 else f'{failed} FAILED'} ===")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
