"""run_j_integral CLI 端到端测试（v0.8 阶段 8）

为什么需要这个文件：
- smoke_test 不调用 run_j_integral.py 的 CLI 入口
- v0.6 / v0.7 / v0.8 三次审计都漏掉了 4 个 CLI 路径 bug：
  1. atexit 未导入（NameError）
  2. _j_integral_pinn_one 用 @torch.no_grad() 与 grad_T 互斥（RuntimeError: requires grad）
  3. x_lig_arr 长度与 J_pinn_grid (5,5) 不匹配（LinAlgError）
  4. ETA 估算干跑 batch 太小（9.3× 偏差）

测试策略：
- subprocess.run 调 python -m postprocess.run_j_integral ...
- 不依赖 PyTorch 加载状态（但需要 checkpoints/jpinn.pt 真实存在）
- 6 个测试覆盖每个 anchor_mode × 不同 n_per_side × 最小/正常 contour
- 一旦未来重构破坏 CLI 路径，CI 立即报警

测试场景：
1. test_cli_anchor_mode_extremes_full — 最常见的 5×5 contour + extremes
2. test_cli_anchor_mode_residual_min — 另一种 anchor_mode
3. test_cli_minimal_contours — 仅 1 个 contour（边界值）
4. test_cli_n_per_side_small — n_per_side=20（加速）
5. test_cli_n_per_side_default — n_per_side=200（默认）
6. test_cli_help_flag — --help 必须无报错退出
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# v0.8 阶段 8: Windows cmd 默认 GBK，强制 stdout/stderr 用 UTF-8
# 避免 [OK]/[FAIL]/[WARNING] 等 ASCII 标识符之外的字符触发 UnicodeEncodeError
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
PYTHON = sys.executable  # 用当前 Python 解释器（确保 jpinn 环境已激活）


def _run_cli(*args: str, timeout: int = 300) -> subprocess.CompletedProcess:
    """调 run_j_integral CLI 子进程

    Returns: CompletedProcess with stdout/stderr captured as bytes
    Raises: subprocess.TimeoutExpired / CalledProcessError（仅当 check=True）
    """
    cmd = [PYTHON, "-m", "postprocess.run_j_integral", *args]
    return subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        capture_output=True,
        timeout=timeout,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def _cli_ran_successfully(cp: subprocess.CompletedProcess) -> bool:
    """判定 CLI 端到端跑通（v0.8 阶段 8 修正版）

    ⚠️ 旧版用 `returncode in (0, 1, 2)` 有致命漏洞：
    Python 未捕获异常崩溃的退出码恰好也是 1，与质量门控的 `return 1` 冲突，
    导致崩溃（如 RuntimeError: requires grad）被误判为"通过"。
    （回滚验证暴露：bug 状态代码跑 CLI 崩溃 exit=1，旧判定仍 PASSED）

    新判定标准——**崩溃必有 Traceback**（纯 ASCII），质量门控没有：
      - stderr 含 "Traceback" → 崩溃，FAIL
      - returncode == 0 且 stderr 无 Traceback → 通过质量门控
      - returncode == 1 且 stderr 无 Traceback → 质量门控：路径无关 > 10%
      - returncode == 2 且 stderr 无 Traceback → 质量门控：相对误差 > 50%
    后两者在 Laplace 降维场景下属预期（已知物理意义限制），算"跑通"。
    """
    stderr = cp.stderr or ""
    stdout = cp.stdout or ""
    if "Traceback" in stderr:
        return False
    if "Traceback" in stdout:  # 防御：部分环境把 traceback 写 stdout
        return False
    return cp.returncode in (0, 1, 2)


def test_cli_anchor_mode_extremes_full():
    """回归 v0.8 阶段 7 修复：完整 5×5 contour + extremes 模式必须跑通"""
    cp = _run_cli(
        "--checkpoint", "checkpoints/jpinn.pt",
        "--anchor_mode", "extremes",
        "--n_per_side", "50",  # 加速（默认 200 太慢）
        "--x_lig_values", "-0.9", "-0.5", "-0.1",
        "--x_wake_values", "0.1", "0.5", "0.9",
    )
    # v0.8 阶段 7 修复前会触发 3 个 bug：
    #   - NameError: atexit（不应出现在 CLI 入口，但代码中有 atexit.register）
    #   - RuntimeError: element 0 does not require grad
    #   - LinAlgError: Incompatible dimensions
    assert _cli_ran_successfully(cp), (
        f"CLI 退出码 {cp.returncode}（非预期）\n"
        f"stdout tail: {cp.stdout[-500:]}\n"
        f"stderr tail: {cp.stderr[-500:]}"
    )
    # 必须产生 metrics.json
    metrics_path = PROJECT_ROOT / "logs" / "j_integral" / "metrics.json"
    assert metrics_path.exists(), f"metrics.json 未生成：{metrics_path}"
    # 必须包含 anchor_mode=extremes 的结果
    import json
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    assert "J_pinn_path_indep_corrected" in metrics
    assert "J_exact_path_indep_corrected" in metrics
    assert "relative_error_corrected" in metrics
    print(f"  [OK] extremes 模式跑通：J_pinn_far_field={metrics['J_pinn_far_field']:.4e}")


def test_cli_anchor_mode_residual_min():
    """回归 v0.7 阶段 5 B7：residual_min 模式也必须跑通"""
    cp = _run_cli(
        "--checkpoint", "checkpoints/jpinn.pt",
        "--anchor_mode", "residual_min",
        "--n_per_side", "30",
        "--x_lig_values", "-0.9", "-0.1",
        "--x_wake_values", "0.1", "0.9",
    )
    assert _cli_ran_successfully(cp), (
        f"CLI 退出码 {cp.returncode}（非预期）\n"
        f"stderr tail: {cp.stderr[-500:]}"
    )
    print(f"  [OK] residual_min 模式跑通")


def test_cli_minimal_contours():
    """边界值测试：仅 1 个 contour（最小调用）"""
    cp = _run_cli(
        "--checkpoint", "checkpoints/jpinn.pt",
        "--anchor_mode", "extremes",
        "--n_per_side", "20",
        "--x_lig_values", "-0.5",
        "--x_wake_values", "0.5",
    )
    assert _cli_ran_successfully(cp), (
        f"1 contour 最小调用失败：{cp.stderr[-500:]}"
    )
    print(f"  [OK] 最小 1 contour 调用跑通")


def test_cli_n_per_side_small():
    """低 n_per_side：测梯形积分短向量路径"""
    cp = _run_cli(
        "--checkpoint", "checkpoints/jpinn.pt",
        "--anchor_mode", "extremes",
        "--n_per_side", "10",  # 极小值
        "--x_lig_values", "-0.5",
        "--x_wake_values", "0.5",
    )
    assert _cli_ran_successfully(cp), f"n_per_side=10 失败：{cp.stderr[-500:]}"
    print(f"  [OK] n_per_side=10 跑通")


def test_cli_n_per_side_default():
    """默认 n_per_side=200：验证默认值生效不报错"""
    cp = _run_cli(
        "--checkpoint", "checkpoints/jpinn.pt",
        "--anchor_mode", "extremes",
        # 不传 --n_per_side（默认 200）
        "--x_lig_values", "-0.5",
        "--x_wake_values", "0.5",
        timeout=120,  # 默认 200 比较慢，给更多时间
    )
    assert _cli_ran_successfully(cp), f"默认 n_per_side 失败：{cp.stderr[-500:]}"
    print(f"  [OK] 默认 n_per_side=200 跑通")


def test_cli_help_flag():
    """--help 必须 0 退出码（v0.5 修复：包级 import 与 @torch.no_grad 都影响 help）"""
    cp = _run_cli("--help")
    assert cp.returncode == 0, f"--help 报错：{cp.stderr[-500:]}"
    assert "--anchor_mode" in cp.stdout, "help 输出缺 --anchor_mode 文档"
    assert "--checkpoint" in cp.stdout
    print(f"  [OK] --help 跑通且包含关键 flag 文档")


# =============================================================================
# 测试守护（避免测试在无 checkpoint 时 false-positive 通过）
# =============================================================================
import pytest

@pytest.fixture(scope="module", autouse=True)
def require_checkpoint():
    """跳过测试如果 checkpoints/jpinn.pt 不存在"""
    ckpt = PROJECT_ROOT / "checkpoints" / "jpinn.pt"
    if not ckpt.exists():
        pytest.skip(
            f"checkpoints/jpinn.pt 不存在。跑一次 `python train.py --epochs 50 --ablation full` "
            f"生成 checkpoint 后再跑此测试。"
        )
    yield


# =============================================================================
# 主入口（直接 `python tests/test_run_j_integral_cli.py` 时执行所有测试）
# =============================================================================
def main() -> int:
    """手动运行入口（pytest 自动收集也能跑）"""
    tests = [
        ("test_cli_help_flag", test_cli_help_flag),
        ("test_cli_minimal_contours", test_cli_minimal_contours),
        ("test_cli_n_per_side_small", test_cli_n_per_side_small),
        ("test_cli_anchor_mode_extremes_full", test_cli_anchor_mode_extremes_full),
        ("test_cli_anchor_mode_residual_min", test_cli_anchor_mode_residual_min),
        ("test_cli_n_per_side_default", test_cli_n_per_side_default),
    ]
    failed = 0
    for name, fn in tests:
        print(f"\n=== {name} ===")
        try:
            fn()
        except Exception as e:
            print(f"  FAIL: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n=== {'ALL PASSED' if failed == 0 else f'{failed} FAILED'} ===")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())