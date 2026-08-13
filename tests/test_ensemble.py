"""P9 多 restart 集成单元测试

测试目标：
1. test_average_identical — 2 个相同 checkpoint 平均返回相同参数
2. test_average_different — 2 个不同 seed 返回均值
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from run_ensemble import average_models


def _make_ckpt(path: Path, seed: int):
    """创建一个简单 checkpoint（含 model_state_dict）"""
    model = torch.nn.Linear(4, 4)
    torch.manual_seed(seed)
    state = {
        "model_state_dict": model.state_dict(),
        "epoch": 10,
        "ablation": "full",
    }
    torch.save(state, str(path))
    return state


def test_average_identical(tmp_path):
    """2 个相同 checkpoint 平均 = 原参数"""
    ckpt1 = tmp_path / "a.pt"
    ckpt2 = tmp_path / "b.pt"
    s1 = _make_ckpt(ckpt1, seed=42)
    # 用相同参数做 ckpt2
    torch.save(s1, str(ckpt2))
    out = tmp_path / "avg.pt"
    merged = average_models([ckpt1, ckpt2], str(out))
    for key in s1["model_state_dict"]:
        assert torch.allclose(merged["model_state_dict"][key], s1["model_state_dict"][key]), \
            f"平均应等于原参数（{key}）"
    print(f"  ✓ 相同 checkpoint 平均 = 原参数")


def test_average_different(tmp_path):
    """2 个不同 seed 平均 = 均值"""
    ckpt1 = tmp_path / "a.pt"
    ckpt2 = tmp_path / "b.pt"
    s1 = _make_ckpt(ckpt1, seed=42)
    s2 = _make_ckpt(ckpt2, seed=43)
    out = tmp_path / "avg.pt"
    merged = average_models([ckpt1, ckpt2], str(out))
    for key in s1["model_state_dict"]:
        expected = (s1["model_state_dict"][key] + s2["model_state_dict"][key]) / 2
        assert torch.allclose(merged["model_state_dict"][key], expected, atol=1e-7), \
            f"平均应等于均值（{key}）"
    print(f"  ✓ 不同 seed 平均 = 参数均值")


def main():
    print("\n=== P9 多 restart 集成单元测试 ===\n")
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        print("[test_average_identical]")
        test_average_identical(Path(td))
        print()
    with tempfile.TemporaryDirectory() as td:
        print("[test_average_different]")
        test_average_different(Path(td))
        print()
    print("=== ALL TESTS PASSED ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())