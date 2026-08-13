"""P17 损失比例 LR 调度器单元测试（论文 §3.4）

测试目标：
1. test_loss_prop_monotonic — loss 减小 → lr 单调下降
2. test_loss_prop_clamp_lr_min — lr 不降到 lr_min 以下
3. test_loss_prop_cap_base — loss 增大 → lr 最多到 base_lr
4. test_loss_prop_first_step_sets_ref — 首次 step 自动设 loss_ref
5. test_loss_prop_state_roundtrip — state_dict/load_state_dict 保存恢复
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from schedulers import LossProportionalLR


def _make_sched(lr: float = 1e-3, lr_min: float = 1e-6):
    model = torch.nn.Linear(4, 4)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = LossProportionalLR(optimizer, loss_ref=None, lr_min=lr_min)
    return optimizer, scheduler


def test_loss_prop_first_step_sets_ref():
    """首次 step 自动设 loss_ref"""
    opt, sched = _make_sched()
    sched.step(loss=0.5)
    assert sched.loss_ref == 0.5, f"loss_ref 应=0.5，实际 {sched.loss_ref}"
    print(f"  ✓ 首次 step 自动设 loss_ref=0.5")


def test_loss_prop_monotonic():
    """loss 减小 → lr 单调下降"""
    opt, sched = _make_sched(lr=1e-3)
    losses = [1.0, 0.8, 0.6, 0.4, 0.2]
    lrs = []
    for loss in losses:
        sched.step(loss=loss)
        lrs.append(opt.param_groups[0]["lr"])
    # 单调不增
    assert all(lrs[i] >= lrs[i + 1] for i in range(len(lrs) - 1)), f"lr 应单调下降：{lrs}"
    print(f"  ✓ lr 单调下降：{[f'{lr:.2e}' for lr in lrs]}")


def test_loss_prop_clamp_lr_min():
    """lr 不降到 lr_min 以下"""
    opt, sched = _make_sched(lr=1e-3, lr_min=1e-6)
    sched.step(loss=1.0)  # 设 ref=1.0
    # 极端小 loss
    for _ in range(5):
        sched.step(loss=1e-12)
    lr = opt.param_groups[0]["lr"]
    assert lr >= 1e-6 - 1e-12, f"lr={lr} 应 >= lr_min=1e-6"
    print(f"  ✓ lr clamp 到 lr_min：lr={lr:.2e}")


def test_loss_prop_cap_base():
    """loss 增大 → lr 最多到 base_lr"""
    opt, sched = _make_sched(lr=1e-3)
    sched.step(loss=1.0)  # 设 ref=1.0
    # 巨大 loss → ratio 应 cap 到 1.0
    sched.step(loss=10.0)
    lr = opt.param_groups[0]["lr"]
    assert abs(lr - 1e-3) < 1e-12, f"lr={lr} 应 cap 到 base_lr=1e-3"
    print(f"  ✓ lr cap 到 base_lr：lr={lr:.2e}")


def test_loss_prop_state_roundtrip():
    """state_dict/load_state_dict 保存恢复"""
    opt, sched = _make_sched(lr=1e-3, lr_min=1e-6)
    sched.step(loss=0.7)
    state = sched.state_dict()
    # 新建 scheduler 恢复
    opt2 = torch.optim.Adam(torch.nn.Linear(4, 4).parameters(), lr=1e-3)
    sched2 = LossProportionalLR(opt2, loss_ref=None, lr_min=1e-6)
    sched2.load_state_dict(state)
    assert sched2.loss_ref == 0.7, f"恢复后 loss_ref={sched2.loss_ref}"
    assert sched2.lr_min == 1e-6, f"恢复后 lr_min={sched2.lr_min}"
    print(f"  ✓ state roundtrip：loss_ref={sched2.loss_ref}")


def main():
    print("\n=== P17 损失比例 LR 调度器单元测试 ===\n")
    tests = [
        ("test_loss_prop_first_step_sets_ref", test_loss_prop_first_step_sets_ref),
        ("test_loss_prop_monotonic", test_loss_prop_monotonic),
        ("test_loss_prop_clamp_lr_min", test_loss_prop_clamp_lr_min),
        ("test_loss_prop_cap_base", test_loss_prop_cap_base),
        ("test_loss_prop_state_roundtrip", test_loss_prop_state_roundtrip),
    ]
    for name, fn in tests:
        print(f"[{name}]")
        fn()
        print()
    print("=== ALL TESTS PASSED ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())