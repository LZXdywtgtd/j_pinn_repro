"""P2 Z-score 边界去噪单元测试（论文 §3.3 Eq.19-20）

测试目标（不依赖真实训练；直接用 BoundaryOutlierTracker）：
1. test_ema_update_basic — 10 点 1 个 spike，5 次更新后 spike EMA 主导
2. test_z_score_masks_outliers — 100 点 5 个高 EMA，burn-in 后被 mask
3. test_burnin_suppresses_mask — epoch < burnin 时全 True
4. test_min_active_per_edge — 某边全被 mask 时保留 Z 最低的 min_active 个
5. test_lambda_bc_normalization — L_bc = L_bc_raw × N_total / N_active 代数验证
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from outlier import BoundaryOutlierTracker, OutlierConfig


def _make_tracker(burnin: int = 100, delta: float = 3.0, alpha: float = 0.5,
                  min_active: int = 5, n_edges: int = 4) -> BoundaryOutlierTracker:
    cfg = OutlierConfig(
        enabled=True, burnin_epochs=burnin, delta=delta,
        ema_alpha=alpha, min_active_per_edge=min_active, n_edges=n_edges,
    )
    return BoundaryOutlierTracker(cfg)


def _make_batch(n_points: int = 100, n_edges: int = 4):
    """构造边界 batch：均匀分布在 [0, 0.01] 网格 + edge_id"""
    rng = np.random.default_rng(0)
    x = torch.tensor(np.linspace(0.0, 0.01, n_points), dtype=torch.float64)
    y = torch.zeros(n_points, dtype=torch.float64)
    edge_id = torch.arange(n_points) % n_edges
    return x, y, edge_id


def test_ema_update_basic():
    """10 点 1 个 spike，5 次更新后 spike EMA 应显著高于其余"""
    tracker = _make_tracker(burnin=0)
    x, y, edge_id = _make_batch(n_points=10, n_edges=1)
    # 5 次更新，每次点 5 有 spike
    for _ in range(5):
        res = torch.zeros(10, dtype=torch.float64)
        res[5] = 100.0
        tracker.update(res, x, y, edge_id, epoch=1)
    # spike 点的 EMA > 普通点 EMA
    spike_ema = tracker.ema_map[(0, tracker._slot(0.0, 0.0) + 5 * 10000)] if False else None
    # 直接验证：spike 对应的 slot EMA 应最高
    emas = list(tracker.ema_map.values())
    max_ema = max(emas)
    mean_others = sum(e for e in emas if e < max_ema) / max(len(emas) - 1, 1)
    assert max_ema > mean_others * 5, f"spike EMA 应显著高于其余：max={max_ema}, mean_others={mean_others}"
    print(f"  ✓ EMA spike 主导：max={max_ema:.1f}, mean_others={mean_others:.1f}")


def test_z_score_masks_outliers():
    """100 点 5 个高 EMA，burn-in 后被 mask"""
    tracker = _make_tracker(burnin=5, delta=3.0)
    x, y, edge_id = _make_batch(n_points=100, n_edges=4)
    # 5 个 outlier 点：idx 10, 30, 50, 70, 90（残差 100，其余 1）
    outlier_idx = {10, 30, 50, 70, 90}
    # 10 次更新
    for _ in range(10):
        res = torch.ones(100, dtype=torch.float64)
        for i in outlier_idx:
            res[i] = 100.0
        tracker.update(res, x, y, edge_id, epoch=10)
    # 第 10 epoch（burn-in 后）→ active mask
    res = torch.ones(100, dtype=torch.float64)
    for i in outlier_idx:
        res[i] = 100.0
    mask = tracker.update(res, x, y, edge_id, epoch=10)
    # outlier 点应被 mask（active=False），其余 active=True
    for i in range(100):
        if i in outlier_idx:
            assert not mask[i], f"outlier 点 {i} 应被 mask"
        else:
            assert mask[i], f"正常点 {i} 应保持 active"
    n_outlier = int((~mask).sum().item())
    print(f"  ✓ Z-score mask：{n_outlier} 个 outlier 被移除")


def test_burnin_suppresses_mask():
    """epoch < burnin 时全 True（宽限期）"""
    tracker = _make_tracker(burnin=100)
    x, y, edge_id = _make_batch(n_points=50)
    res = torch.ones(50, dtype=torch.float64) * 100.0  # 全 spike
    mask = tracker.update(res, x, y, edge_id, epoch=50)
    assert bool(mask.all().item()), "burn-in 期间应全 True"
    print(f"  ✓ burn-in 期间全 True（epoch=50 < burnin=100）")


def test_min_active_per_edge():
    """某边全被 mask 时保留 Z 最低的 min_active 个"""
    tracker = _make_tracker(burnin=5, min_active=3)
    x, y, edge_id = _make_batch(n_points=40, n_edges=4)
    # 让 edge 0 的点（idx 0,4,8,12,...）全高残差（都被 mask）
    for _ in range(10):
        res = torch.ones(40, dtype=torch.float64)
        for i in range(0, 40, 4):  # edge 0 的 idx：0,4,8,...,36
            res[i] = 100.0
        tracker.update(res, x, y, edge_id, epoch=10)
    mask = tracker.update(
        torch.ones(40, dtype=torch.float64) * 100.0, x, y, edge_id, epoch=10
    )
    # edge 0 至少保留 3 个 active（min_active_per_edge 兜底）
    edge0_active = sum(1 for i in range(0, 40, 4) if mask[i])
    assert edge0_active >= 3, f"edge 0 应至少保留 {3} 个，实际 {edge0_active}"
    print(f"  ✓ min_active 兜底：edge 0 保留 {edge0_active} 个（>=3）")


def test_lambda_bc_normalization():
    """L_bc = L_bc_raw × N_total / N_active 代数验证"""
    # 模拟：N_total=100，N_active=80（20 个 outlier）
    # L_bc_raw = 0.5（active 点平均 loss），L_bc = 0.5 × 100/80 = 0.625
    L_bc_raw = 0.5
    N_total, N_active = 100, 80
    L_bc = L_bc_raw * (N_total / N_active)
    assert abs(L_bc - 0.625) < 1e-10, f"L_bc={L_bc}"
    print(f"  ✓ |A| 归一化：L_bc={L_bc}（= 0.5 × 100/80）")


def main():
    print("\n=== P2 Z-score 边界去噪单元测试 ===\n")
    tests = [
        ("test_ema_update_basic", test_ema_update_basic),
        ("test_z_score_masks_outliers", test_z_score_masks_outliers),
        ("test_burnin_suppresses_mask", test_burnin_suppresses_mask),
        ("test_min_active_per_edge", test_min_active_per_edge),
        ("test_lambda_bc_normalization", test_lambda_bc_normalization),
    ]
    for name, fn in tests:
        print(f"[{name}]")
        fn()
        print()
    print("=== ALL TESTS PASSED ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())