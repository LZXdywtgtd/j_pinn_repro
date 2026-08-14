"""
P2 Z-score 边界去噪（论文 §3.3 Eq.19-20）

论文算法：
- 对每个外边界点，维护残差平方的 EMA
- 每 Γ=100 epoch（burn-in 后）计算 Z-score：Z_i = |EMA_i - μ_edge| / (σ_edge + ε)
- Z_i > δ=3.0 的点标记为 outlier，从 L_bc 排除
- L_bc 除以 |A| / N_total 归一化（论文 Table 5：88% 损失下降）

设计要点（v0.5）：
- `boundary_strategy="fixed"` 时外边界点固定（不随 epoch 重采样），mask 可持久
- EMA 按 (edge_id, point_slot) 索引（用坐标量化到 0.01 网格）
- `min_active_per_edge` 防止活跃集过空
- checkpoint 可保存/恢复 tracker 状态（续训用）

用法（在 LossAggregator.__call__ 中）：
    tracker = BoundaryOutlierTracker(OutlierConfig(), device=..., dtype=...)
    active_mask = tracker.update(residuals, bd["x"], bd["y"], bd["edge_id"], epoch)
    L_b = bc_loss_dirichlet(model, bd["x"][active_mask], ...) * (N_total / N_active)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import torch


def _np_median(values: list) -> float:
    """列表的中位数（用 numpy，纯 Python fallback）"""
    if not values:
        return 0.0
    return float(np.median(np.array(values)))


@dataclass
class OutlierConfig:
    """P2 边界 outlier 配置（论文 §3.3 Eq.19-20）"""

    enabled: bool = True
    burnin_epochs: int = 100            # Γ 宽限期；前 Γ 个 epoch 不更新 mask
    delta: float = 3.0                   # δ 阈值（Z-score 绝对值 > δ 视作 outlier）
    ema_alpha: float = 0.1               # EMA 平滑系数（residual² 累积）
    min_active_per_edge: int = 5         # 每条边最少保留点数（防活跃集过空）
    n_edges: int = 4                     # 外边界边数（top/bottom/left/right）


class BoundaryOutlierTracker:
    """
    跟踪每个外边界点的残差平方 EMA + Z-score + 活跃集。

    内部状态：
    - self.ema_map: Dict[(edge_id, slot_id), float]  # 残差平方 EMA
    - self.count_map: Dict[(edge_id, slot_id), int]  # 累积更新次数
    - self.active: Dict[(edge_id, slot_id), bool]    # 是否活跃

    slot_id = 坐标量化（x/y → 0.01 网格 index），使固定/重采样边界都可跟踪。
    """

    def __init__(
        self,
        cfg: OutlierConfig,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        self.cfg = cfg
        self.device = torch.device(device)
        self.dtype = dtype
        self.ema_map: Dict[tuple, float] = {}
        self.count_map: Dict[tuple, int] = {}
        self.active: Dict[tuple, bool] = {}
        self._grid = 0.01  # 坐标量化网格（与 xy_extent 0.01m 对齐）
        # slot 分辨率：0.01/1000 = 1e-5，保证边界点上 ~1000 个分辨级
        # （避免 100 个边界点坍缩到少量 slot 导致 Z-score 失效）
        self._slot_scale = 1000

    # ============================================================
    # 主接口
    # ============================================================
    def update(
        self,
        residuals: torch.Tensor,    # (N,) 残差平方 (T_pred - T_target)²
        x: torch.Tensor,
        y: torch.Tensor,
        edge_id: torch.Tensor,      # (N,)
        epoch: int,
    ) -> torch.Tensor:
        """返回 active_mask: (N,) bool。更新内部 EMA + 执行 Z-score 测试。"""
        if not self.cfg.enabled:
            return torch.ones(x.shape[0], dtype=torch.bool, device=x.device)

        res = residuals.detach().cpu().tolist()
        xs = x.detach().cpu().tolist()
        ys = y.detach().cpu().tolist()
        eds = edge_id.detach().cpu().tolist()

        # 更新 EMA（每个点 slot）
        n = len(res)
        for i in range(n):
            slot = self._slot(xs[i], ys[i])
            key = (int(eds[i]), slot)
            r = float(res[i])
            if key in self.ema_map:
                old = self.ema_map[key]
                self.ema_map[key] = (1 - self.cfg.ema_alpha) * old + self.cfg.ema_alpha * r
            else:
                self.ema_map[key] = r
            self.count_map[key] = self.count_map.get(key, 0) + 1
            self.active.setdefault(key, True)

        # burn-in 前：全 True
        if epoch < self.cfg.burnin_epochs:
            return torch.ones(n, dtype=torch.bool, device=x.device)

        # Z-score 测试（按 edge 分组）
        # 用中位数 + MAD（median absolute deviation）而非 mean/std——
        # 对 outlier 更鲁棒（outlier 拉高 σ 会导致 Z 阈值失效，论文场景常见）
        per_edge_ema: Dict[int, list] = {}
        for key, ema in self.ema_map.items():
            eid = key[0]
            per_edge_ema.setdefault(eid, []).append(ema)

        active_mask = [True] * n
        # 先按当前 batch 的点计算 Z（用已累积的 EMA）
        edge_stats: Dict[int, tuple] = {}
        for eid, emas in per_edge_ema.items():
            mu = float(_np_median(emas))
            # MAD = median(|x - median|)
            devs = [abs(e - mu) for e in emas]
            mad = float(_np_median(devs))
            # MAD → σ 估计（正态假设：σ ≈ 1.4826 * MAD）
            sigma = 1.4826 * mad if mad > 0 else 1e-8
            edge_stats[eid] = (mu, sigma)

        # 对每个点：Z_i = |EMA_i - μ_edge| / (σ_edge + ε)
        for i in range(n):
            key = (int(eds[i]), self._slot(xs[i], ys[i]))
            if key not in self.ema_map:
                continue
            ema = self.ema_map[key]
            mu, sigma = edge_stats[int(eds[i])]
            denom = sigma + 1e-8
            z = abs(ema - mu) / denom if denom > 0 else 0.0
            if z > self.cfg.delta:
                active_mask[i] = False

        # 每条边强制保留 min_active_per_edge 个（取 Z 最小的）
        self._enforce_min_active(active_mask, xs, ys, eds, edge_stats)

        return torch.tensor(active_mask, dtype=torch.bool, device=x.device)

    # ============================================================
    # 辅助
    # ============================================================
    def _slot(self, x: float, y: float) -> int:
        """坐标 → slot_id（量化到 self._slot_scale 级）
        x ∈ [-0.005, 0.005] → slot_x ∈ [0, self._slot_scale]
        """
        half = self._grid / 2  # 0.005
        sx = int(round((x + half) / self._grid * self._slot_scale))
        sy = int(round((y + half) / self._grid * self._slot_scale))
        return sx * (self._slot_scale + 1) + sy

    def _enforce_min_active(
        self, active_mask: list, xs: list, ys: list, eds: list, edge_stats: dict
    ) -> None:
        """每条边至少保留 min_active_per_edge 个点（取 Z 最小的）"""
        n = len(active_mask)
        per_edge_idx: Dict[int, list] = {}
        for i in range(n):
            eid = int(eds[i])
            per_edge_idx.setdefault(eid, []).append(i)

        for eid, idxs in per_edge_idx.items():
            # 当前这条边 active 的数量
            active_cnt = sum(1 for i in idxs if active_mask[i])
            if active_cnt >= self.cfg.min_active_per_edge:
                continue
            # 找出被 mask 的点，按 Z 从小到大，恢复前几个
            mu, sigma = edge_stats.get(eid, (0.0, 0.0))
            masked = [
                i for i in idxs if not active_mask[i]
            ]
            if not masked:
                continue
            # 按 Z 排序（Z 小的优先恢复）
            masked.sort(
                key=lambda i: abs(
                    self.ema_map.get((eid, self._slot(xs[i], ys[i])), 0.0) - mu
                ) / (sigma + 1e-8)
            )
            need = self.cfg.min_active_per_edge - active_cnt
            for i in masked[:need]:
                active_mask[i] = True

    # ============================================================
    # checkpoint
    # ============================================================
    def state_dict(self) -> dict:
        """保存 tracker 状态（续训用）"""
        return {
            "ema_map": self.ema_map,
            "count_map": self.count_map,
            "active": self.active,
            "cfg": {
                "burnin_epochs": self.cfg.burnin_epochs,
                "delta": self.cfg.delta,
                "ema_alpha": self.cfg.ema_alpha,
                "min_active_per_edge": self.cfg.min_active_per_edge,
            },
        }

    def load_state_dict(self, state: dict) -> None:
        """恢复 tracker 状态（续训用）；cfg 不匹配时警告 + 重置"""
        if state.get("cfg", {}).get("burnin_epochs") != self.cfg.burnin_epochs:
            import warnings
            warnings.warn("P2 OutlierTracker: burnin_epochs 与 checkpoint 不一致，重置状态")
            return
        self.ema_map = dict(state.get("ema_map", {}))
        self.count_map = dict(state.get("count_map", {}))
        self.active = dict(state.get("active", {}))


# ============================================================
# 辅助：把外边界残差传给 tracker（集成到 losses.py 用）
# ============================================================
def bc_residuals(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    region_id: torch.Tensor,
    t_target: torch.Tensor,
) -> torch.Tensor:
    """计算外边界每点残差平方 (T_pred - T_target)²（no_grad，用于 tracker）"""
    with torch.no_grad():
        t_pred = model(x, y, region_id)
        res = (t_pred - t_target) ** 2
    return res