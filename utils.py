"""
工具模块：区域掩码、配点采样、归一化。

设计要点（对应论文 §2.3 / §4.5）：
- region_masks: 把坐标 (x, y) 映射到 4 区域 {0=A, 1=B, 2=C, 3=D}
- sample_interior: 每个区域内拉丁超立方采样
- sample_boundary: 4 条外边
- sample_interface: 3 条缝合边（每条返回镜像两点对）
- sample_crack: 裂纹段上/下两侧（强制 Neumann 跳跃）
- normalize / denormalize: [-1, 1] 归一化 + 链式法则反函数（保留物理导数）
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from scipy.stats import qmc


# ============================================================
# 域描述（与 data/generate_synthetic_thermal_data.py 同步）
# ============================================================
@dataclass(frozen=True)
class DomainSpec:
    x_min: float = -1.0
    x_max: float = 1.0
    y_min: float = -1.0
    y_max: float = 1.0
    crack_x_min: float = -0.5
    crack_x_max: float = 0.5
    crack_y_loc: float = 0.0
    crack_offset: float = 1e-3  # 镜像采样偏移（缝合边两侧微距 ε）


DEFAULT_DOMAIN = DomainSpec()


# ============================================================
# 区域映射
# ============================================================
def region_id(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    论文 §2.3 象限划分：
      A (0): x<0, y>=0
      B (1): x>=0, y>=0
      C (2): x<0, y<0
      D (3): x>=0, y<0
    """
    rid = torch.zeros_like(x, dtype=torch.long)
    rid[(x >= 0) & (y >= 0)] = 1
    rid[(x < 0) & (y < 0)] = 2
    rid[(x >= 0) & (y < 0)] = 3
    return rid


# ============================================================
# 归一化（保留导数链式法则反函数）
# ============================================================
def normalize_to_unit(v: torch.Tensor, v_min: float, v_max: float) -> torch.Tensor:
    """Min-max → [-1, 1]"""
    return 2.0 * (v - v_min) / (v_max - v_min) - 1.0


def denormalize_from_unit(v_norm: torch.Tensor, v_min: float, v_max: float) -> torch.Tensor:
    """[-1, 1] → 物理值"""
    return 0.5 * (v_norm + 1.0) * (v_max - v_min) + v_min


def normalize_derivative_chain(v_max: float, v_min: float) -> float:
    """链式法则缩放系数：d(physical)/d(normalized) = (v_max - v_min) / 2"""
    return 0.5 * (v_max - v_min)


# ============================================================
# 内部配点采样（拉丁超立方）
# ============================================================
def sample_interior(
    n_per_region: int,
    spec: DomainSpec = DEFAULT_DOMAIN,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
    seed: int | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    在 4 个区域内分别用拉丁超立方采样。
    返回:
      x: (4*n_per_region,)
      y: (4*n_per_region,)
      region_id: (4*n_per_region,)
    """
    rng = np.random.default_rng(seed)
    half = spec.x_max - spec.x_min  # 不假设对称，容错
    pieces = []
    bounds_list = [
        (spec.x_min, 0.0, 0.0, spec.y_max),       # A
        (0.0, spec.x_max, 0.0, spec.y_max),       # B
        (spec.x_min, 0.0, spec.y_min, 0.0),       # C
        (0.0, spec.x_max, spec.y_min, 0.0),       # D
    ]
    for rid, (xlo, xhi, ylo, yhi) in enumerate(bounds_list):
        # Latin Hypercube: 在 [0,1]^2 上
        sampler = qmc.LatinHypercube(d=2, seed=int(rng.integers(0, 1 << 31)))
        u01 = sampler.random(n=n_per_region)  # (N, 2) in [0,1]^2
        xs = xlo + u01[:, 0] * (xhi - xlo)
        ys = ylo + u01[:, 1] * (yhi - ylo)
        rids = np.full(n_per_region, rid, dtype=np.int64)
        pieces.append(np.stack([xs, ys, rids], axis=1))

    arr = np.concatenate(pieces, axis=0)
    x = torch.as_tensor(arr[:, 0], dtype=dtype, device=device)
    y = torch.as_tensor(arr[:, 1], dtype=dtype, device=device)
    rid = torch.as_tensor(arr[:, 2], dtype=torch.long, device=device)
    return x, y, rid


# ============================================================
# 边界采样（4 条外边）
# ============================================================
def sample_boundary(
    n_per_edge: int,
    spec: DomainSpec = DEFAULT_DOMAIN,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
    seed: int | None = None,
    exclude_crack_overlap: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    在 4 条外边均匀采样。
    返回:
      x: (4*n_per_edge,)
      y: (4*n_per_edge,)
      edge_id: (4*n_per_edge,) ∈ {0=左, 1=右, 2=下, 3=上}
    exclude_crack_overlap: 排除与裂纹段重叠的点（y=0, |x|<=0.5 区域已被 4 区域 BC 覆盖，避免重复）
    """
    rng = np.random.default_rng(seed)
    edges = []
    # 左 x=x_min
    yt = rng.uniform(spec.y_min, spec.y_max, size=n_per_edge)
    edges.append(np.stack([np.full(n_per_edge, spec.x_min), yt, np.zeros(n_per_edge)], axis=1))
    # 右 x=x_max
    yt = rng.uniform(spec.y_min, spec.y_max, size=n_per_edge)
    edges.append(np.stack([np.full(n_per_edge, spec.x_max), yt, np.ones(n_per_edge)], axis=1))
    # 下 y=y_min
    xt = rng.uniform(spec.x_min, spec.x_max, size=n_per_edge)
    edges.append(np.stack([xt, np.full(n_per_edge, spec.y_min), np.full(n_per_edge, 2)], axis=1))
    # 上 y=y_max
    xt = rng.uniform(spec.x_min, spec.x_max, size=n_per_edge)
    edges.append(np.stack([xt, np.full(n_per_edge, spec.y_max), np.full(n_per_edge, 3)], axis=1))

    arr = np.concatenate(edges, axis=0)
    if exclude_crack_overlap:
        # 外边界 y=y_min / y=y_max 上的裂纹段在外部，因裂纹 y=0；不影响。BC 在外边采样完整。
        pass
    x = torch.as_tensor(arr[:, 0], dtype=dtype, device=device)
    y = torch.as_tensor(arr[:, 1], dtype=dtype, device=device)
    edge_id = torch.as_tensor(arr[:, 2], dtype=torch.long, device=device)
    return x, y, edge_id


# ============================================================
# 接口采样（3 条缝合边，每条返回镜像两点对）
# ============================================================
def sample_interface(
    n_per_seam: int,
    spec: DomainSpec = DEFAULT_DOMAIN,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
    seed: int | None = None,
) -> dict:
    """
    返回 dict，包含 3 个 seam：
      "A_B": (x_left, y_left, rid_left, x_right, y_right, rid_right)
      "A_C": 同上
      "B_D": 同上
    每条 seam 长度 = n_per_seam
    镜像偏移 epsilon = spec.crack_offset（防止两坐标完全重合时 autograd 数值爆炸）
    """
    rng = np.random.default_rng(seed)
    eps = spec.crack_offset
    out = {}

    # A↔B: 垂直缝合 x=0, y∈[0,1]；A 在 x=-eps，B 在 x=+eps
    yt = rng.uniform(0.0, spec.y_max, size=n_per_seam)
    out["A_B"] = (
        torch.full((n_per_seam,), -eps, dtype=dtype, device=device),
        torch.as_tensor(yt, dtype=dtype, device=device),
        torch.zeros(n_per_seam, dtype=torch.long, device=device),  # rid=0 → A
        torch.full((n_per_seam,), +eps, dtype=dtype, device=device),
        torch.as_tensor(yt, dtype=dtype, device=device),
        torch.ones(n_per_seam, dtype=torch.long, device=device),  # rid=1 → B
    )

    # A↔C: 水平缝合 y=0, x∈[-1, -0.5]；A 在 y=+eps，C 在 y=-eps
    xt = rng.uniform(spec.crack_x_min - eps, -eps, size=n_per_seam)
    if xt.size > 0:
        pass  # 占位（实际区间 [-1, -0.5]）
    xt = rng.uniform(spec.x_min, spec.crack_x_min, size=n_per_seam)
    out["A_C"] = (
        torch.as_tensor(xt, dtype=dtype, device=device),
        torch.full((n_per_seam,), +eps, dtype=dtype, device=device),
        torch.zeros(n_per_seam, dtype=torch.long, device=device),  # A
        torch.as_tensor(xt, dtype=dtype, device=device),
        torch.full((n_per_seam,), -eps, dtype=dtype, device=device),
        torch.full((n_per_seam,), 2, dtype=torch.long, device=device),  # C
    )

    # B↔D: 水平缝合 y=0, x∈[0.5, 1]；B 在 y=+eps，D 在 y=-eps
    xt = rng.uniform(spec.crack_x_max, spec.x_max, size=n_per_seam)
    out["B_D"] = (
        torch.as_tensor(xt, dtype=dtype, device=device),
        torch.full((n_per_seam,), +eps, dtype=dtype, device=device),
        torch.ones(n_per_seam, dtype=torch.long, device=device),  # B
        torch.as_tensor(xt, dtype=dtype, device=device),
        torch.full((n_per_seam,), -eps, dtype=dtype, device=device),
        torch.full((n_per_seam,), 3, dtype=torch.long, device=device),  # D
    )
    return out


# ============================================================
# 裂纹段采样（不缝合；强制 Neumann 跳跃）
# ============================================================
def sample_crack(
    n_per_side: int,
    spec: DomainSpec = DEFAULT_DOMAIN,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
    seed: int | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    裂纹段 x∈[crack_x_min, crack_x_max], y=0 两侧采样。
    返回:
      (x_top, y_top, rid_top, x_bot, y_bot, rid_bot)
    其中：
      top: y=+eps → B (rid=1)
      bot: y=-eps → D (rid=3)
    """
    rng = np.random.default_rng(seed)
    eps = spec.crack_offset
    xt = rng.uniform(spec.crack_x_min, spec.crack_x_max, size=n_per_side)

    x_top = torch.as_tensor(xt, dtype=dtype, device=device)
    y_top = torch.full((n_per_side,), +eps, dtype=dtype, device=device)
    rid_top = torch.ones(n_per_side, dtype=torch.long, device=device)

    x_bot = torch.as_tensor(xt, dtype=dtype, device=device)
    y_bot = torch.full((n_per_side,), -eps, dtype=dtype, device=device)
    rid_bot = torch.full((n_per_side,), 3, dtype=torch.long, device=device)

    return x_top, y_top, rid_top, x_bot, y_bot, rid_bot


# ============================================================
# 真值函数（从合成模块取，便于训练时评估）
# ============================================================
def T_exact_torch(
    x: torch.Tensor,
    y: torch.Tensor,
    include_crack: bool = True,
    eps: float = 1e-4,
    hot_xy: Tuple[float, float] = (-0.6, 0.5),
    cold_xy: Tuple[float, float] = (0.6, -0.5),
    crack_x_max: float = 0.5,
    crack_jump: float = 2.0,
    crack_steepness: float = 50.0,
) -> torch.Tensor:
    """对 generate_synthetic_thermal_data.py 解析场的 torch 版本。"""
    r_hot = torch.sqrt((x - hot_xy[0]) ** 2 + (y - hot_xy[1]) ** 2 + eps)
    r_cold = torch.sqrt((x - cold_xy[0]) ** 2 + (y - cold_xy[1]) ** 2 + eps)
    T = torch.log(r_hot) - torch.log(r_cold)
    if include_crack:
        in_crack = (x.abs() < crack_x_max).to(T.dtype)
        T = T + crack_jump * torch.tanh(crack_steepness * y) * in_crack
    return T