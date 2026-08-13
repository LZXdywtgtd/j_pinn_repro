"""
J-integral 数值积分（论文 §4.4 Eq.9-10 降维）

J = ∮_Γ [ W·n₁ - σ_ij·s^i₁ · ∂u_j/∂x₁ ] ds  (Mode I, ligament side)

在 Laplace 降维下：
- u → T
- σ_ij → (∂T/∂x_i)(∂T/∂x_j)
- s^i₁ = n_i = (n_x, n_y) 外法向
- ∂u_j/∂x₁ → ∂T/∂x（沿 ligament 方向 x）

J = ∮_Γ [ W·n_x - (σ_xx·n_x + σ_xy·n_y)·∂T/∂x - (σ_xy·n_x + σ_yy·n_y)·∂T/∂y ] ds

简化（论文 §4.4 J_I 部分）：用 ligament 方向（n_x）作为投影轴

实际上 J 是标量，不依赖方向选择。我们实现论文完整 2D 公式（含两个分量）。
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn

from analytic_J import analytic_grad_T_phys, j_integral_exact
from contour_sampling import RectContour, contour_to_tensor
from stress_from_T import grad_T_physical, strain_energy_W, stress_analog


@torch.no_grad()
def _j_integral_pinn_one(
    model: nn.Module,
    contour: RectContour,
    T_min: float,
    T_max: float,
    x_min: float, x_max: float,
    y_min: float, y_max: float,
) -> float:
    """单条 contour 的 J 积分（基于 PINN 输出）

    步骤：
    1. 采样 contour → (x_norm, y_norm, n_x, n_y) 4 个数组
    2. region_id 推断（每点）
    3. JPINN.forward(x_norm, y_norm, rid) → T_norm
    4. autograd 求 ∂T/∂x_norm, ∂T/∂y_norm → 链式法则 → ∂T/∂x_phys, ∂T/∂y_phys
    5. σ_ij = 梯度积，W = ½|∇T|²
    6. 4 段分别梯形积分 J = ∮ (W·n - σ·∇T) ds
    """
    from utils import region_id  # 延迟导入避免循环

    x_norm, y_norm, n_x, n_y = contour_to_tensor(contour)
    # region_id 需要 requires_grad=False
    rid = region_id(x_norm.detach(), y_norm.detach())
    # autograd 求导
    dT_dx, dT_dy = grad_T_physical(
        model, x_norm, y_norm, rid,
        T_max=T_max, T_min=T_min,
        x_min=x_min, x_max=x_max,
        y_min=y_min, y_max=y_max,
    )
    sig_xx, sig_yy, sig_xy = stress_analog(dT_dx, dT_dy)
    W = strain_energy_W(sig_xx, sig_yy)
    # 论文 J = ∮ (W·n₁ - σ_ij·n_j·∂u_i/∂x₁) ds（沿 ligament 方向 x₁）
    # i 索引 u_i = T，j 索引 n_j = (n_x, n_y)，x₁ = x 方向
    # σ_ij·n_j = (σ_xx·n_x + σ_xy·n_y, σ_xy·n_x + σ_yy·n_y)
    # 点积 (σ_ij·n_j)·∂T/∂x_1 = σ_ij·n_j·∂T/∂x
    sigma_dot_n = sig_xx * n_x + sig_xy * n_y
    traction_x = sigma_dot_n * dT_dx
    integrand = W * n_x - traction_x
    # 梯形积分
    return float(torch.trapz(integrand, torch.arange(len(integrand), dtype=torch.float64)).item())


def j_integral_one_contour(
    model: nn.Module,
    contour: RectContour,
    T_min: float,
    T_max: float,
    spec=None,
) -> float:
    """对外接口：单条 contour 的 J 积分

    spec: DomainSpec 或 None；若 None，从 contour 默认 y_min/y_max 推断
    """
    if spec is None:
        x_min, x_max = contour.x_lig, contour.x_wake
        # 取 contour 边界（更稳）
        x_min = min(x_min, contour.x_lig, contour.x_wake)
        x_max = max(x_max, contour.x_lig, contour.x_wake)
        y_min, y_max = contour.y_min, contour.y_max
    else:
        x_min, x_max = spec.x_min, spec.x_max
        y_min, y_max = spec.y_min, spec.y_max
    return _j_integral_pinn_one(
        model, contour, T_min, T_max, x_min, x_max, y_min, y_max,
    )


def j_integral_surface(
    model: nn.Module,
    contours: List[RectContour],
    T_min: float,
    T_max: float,
    spec=None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """批量：返回 (x_lig_grid, x_wake_grid, J_grid)

    J_grid 形状 (N_lig × N_wake)，flatten 后可与 contours 索引对齐
    """
    J = np.zeros(len(contours), dtype=np.float64)
    x_ligs = np.zeros(len(contours), dtype=np.float64)
    x_wakes = np.zeros(len(contours), dtype=np.float64)
    for i, c in enumerate(contours):
        J[i] = j_integral_one_contour(model, c, T_min, T_max, spec)
        x_ligs[i] = c.x_lig
        x_wakes[i] = c.x_wake
    # reshape 到 (N_lig × N_wake)
    unique_lig = sorted(set(x_ligs.tolist()))
    unique_wake = sorted(set(x_wakes.tolist()))
    if len(unique_lig) * len(unique_wake) != len(contours):
        # 不规则网格，返回扁平
        return x_ligs, x_wakes, J
    J_grid = J.reshape(len(unique_lig), len(unique_wake))
    return np.array(unique_lig), np.array(unique_wake), J_grid


def j_integral_exact_for_surface(
    contours: List[RectContour],
    include_crack: bool = False,
) -> np.ndarray:
    """对 contour 列表跑解析 J（用于对比 PINN 输出）"""
    J = np.zeros(len(contours), dtype=np.float64)
    for i, c in enumerate(contours):
        J[i] = j_integral_exact(c, include_crack=include_crack)
    return J