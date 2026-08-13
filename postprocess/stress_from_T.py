"""
弹性类比应力 + 应变能密度（论文 §2.3 + 弹性张量 → Laplace 降维）

复用 losses.py:47-69 pde_residual 的 autograd 模式（一阶导）。
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


def grad_T(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    region_id: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """∂T/∂x, ∂T/∂y via autograd（在归一化空间）"""
    x = x.detach().requires_grad_(True)
    y = y.detach().requires_grad_(True)
    T = model(x, y, region_id)
    dT_dx = torch.autograd.grad(
        T.sum(), x, create_graph=False, retain_graph=False, allow_unused=False
    )[0]
    dT_dy = torch.autograd.grad(
        T.sum(), y, create_graph=False, retain_graph=False, allow_unused=False
    )[0]
    return dT_dx, dT_dy


def grad_T_physical(
    model: nn.Module,
    x_norm: torch.Tensor,
    y_norm: torch.Tensor,
    region_id: torch.Tensor,
    T_max: float,
    T_min: float,
    x_min: float, x_max: float,
    y_min: float, y_max: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """物理空间 ∂T/∂x_phys, ∂T/∂y_phys（链式法则：× (max-min)/2 / (max-min)/2）

    归一化 x∈[x_min, x_max] → x_norm∈[-1,1]：
      T_norm = 2(T - T_min)/(T_max - T_min) - 1
      x_norm = 2(x - x_min)/(x_max - x_min) - 1
      ∂T/∂x = ∂T/∂x_norm × ∂x_norm/∂x = ∂T/∂x_norm × 2/(x_max - x_min)
    """
    dT_dx_norm, dT_dy_norm = grad_T(model, x_norm, y_norm, region_id)
    chain_x = 2.0 / (x_max - x_min)
    chain_y = 2.0 / (y_max - y_min)
    chain_T = (T_max - T_min) / 2.0
    # ∂T/∂x_phys = (∂T_norm/∂x_norm) × chain_T × chain_x
    dT_dx_phys = dT_dx_norm * chain_T * chain_x
    dT_dy_phys = dT_dy_norm * chain_T * chain_y
    return dT_dx_phys, dT_dy_phys


def stress_analog(
    dT_dx: torch.Tensor,
    dT_dy: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """σ_xx, σ_yy, σ_xy = (∂T/∂x_i)(∂T/∂x_j)（对称 rank-2 张量）

    论文弹性 σ_ij = λ tr(ε) δ_ij + 2μ ε_ij → 这里类比为
    σ_ij = (∂T/∂x_i)(∂T/∂x_j)（无 λ/μ，纯几何类比）
    """
    sig_xx = dT_dx * dT_dx
    sig_yy = dT_dy * dT_dy
    sig_xy = dT_dx * dT_dy
    return sig_xx, sig_yy, sig_xy


def strain_energy_W(
    sig_xx: torch.Tensor,
    sig_yy: torch.Tensor,
) -> torch.Tensor:
    """W = ½(σ_xx + σ_yy)（von Mises 类比）

    注：完整 W = ½ σ_ij ε_ij，ε=  = ∂T/∂x_i；
        我们的类比 σ_ij ε_ij = σ_ij (∂T/∂x_j) = (∂T/∂x_i)(∂T/∂x_j)(∂T/∂x_j)
        不易降维，故用 von Mises 类比 ½|∇T|² = ½(σ_xx + σ_yy)
        （在 Laplace 场下与 ½(σ_xx + σ_yy + 2σ_xy) 数值接近，因 σ_xy 是交叉项）
    """
    return 0.5 * (sig_xx + sig_yy)