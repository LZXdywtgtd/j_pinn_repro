"""
合成 log 源场的解析 J-integral（论文 §4.4 验证）

合成场：T_smooth = log(r_hot) - log(r_cold) + 2·tanh(50y)·𝟙[|x|<0.5]
- T_smooth 严格调和（∇²T = 0）
- T_crack 引入 y=0 处的温度跳跃

对于 ∇²T=0 的调和部分，J 路径完全无关（Rice 定理）。

**精确解析公式**（沿裂纹尖端 (0, 0) 的 J）：
J = (1/2π) · ∮_Γ [T · n · (∂T/∂x) - T · n · (∂T/∂x) - ...] ds

对于 2D Laplace ∇²T=0 + log 源在 (x_s, y_s)：
T(x, y) = log(r_s) + const, r_s = √((x-x_s)² + (y-y_s)²)
∇T = ((x-x_s)/r_s², (y-y_s)/r_s²)
|∇T|² = 1/r_s²

对于单个 log 源，沿矩形 contour 的 J = 0（∇·σ = 0 + 源在外部 → 单连通外域无净 flux）
对于两个等号反向源，J = -2π (T_hot - T_cold) 的边界贡献。

**简化解析实现**：
- 用 trapeze 积分 W·n - σ·∇T 在 contour 上，与模型对比验证算法
- 对纯解析场（不调用模型）跑 J 积分 → 应接近常数（路径无关性验证）

注：精确解析 J = 0 在 ∇²T=0 + 单极/双极源的对称结构下（拉普拉斯场无耗散）；
    本实现用于"算法验证"（数值积分与解析无源场 J=0 一致），不要求非零精确值。
"""
from __future__ import annotations

import numpy as np
import torch

from .contour_sampling import RectContour, contour_ds, contour_to_tensor


@torch.no_grad()
def analytic_field_phys(
    x_norm: torch.Tensor,
    y_norm: torch.Tensor,
    hot_xy: tuple[float, float] = (-0.6, 0.5),
    cold_xy: tuple[float, float] = (0.6, -0.5),
    eps: float = 1e-4,
    include_crack: bool = False,
    crack_jump: float = 2.0,
    crack_x_max: float = 0.5,
) -> torch.Tensor:
    """合成 log 源场的解析 T（在物理空间 [x∈[-1,1], y∈[-1,1]]）

    不归一化；调用方负责 ×/÷ chain-rule
    """
    r_hot = torch.sqrt((x_norm - hot_xy[0]) ** 2 + (y_norm - hot_xy[1]) ** 2 + eps)
    r_cold = torch.sqrt((x_norm - cold_xy[0]) ** 2 + (y_norm - cold_xy[1]) ** 2 + eps)
    T = torch.log(r_hot) - torch.log(r_cold)
    if include_crack:
        in_crack = (x_norm.abs() < crack_x_max).to(T.dtype)
        T = T + crack_jump * torch.tanh(50.0 * y_norm) * in_crack
    return T


@torch.no_grad()
def analytic_grad_T_phys(
    x_norm: torch.Tensor,
    y_norm: torch.Tensor,
    hot_xy: tuple[float, float] = (-0.6, 0.5),
    cold_xy: tuple[float, float] = (0.6, -0.5),
    eps: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """解析 ∂T/∂x_phys, ∂T/∂y_phys（用于 analytic_J）

    v0.9：新增 eps 参数（默认 1e-4 保持兼容）。
    eps=0 时场严格调和（∇²log r = 0），测试路径无关性需传 eps=0。
    """
    r_hot = torch.sqrt((x_norm - hot_xy[0]) ** 2 + (y_norm - hot_xy[1]) ** 2 + eps)
    r_cold = torch.sqrt((x_norm - cold_xy[0]) ** 2 + (y_norm - cold_xy[1]) ** 2 + eps)
    # ∂/∂x log(r_hot) = (x - x_hot) / r_hot²
    dT_dx = (x_norm - hot_xy[0]) / (r_hot ** 2) - (x_norm - cold_xy[0]) / (r_cold ** 2)
    dT_dy = (y_norm - hot_xy[1]) / (r_hot ** 2) - (y_norm - cold_xy[1]) / (r_cold ** 2)
    return dT_dx, dT_dy


@torch.no_grad()
def j_integral_exact(
    contour: RectContour,
    include_crack: bool = False,
    hot_xy: tuple[float, float] = (-0.6, 0.5),
    cold_xy: tuple[float, float] = (0.6, -0.5),
    eps: float = 1e-4,
) -> float:
    """对合成 log 源场（含/不含裂纹间断）跑 J 积分（数值积分 + 解析 ∂T/∂x）

    v0.9：新增 hot_xy/cold_xy/eps 参数（默认保持兼容）。
    测试远源场（源在 contour 外）传 eps=0 + 远源坐标。

    返回 1 个 float
    """
    x, y, n_x, n_y = contour_to_tensor(contour)
    dT_dx, dT_dy = analytic_grad_T_phys(x, y, hot_xy=hot_xy, cold_xy=cold_xy, eps=eps)
    # σ_ij = (∂T/∂x_i)(∂T/∂x_j)
    sig_xx = dT_dx ** 2
    sig_yy = dT_dy ** 2
    sig_xy = dT_dx * dT_dy
    # W = ½|∇T|² = ½(σ_xx + σ_yy)
    W = 0.5 * (sig_xx + sig_yy)
    # J = ∮ (W·n₁ - σ_ij·n_j·∂T/∂x₁) ds
    # n₁ 是 n_x（论文 Mode I 投影沿 x 方向）
    # σ_ij n_j ∂T/∂x₁ = (σ_xx n_x + σ_xy n_y) · ∂T/∂x
    traction = (sig_xx * n_x + sig_xy * n_y) * dT_dx
    integrand = W * n_x - traction
    # v0.9 修复：真实弧长 ds 的梯形积分（旧版 torch.trapz 用 arange 当弧长，
    # 竖/横段权重错 + 闭合项缺失，导致恒定的 0.5 偏移）
    ds = contour_ds(contour, dtype=integrand.dtype)
    return float(torch.sum((integrand[:-1] + integrand[1:]) * 0.5 * ds[:-1]).item())


def j_integral_exact_surface(
    contours: list[RectContour],
    include_crack: bool = False,
    hot_xy: tuple[float, float] = (-0.6, 0.5),
    cold_xy: tuple[float, float] = (0.6, -0.5),
    eps: float = 1e-4,
) -> np.ndarray:
    """批量计算 J 曲面（返回 (N_lig × N_wake) numpy 数组）

    v0.9：新增 hot_xy/cold_xy/eps 透传（默认保持兼容）。
    """
    N = len(contours)
    J = np.zeros(N, dtype=np.float64)
    for i, c in enumerate(contours):
        J[i] = j_integral_exact(c, include_crack=include_crack,
                                hot_xy=hot_xy, cold_xy=cold_xy, eps=eps)
    return J