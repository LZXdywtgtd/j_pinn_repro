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

⚠️⚠️⚠️ 物理意义限制（v0.7 阶段 4 / B5+C2）⚠️⚠️⚠️

Mode I/II 分解基于 Rigby-Aliabadi 对称/反对称应力分解（论文 §2.2 Eq.9-10），
本项目 Laplace ∇²T=0 降维后存在重大物理意义降级：

| 维度 | 论文 | 本项目 |
|------|------|--------|
| 场类型 | 位移场 u(x,y)（向量）| 温度场 T(x,y)（标量）|
| Mode 定义 | 法向位移+切向零应力 / 切向位移+法向零应力 | 标量场无方向性断裂概念 |
| 裂纹面行为 | 法向张开 / 切向滑动 | 仅温度数值跳变，无运动方向 |
| 应力分解 | σ_ij 由 Hooke 律 σ_ij = λ tr(ε) δ_ij + 2μ ε_ij | σ_ij = ∇T·∇T^T（纯几何外积，无物理含义）|

热场没有"裂纹模式"概念：本函数 J_I / J_II 数值不应解读为 Mode 贡献，
仅作 Rigby-Aliabadi 分解算法的**数学正确性验证** + **未来反转 ADR-0001 时的复用**。

反转条件（ADR-0001 §11）：
- 获得论文原版 DIC 全场位移数据
- 切换 Navier-Cauchy PDE（恢复混合二阶导 + Lamé 参数）
- 届时 J_I / J_II 数值将恢复物理意义

详见 docs/DECISIONS/0001-laplace-substitute.md §10.1.2
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


def j_integral_mode_decomposed(
    model: nn.Module,
    contour: RectContour,
    T_min: float,
    T_max: float,
    spec=None,
) -> Tuple[float, float]:
    """Rigby-Aliabadi 对称/反对称 Mode I/II 分解（论文 §2.2 Eq.9-10）

    ┌──────────────────────────────────────────────────────────────────┐
    │ ⚠️⚠️⚠️ 热场降维下的物理意义限制（v0.7 B5+C2）⚠️⚠️⚠️                │
    │                                                                  │
    │ 论文 Mode I/II 描述弹性力学裂纹模式：                              │
    │   Mode I（张开型）：裂纹面法向被拉开（σ_22 拉伸）                  │
    │   Mode II（滑开型）：裂纹面切向相对滑动（σ_12 剪切）              │
    │                                                                  │
    │ 本项目 Laplace ∇²T=0 降维后：                                     │
    │   - 标量温度场没有"方向性断裂模式"概念                             │
    │   - σ_ij = ∇T·∇T^T 是几何外积（不是 Hooke 律应力）                │
    │   - 本函数 J_I / J_II 数值不应解读为 Mode 贡献                     │
    │                                                                  │
    │ 本函数保留用途：                                                   │
    │   1. 验证 Rigby-Aliabadi 分解算法的数学正确性（正交性、路径无关）  │
    │   2. 准备未来反转 ADR-0001 时直接复用（拿到 DIC 后）              │
    │                                                                  │
    │ 反转条件（ADR-0001 §11）：                                         │
    │   - 获得论文原版 DIC 全场位移数据                                  │
    │   - 切换 Navier-Cauchy PDE（恢复混合二阶导 + Lamé 参数）           │
    │   - 届时 J_I / J_II 数值将恢复物理意义                             │
    └──────────────────────────────────────────────────────────────────┘

    算法（Rigby-Aliabadi 对称/反对称分解）：
    1. 同时评估 T(x,y) 与 T(x,-y)（裂纹面对称）
       T_sym = 0.5 * (T(x,y) + T(x,-y))
       T_asym = 0.5 * (T(x,y) - T(x,-y))
    2. 对 T_sym 求梯度 → σ_sym = ∇T_sym · ∇T_sym^T
       对 T_asym 求梯度 → σ_asym = ∇T_asym · ∇T_asym^T
    3. J_I = ∮ (W_sym·n_x - σ_sym·n·∂T_asym/∂x) ds
       J_II = ∮ (W_asym·n_x - σ_asym·n·∂T_sym/∂x) ds
    4. 交叉项正交抵消（数学基础）：J ≈ J_I + J_II

    Args:
        model: PINN 模型（接受 x, y, region_id → T）
        contour: 矩形轮廓
        T_min, T_max: 温度归一化边界
        spec: DomainSpec 或 None

    Returns:
        (J_I, J_II): Mode I / Mode II J-integral 分量
    """
    # 延迟导入 region_id；既兼容包级调用（from postprocess.j_integral）也兼容
    # 旧的 sys.path 注入式调用（v0.3 历史遗留导入路径）
    try:
        from utils import region_id  # 旧路径
    except ImportError:
        from ..utils import region_id  # 包级相对路径

    if spec is None:
        x_min, x_max = contour.x_lig, contour.x_wake
        x_min = min(x_min, contour.x_lig, contour.x_wake)
        x_max = max(x_max, contour.x_lig, contour.x_wake)
        y_min, y_max = contour.y_min, contour.y_max
    else:
        x_min, x_max = spec.x_min, spec.x_max
        y_min, y_max = spec.y_min, spec.y_max

    x_norm, y_norm, _, _ = contour_to_tensor(contour)
    rid = region_id(x_norm.detach(), y_norm.detach())

    # 评估 T(x,y) 与 T(x,-y)，保留 graph 用于两次 autograd.grad
    x_norm_g = x_norm.detach().requires_grad_(True)
    y_norm_g = y_norm.detach().requires_grad_(True)
    y_norm_neg_g = (-y_norm).detach().requires_grad_(True)

    T_pos = model(x_norm_g, y_norm_g, rid)
    # 第二次 forward（重算，保 graph）用于 T_neg 的 ∂T/∂y_neg
    T_neg_1 = model(x_norm_g, y_norm_neg_g, rid)

    # 求 T_pos 的 ∂T/∂x 与 ∂T/∂y（同一张图，分两次 backward）
    dT_dx_norm_pos = torch.autograd.grad(T_pos.sum(), x_norm_g, create_graph=False, retain_graph=True)[0]
    dT_dy_norm_pos = torch.autograd.grad(T_pos.sum(), y_norm_g, create_graph=False, retain_graph=False)[0]

    # 求 T_neg 的 ∂T/∂x 与 ∂T/∂y_neg（重新 forward 保 graph）
    T_neg_2 = model(x_norm_g, y_norm_neg_g, rid)
    dT_dx_norm_neg = torch.autograd.grad(T_neg_2.sum(), x_norm_g, create_graph=False, retain_graph=True)[0]
    dT_dy_norm_neg = torch.autograd.grad(T_neg_2.sum(), y_norm_neg_g, create_graph=False, retain_graph=False)[0]

    chain_x = 2.0 / (x_max - x_min)
    chain_y = 2.0 / (y_max - y_min)
    chain_T = (T_max - T_min) / 2.0

    # 物理空间梯度
    dT_dx_pos = dT_dx_norm_pos * chain_T * chain_x
    dT_dy_pos = dT_dy_norm_pos * chain_T * chain_y
    dT_dx_neg = dT_dx_norm_neg * chain_T * chain_x
    dT_dy_neg = dT_dy_norm_neg * chain_T * chain_y

    # 对称/反对称梯度（y 反号 → dT_dy 变号 → sym 抵消, asym 加倍）
    dT_dx_sym = 0.5 * (dT_dx_pos + dT_dx_neg)
    dT_dx_asym = 0.5 * (dT_dx_pos - dT_dx_neg)
    dT_dy_sym = 0.5 * (dT_dy_pos - dT_dy_neg)
    dT_dy_asym = 0.5 * (dT_dy_pos + dT_dy_neg)

    # 对称/反对称应力
    sig_xx_sym = dT_dx_sym * dT_dx_sym
    sig_yy_sym = dT_dy_sym * dT_dy_sym
    sig_xy_sym = dT_dx_sym * dT_dy_sym

    sig_xx_asym = dT_dx_asym * dT_dx_asym
    sig_yy_asym = dT_dy_asym * dT_dy_asym
    sig_xy_asym = dT_dx_asym * dT_dy_asym

    # J_I = ∮ (W_sym·n - σ_sym·n·∂T_asym/∂x) ds
    # J_II = ∮ (W_asym·n - σ_asym·n·∂T_sym/∂x) ds
    # 交叉项正交抵消（数学基础）
    _, _, n_x, n_y = contour_to_tensor(contour)

    def _integrate(sig_xx, sig_yy, sig_xy, dT_dx, dT_dy):
        W = 0.5 * (sig_xx + sig_yy)
        sigma_dot_n_x = sig_xx * n_x + sig_xy * n_y
        sigma_dot_n_y = sig_xy * n_x + sig_yy * n_y
        integrand = W * n_x - sigma_dot_n_x * dT_dx - sigma_dot_n_y * dT_dy
        return float(torch.trapz(integrand, torch.arange(len(integrand), dtype=torch.float64)).item())

    J_I = _integrate(sig_xx_sym, sig_yy_sym, sig_xy_sym, dT_dx_asym, dT_dy_asym)
    J_II = _integrate(sig_xx_asym, sig_yy_asym, sig_xy_asym, dT_dx_sym, dT_dy_sym)
    return J_I, J_II