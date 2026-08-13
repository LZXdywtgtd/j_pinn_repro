"""
损失函数（5 类 + P2 outlier）+ 聚合器

对应论文 §2.3 Eq. 12-19，把 Navier-Cauchy 替换为 2D Laplace ∇²T=0：

1. pde_loss          — ∇²T=0 残差（4 区域分别计算）
2. interface_loss    — 缝合边界两侧温度连续（MSE）
3. bc_loss_dirichlet — 外边界 Dirichlet（Huber，论文 Eq. 20）
4. neumann_crack_loss — 裂纹段两侧法向跳跃（Huber）
5. smoothness_loss   — Sobolev Hessian 正则（论文 L_smooth，可关）
6. (P2) outlier.py   — Z-score 边界去噪（论文 §3.3 Eq.19-20，可选）

实现要点：
- 全程 PyTorch autograd 计算 ∂²T/∂x² 与 ∂²T/∂y²
- 坐标 must requires_grad=True 才能求导
- dtype=float64 保持精度
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

# P2 Z-score 边界去噪（v0.5）
try:
    from outlier import BoundaryOutlierTracker, OutlierConfig, bc_residuals
except ImportError:  # 兼容非模块上下文
    BoundaryOutlierTracker = None
    OutlierConfig = None
    bc_residuals = None


# ============================================================
# 1. PDE 残差（∇²T=0）
# ============================================================
def pde_residual(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    region_id: torch.Tensor,
) -> torch.Tensor:
    """
    计算 ∇²T = ∂²T/∂x² + ∂²T/∂y² 在每点的残差。
    通过 torch.autograd.grad 二次求导。

    Args:
        model: PINN 模型
        x, y: (N,) 已归一化坐标
        region_id: (N,) ∈ {0,1,2,3}

    Returns:
        residual: (N,)
    """
    # 确保 requires_grad 以便求导
    x = x.detach().requires_grad_(True)
    y = y.detach().requires_grad_(True)

    T = model(x, y, region_id)  # (N,)

    # 一阶导
    dT_dx = torch.autograd.grad(
        T.sum(), x, create_graph=True, retain_graph=True, allow_unused=False
    )[0]
    dT_dy = torch.autograd.grad(
        T.sum(), y, create_graph=True, retain_graph=True, allow_unused=False
    )[0]

    # 二阶导
    d2T_dx2 = torch.autograd.grad(
        dT_dx.sum(), x, create_graph=True, retain_graph=True, allow_unused=False
    )[0]
    d2T_dy2 = torch.autograd.grad(
        dT_dy.sum(), y, create_graph=True, retain_graph=True, allow_unused=False
    )[0]

    return d2T_dx2 + d2T_dy2


def pde_loss(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    region_id: torch.Tensor,
) -> torch.Tensor:
    """L_pde = mean(∇²T)²"""
    r = pde_residual(model, x, y, region_id)
    return (r ** 2).mean()


# ============================================================
# 2. 接口缝合损失（3 条 seam）
# ============================================================
def interface_loss_single(
    model: torch.nn.Module,
    x_l: torch.Tensor, y_l: torch.Tensor, rid_l: torch.Tensor,
    x_r: torch.Tensor, y_r: torch.Tensor, rid_r: torch.Tensor,
) -> torch.Tensor:
    """单条 seam 两点的 (T_l - T_r)²"""
    T_l = model(x_l, y_l, rid_l)
    T_r = model(x_r, y_r, rid_r)
    return F.mse_loss(T_l, T_r)


def interface_loss(
    model: torch.nn.Module,
    ifaces: Dict[str, Tuple],
) -> torch.Tensor:
    """
    3 条缝合边（A_B / A_C / B_D）之和。
    ifaces: dict[key] = (x_l, y_l, rid_l, x_r, y_r, rid_r)
    """
    losses = []
    for key in ("A_B", "A_C", "B_D"):
        losses.append(interface_loss_single(model, *ifaces[key]))
    return sum(losses) / len(losses)


def interface_loss_normal_single(
    model: torch.nn.Module,
    x_l: torch.Tensor, y_l: torch.Tensor, rid_l: torch.Tensor,
    x_r: torch.Tensor, y_r: torch.Tensor, rid_r: torch.Tensor,
    normal_axis: str,  # "x" for A_B (vertical seam); "y" for A_C / B_D (horizontal)
) -> torch.Tensor:
    """单条 seam 两侧法向导数连续：mean((∂T/∂n)_l - (∂T/∂n)_r)²

    对应论文 §3.3 Eq.16 L_traction 的 Laplace 降维（t = σ·n → ∂T/∂n）。
    """
    x_l = x_l.detach().requires_grad_(True)
    y_l = y_l.detach().requires_grad_(True)
    x_r = x_r.detach().requires_grad_(True)
    y_r = y_r.detach().requires_grad_(True)
    T_l = model(x_l, y_l, rid_l)
    T_r = model(x_r, y_r, rid_r)
    if normal_axis == "x":
        dT_dn_l = torch.autograd.grad(T_l.sum(), x_l, create_graph=True)[0]
        dT_dn_r = torch.autograd.grad(T_r.sum(), x_r, create_graph=True)[0]
    elif normal_axis == "y":
        dT_dn_l = torch.autograd.grad(T_l.sum(), y_l, create_graph=True)[0]
        dT_dn_r = torch.autograd.grad(T_r.sum(), y_r, create_graph=True)[0]
    else:
        raise ValueError(f"normal_axis must be 'x' or 'y', got {normal_axis!r}")
    return F.mse_loss(dT_dn_l, dT_dn_r)


def interface_loss_normal(
    model: torch.nn.Module,
    ifaces: Dict[str, Tuple],
) -> torch.Tensor:
    """3 条缝合 seam 法向连续（论文 L_traction 的 Laplace 降维）"""
    losses_list = [
        interface_loss_normal_single(model, *ifaces["A_B"], normal_axis="x"),
        interface_loss_normal_single(model, *ifaces["A_C"], normal_axis="y"),
        interface_loss_normal_single(model, *ifaces["B_D"], normal_axis="y"),
    ]
    return sum(losses_list) / len(losses_list)


# ============================================================
# 3. 外边界 Dirichlet 损失（Huber）
# ============================================================
def bc_loss_dirichlet(
    model: torch.nn.Module,
    x_b: torch.Tensor,
    y_b: torch.Tensor,
    region_id_b: torch.Tensor,
    T_target: torch.Tensor,
    huber_beta: float = 0.1,
) -> torch.Tensor:
    """
    外边界硬约束（论文 §2.3 L_bc，Huber 增强鲁棒性）。
    smooth_l1_loss 在 |r| <= beta 时退化为 0.5 r²/beta，
    在 |r| > beta 时退化为 |r| - 0.5*beta。
    """
    T_pred = model(x_b, y_b, region_id_b)
    return F.smooth_l1_loss(T_pred, T_target, beta=huber_beta)


# ============================================================
# 4. 裂纹段 Neumann 跳跃损失
# ============================================================
def neumann_crack_loss(
    model: torch.nn.Module,
    x_top: torch.Tensor, y_top: torch.Tensor, rid_top: torch.Tensor,
    x_bot: torch.Tensor, y_bot: torch.Tensor, rid_bot: torch.Tensor,
    dT_jump_value: float,
    eps: float = 1e-3,
    huber_beta: float = 0.05,
) -> torch.Tensor:
    """
    裂纹段上/下两侧法向导数跳跃。
    解析场 dT/dy 跳跃值 ≈ 100（因为 tanh(50y) 在 y=±eps 处导数跳变）。
    数值近似：(T_top - T_bot) / (2*eps) ≈ dT_jump_value
    """
    x_top_g = x_top.detach().requires_grad_(True)
    y_top_g = y_top.detach().requires_grad_(True)
    x_bot_g = x_bot.detach().requires_grad_(True)
    y_bot_g = y_bot.detach().requires_grad_(True)

    T_top = model(x_top_g, y_top_g, rid_top)
    T_bot = model(x_bot_g, y_bot_g, rid_bot)

    dT_dy_estimate = (T_top - T_bot) / (2.0 * eps)
    target = torch.full_like(dT_dy_estimate, dT_jump_value)
    return F.smooth_l1_loss(dT_dy_estimate, target, beta=huber_beta)


# ============================================================
# 5. Sobolev 平滑正则（Hessian Frobenius）
# ============================================================
def smoothness_loss(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    region_id: torch.Tensor,
) -> torch.Tensor:
    """
    L_smooth = mean( (∂²T/∂x²)² + (∂²T/∂y²)² + 2·(∂²T/∂x∂y)² )
    论文 §2.3 L_smooth，防谱偏置。Hessian 的 Frobenius 范数。
    注意：(∂²T/∂x² + ∂²T/∂y²)² 已被 PDE loss 覆盖；这里补交叉项。
    """
    x = x.detach().requires_grad_(True)
    y = y.detach().requires_grad_(True)
    T = model(x, y, region_id)

    dT_dx = torch.autograd.grad(T.sum(), x, create_graph=True)[0]
    dT_dy = torch.autograd.grad(T.sum(), y, create_graph=True)[0]
    d2T_dx2 = torch.autograd.grad(dT_dx.sum(), x, create_graph=True)[0]
    d2T_dy2 = torch.autograd.grad(dT_dy.sum(), y, create_graph=True)[0]
    # 交叉项：d²T/dxdy（注意：先 d/dx 后 d/dy）
    d2T_dxdy = torch.autograd.grad(dT_dx.sum(), y, create_graph=True)[0]

    hess_norm_sq = d2T_dx2 ** 2 + d2T_dy2 ** 2 + 2.0 * d2T_dxdy ** 2
    return hess_norm_sq.mean()


# ============================================================
# 6. 损失聚合器
# ============================================================
@dataclass
class LossWeights:
    """权重（论文热力场降维版）

    调参依据（5000 epoch 实测）：
    - lambda_pde 主导残差收敛；paper 用 1e-6 是因 Navier-Cauchy 含二阶混合导数量级大
    - 我们是 Laplace 方程（∂² + ∂²），pde_loss 量级 1.0+，所以 λ_pde=100 让它主导
    - lambda_interface 缝合太弱会跨区域漂移；强到 50 反而抑制 PDE 学习（实测）
    """
    lambda_pde: float = 100.0
    lambda_interface: float = 10.0
    lambda_interface_normal: float = 1.0  # L_traction 降维（论文 0.8e-6；Laplace 量级 1.0+）
    lambda_bc: float = 1.0
    lambda_neumann_crack: float = 0.05  # Neumann 量级 ~50，调小避免主导
    lambda_smooth: float = 0.0  # 默认关闭（Hessian 计算开销大且不稳）


class LossAggregator:
    """
    把 5 类损失按权重汇总，返回 total + 各分量 dict。

    batch schema（来自 ThermalDataset.get_collocation_batch）：
    {
        "interior": (x, y, rid),
        "boundary": {"x", "y", "edge_id", "T_target"},
        "interface": {"A_B": ..., "A_C": ..., "B_D": ...},
        "crack": {"top": (x,y,rid), "bot": (x,y,rid),
                  "T_jump_value": float, "dT_jump_value": float},
    }

    P2 扩展（v0.5）：
    - outlier_cfg 传入 OutlierConfig 时启用 Z-score 边界去噪
    - 需在 __call__ 传 current_epoch（用于 burn-in 判断）
    """

    def __init__(
        self,
        weights: LossWeights | None = None,
        outlier_cfg: Optional[OutlierConfig] = None,
        device: torch.device | str = "cpu",
    ) -> None:
        self.w = weights or LossWeights()
        self.outlier: Optional[BoundaryOutlierTracker] = None
        if outlier_cfg is not None and outlier_cfg.enabled:
            self.outlier = BoundaryOutlierTracker(
                outlier_cfg, device=device, dtype=torch.float64
            )

    def __call__(
        self,
        model: torch.nn.Module,
        batch: dict,
        current_epoch: int = 0,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        # 1. PDE
        xi, yi, rid_i = batch["interior"]
        L_p = pde_loss(model, xi, yi, rid_i)

        # 2. Interface（3 条 seam）
        L_i = interface_loss(model, batch["interface"])
        L_tn = interface_loss_normal(model, batch["interface"])

        # 3. Dirichlet BC（外边界）
        bd = batch["boundary"]
        # 给外边界构造 region_id（边界点可能跨区域，按 (x,y) 推断）
        rid_b = (
            ((bd["x"] < 0) & (bd["y"] >= 0)).long() * 0
            + ((bd["x"] >= 0) & (bd["y"] >= 0)).long() * 1
            + ((bd["x"] < 0) & (bd["y"] < 0)).long() * 2
            + ((bd["x"] >= 0) & (bd["y"] < 0)).long() * 3
        )
        L_b = bc_loss_dirichlet(
            model, bd["x"], bd["y"], rid_b.to(bd["x"].device), bd["T_target"]
        )

        # P2：Z-score 边界去噪（可选）
        bc_active_frac = 1.0
        bc_n_outliers = 0
        if self.outlier is not None:
            res = bc_residuals(model, bd["x"], bd["y"], rid_b.to(bd["x"].device), bd["T_target"])
            active_mask = self.outlier.update(
                res, bd["x"], bd["y"], bd["edge_id"], current_epoch
            )
            n_active = int(active_mask.sum().item())
            n_total = int(active_mask.numel())
            bc_n_outliers = n_total - n_active
            bc_active_frac = n_active / max(n_total, 1)
            if n_active == 0:
                # 防御：空活跃集 → L_bc = 0（避免 NaN）
                import warnings
                warnings.warn("P2: 外边界活跃集为空，L_bc 置 0")
                L_b = torch.tensor(0.0, device=L_p.device, dtype=L_p.dtype)
            else:
                L_b_raw = bc_loss_dirichlet(
                    model,
                    bd["x"][active_mask], bd["y"][active_mask],
                    rid_b.to(bd["x"].device)[active_mask],
                    bd["T_target"][active_mask],
                )
                # 论文 Table 5：|A| 归一化（N_total / N_active）
                L_b = L_b_raw * (n_total / n_active)

        # 4. Neumann 裂纹
        ck = batch["crack"]
        L_n = neumann_crack_loss(
            model, *ck["top"], *ck["bot"],
            dT_jump_value=ck["dT_jump_value"],
        )

        # 5. Sobolev（可选）
        L_s = (
            smoothness_loss(model, xi, yi, rid_i)
            if self.w.lambda_smooth > 0
            else torch.tensor(0.0, device=L_p.device, dtype=L_p.dtype)
        )

        total = (
            self.w.lambda_pde * L_p
            + self.w.lambda_interface * L_i
            + self.w.lambda_interface_normal * L_tn
            + self.w.lambda_bc * L_b
            + self.w.lambda_neumann_crack * L_n
            + self.w.lambda_smooth * L_s
        )

        comps = {
            "pde": float(L_p.item()),
            "iface": float(L_i.item()),
            "tnormal": float(L_tn.item()),
            "bc": float(L_b.item()),
            "neumann": float(L_n.item()),
            "smooth": float(L_s.item()),
            "bc_active_frac": bc_active_frac,
            "bc_n_outliers": bc_n_outliers,
            "total": float(total.item()),
        }
        return total, comps