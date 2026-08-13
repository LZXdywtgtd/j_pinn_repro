"""
远场锚定误差补偿（论文 §4.4 + 附录 A Eq. A.23-A.24）

PINN 不完美收敛会留下残留体力 `f_res = (∂σ_xx/∂x + ∂σ_xy/∂y, ...)`；
由散度定理，矩形 contour 的 J 误差项正比于 (|x_lig| + |x_wake|) · H_contour
（Eq. A.24，线性增长）。

补偿算法（论文 §4.4）：
1. J_raw(x_lig, x_wake) → 双线性平面拟合
   J_raw ≈ a + b_lig·x_lig + b_wake·x_wake
2. 远场锚定：在域边界 contour（x_lig_far, x_wake_far）处的 J_raw 作为"真值"
   J_far = J_raw(x_lig_far, x_wake_far)
3. 线性 detrend + 锚定到远场
   J_corrected(x, y) = J_raw(x, y) - b_lig·(x - x_lig_far) - b_wake·(y - x_wake_far)
                      + J_far
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def fit_linear_drift(
    x_lig: np.ndarray,
    x_wake: np.ndarray,
    J: np.ndarray,
) -> Tuple[float, float, float]:
    """最小二乘拟合 J = a + b_lig·x_lig + b_wake·x_wake

    Returns: (a, b_lig, b_wake)
    """
    # 设计矩阵：[1, x_lig, x_wake]
    A = np.stack([np.ones_like(x_lig), x_lig, x_wake], axis=1)  # (N, 3)
    # 最小二乘 (A^T A) θ = A^T J
    coeffs, _, _, _ = np.linalg.lstsq(A, J, rcond=None)
    a, b_lig, b_wake = coeffs
    return float(a), float(b_lig), float(b_wake)


def compensate_j_surface(
    x_lig: np.ndarray,
    x_wake: np.ndarray,
    J: np.ndarray,
    x_lig_far: float,
    x_wake_far: float,
) -> np.ndarray:
    """远场锚定补偿（论文 §4.4 公式）

    Args:
        x_lig, x_wake: contour 参数数组（1D）
        J: J 积分值数组（与 x_lig 同长）
        x_lig_far, x_wake_far: 远场 contour 参数（域边界附近）
    """
    a, b_lig, b_wake = fit_linear_drift(x_lig, x_wake, J)
    J_far = a + b_lig * x_lig_far + b_wake * x_wake_far  # 远场 raw J
    J_corrected = J - b_lig * (x_lig - x_lig_far) - b_wake * (x_wake - x_wake_far) + J_far
    return J_corrected


def path_independence_metric(J: np.ndarray) -> float:
    """路径无关度 = std/mean（理想为 0；论文 §4.4 报告 < 5%）"""
    if len(J) == 0:
        return float("inf")
    mean = float(np.mean(J))
    if abs(mean) < 1e-12:
        # mean ≈ 0 → 用 std 本身
        return float(np.std(J))
    return float(np.std(J) / abs(mean))


def relative_error(
    J_pred: np.ndarray,
    J_exact: np.ndarray,
) -> float:
    """相对误差 = |J_pred - J_exact| / max(|J_exact|)"""
    denom = max(np.abs(J_exact).max(), 1e-12)
    return float(np.abs(J_pred - J_exact).max() / denom)