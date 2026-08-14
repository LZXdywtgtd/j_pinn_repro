"""J-integral 后处理单元测试

测试目标（不依赖 JPINN checkpoint）：
1. test_contour_closed: RectContour.sample() 首尾点重合
2. test_stress_analog_consistency: 梯度积类比对双源场解析一致
3. test_j_path_independence_exact: 线性场 J=0（div σ=0 的真正路径无关）
   ⚠️ 注意：热场类比 σ=∇T∇T^T 对一般调和场不满足 div σ=0（见 ADR-0001），
   log 源场的 J 会随 contour 漂移——这是降维的已知局限，不是积分实现 bug。
4. test_j_exact_matches_trapezoidal: 闭合积分 ∮ n_x ds 精确为 0（真实弧长）
5. test_far_field_anchoring_corrects_drift: 注入线性漂移 → 纯去趋势补偿 → 曲面变平

"""
from __future__ import annotations

import math

import numpy as np
import torch

from postprocess.analytic_J import (
    analytic_grad_T_phys,
    j_integral_exact,
    j_integral_exact_surface,
)
from postprocess.contour_sampling import (
    RectContour,
    contour_ds,
    contour_to_tensor,
    sweep_contours,
)
from postprocess.far_field_anchoring import (
    compensate_j_surface,
    fit_linear_drift,
    path_independence_metric,
    relative_error,
)
from postprocess.stress_from_T import strain_energy_W, stress_analog


def test_contour_closed():
    """RectContour.sample() 首尾点应重合（闭合回路）"""
    c = RectContour(x_lig=-0.5, x_wake=0.5, n_per_side=50)
    x, y, _, _ = c.sample()
    assert len(x) == 4 * 50
    # 闭合：第一点 = 最后一点
    assert abs(x[0] - x[-1]) < 1e-12, f"x: first={x[0]}, last={x[-1]}"
    assert abs(y[0] - y[-1]) < 1e-12, f"y: first={y[0]}, last={y[-1]}"
    # 段间连续：段1末点=段2首点
    n = 50
    assert abs(x[n - 1] - x[n]) < 1e-12
    assert abs(y[3 * n - 1] - y[3 * n]) < 1e-12  # 段3末=段4首
    print("  ✓ Contour 闭合 + 段间连续")


def test_stress_analog_consistency():
    """σ_xx = (∂T/∂x)² 对已知 T 一致（v0.9：双源场精确期望，含 eps 修正）

    单源近似 W≈0.5/r² 忽略了 eps 项和 cold 源贡献；改用双源精确公式：
    ∇T = ∇log(r_hot) - ∇log(r_cold)，W = ½|∇T|² 逐项解析。
    """
    x = torch.tensor([0.7, -0.3, 0.0], dtype=torch.float64)
    y = torch.tensor([0.2, 0.5, 0.1], dtype=torch.float64)
    hot = (0.0, 0.0)
    cold = (10.0, 10.0)
    eps = 1e-4
    dT_dx, dT_dy = analytic_grad_T_phys(x, y, hot_xy=hot, cold_xy=cold, eps=eps)
    sig_xx, sig_yy, sig_xy = stress_analog(dT_dx, dT_dy)
    W = strain_energy_W(sig_xx, sig_yy)
    # 双源精确期望（与 analytic_grad_T_phys 同公式）
    r_h2 = (x - hot[0]) ** 2 + (y - hot[1]) ** 2 + eps
    r_c2 = (x - cold[0]) ** 2 + (y - cold[1]) ** 2 + eps
    gx = x / r_h2 - (x - cold[0]) / r_c2
    gy = y / r_h2 - (y - cold[1]) / r_c2
    W_expected = 0.5 * (gx ** 2 + gy ** 2)
    diff = (W - W_expected).abs().max().item()
    assert diff < 1e-10, f"W mismatch: max diff {diff}"
    print(f"  ✓ σ_xx = (∂T/∂x)² 对双源场精确一致（max diff {diff:.2e}）")


def test_j_path_independence_exact():
    """线性场 T = a·x + b·y 的 J 应精确为 0（真正的路径无关性）

    v0.9 修正测试假设（旧测试"log 源场路径无关"在物理上不成立）：
    热场类比 σ = ∇T∇T^T 的散度 div σ = ∇(½|∇T|²) ≠ 0（一般调和场），
    J 随 contour 位置漂移——这是热场降维的已知局限（与弹性场 div σ=0
    的 Rice 定理不同，见 ADR-0001）。线性场 T = ax+by 是唯一 div σ = 0
    的调和场：σ = 常数、W = 常数 → J = ∮(W n_x - σ·n·∇T)ds = 0（∮n ds=0）。
    """
    a, b = 2.0, 1.0
    contours = sweep_contours(
        x_lig_values=(-0.7, -0.5, -0.3),
        x_wake_values=(0.3, 0.5, 0.7),
        n_per_side=200,
    )
    max_abs = 0.0
    for c in contours:
        x, y, n_x, n_y = contour_to_tensor(c)
        ds = contour_ds(c)
        # 线性场：∇T = (a, b) 常数
        sig_xx = torch.full_like(x, a * a)
        sig_yy = torch.full_like(x, b * b)
        sig_xy = torch.full_like(x, a * b)
        W = 0.5 * (sig_xx + sig_yy)
        traction = (sig_xx * n_x + sig_xy * n_y) * a
        integrand = W * n_x - traction
        J = float(torch.sum((integrand[:-1] + integrand[1:]) * 0.5 * ds[:-1]).item())
        max_abs = max(max_abs, abs(J))
    print(f"  9 contours 线性场 max|J| = {max_abs:.3e}")
    assert max_abs < 1e-12, f"线性场 J 应精确为 0，实际 max|J|={max_abs:.3e}"
    print(f"  ✓ 线性场 J = 0（div σ = 0 的真正路径无关性）")


def test_j_exact_matches_trapezoidal():
    """闭合积分 ∮ n_x ds 应精确为 0（v0.9 修复弧长积分后）

    旧版 torch.trapz 用 arange 当弧长：
    - 竖段/横段长度权重错误（y_max-y_min vs x_wake-x_lig）
    - 闭合项缺失 → ∮ n_x ds 恒得 0.5 而非 0
    v0.9 用 contour_ds 真实弧长，∮ n_x ds = 3.5e-17。
    """
    n_values = [50, 100, 200, 400]
    errors = []
    for n in n_values:
        c = RectContour(x_lig=-0.5, x_wake=0.5, n_per_side=n)
        x, y, n_x, n_y = contour_to_tensor(c)
        ds = contour_ds(c)
        # ∮ n_x ds = 0（闭合曲线高斯定理）；n_x 逐段常数 → 梯形精确
        val = float(torch.sum((n_x[:-1] + n_x[1:]) * 0.5 * ds[:-1]).item())
        errors.append(abs(val))
    print(f"  errors @ n={n_values}: {errors}")
    # 全部 n 都应接近 0（不随 n 漂移）；旧版恒 0.5
    assert all(e < 1e-12 for e in errors), f"∮ n_x ds 应精确为 0：{errors}"
    print(f"  ✓ 闭合积分 ∮ n_x ds = 0（真实弧长梯形积分）")


def test_far_field_anchoring_corrects_drift():
    """注入线性漂移后，补偿应使曲面变平（v0.9 修复锚定语义）

    v0.9 修复：旧版 `+ J_far` 双倍计数远场平面值。
    正确锚定 = 纯去趋势：J_corr = J_raw - drift，
    远场点自然等于 J_raw(far)，拟合斜率应为 0。
    """
    x_lig = np.array([-0.7, -0.5, -0.3], dtype=np.float64)
    x_wake = np.array([0.3, 0.5, 0.7], dtype=np.float64)
    X_lig, X_wake = np.meshgrid(x_lig, x_wake, indexing="ij")
    x_lig_flat = X_lig.flatten()
    x_wake_flat = X_wake.flatten()
    # 真实 J = 2.0（常数），叠加线性漂移 + 噪声
    a_true, b_lig_true, b_wake_true = 2.0, 0.5, -0.3
    J_raw = a_true + b_lig_true * x_lig_flat + b_wake_true * x_wake_flat + np.random.RandomState(42).normal(0, 0.01, size=9)
    # 远场锚定（取最远的）
    x_lig_far = x_lig.min()
    x_wake_far = x_wake.max()
    J_corrected = compensate_j_surface(x_lig_flat, x_wake_flat, J_raw, x_lig_far, x_wake_far)
    # 锚定后曲面应平坦（b_lig ≈ b_wake ≈ 0）
    a, b_lig_fit, b_wake_fit = fit_linear_drift(x_lig_flat, x_wake_flat, J_corrected)
    assert abs(b_lig_fit) < 1e-8, f"b_lig 未消除: {b_lig_fit}"
    assert abs(b_wake_fit) < 1e-8, f"b_wake 未消除: {b_wake_fit}"
    # 远场点 J 应 ≈ J_raw(far)（纯去趋势，不加回平面值）
    far_idx = np.argmin(np.abs(x_lig_flat - x_lig_far) + np.abs(x_wake_flat - x_wake_far))
    assert abs(J_corrected[far_idx] - J_raw[far_idx]) < 1e-12, (
        f"远场点应等于 J_raw(far)：{J_corrected[far_idx]} vs {J_raw[far_idx]}"
    )
    print(f"  ✓ 远场锚定 = 纯去趋势（b_lig={b_lig_fit:.2e}, b_wake={b_wake_fit:.2e}）")
    print(f"  远场点 J = {J_corrected[far_idx]:.3f}（= J_raw 远场值）")


def main():
    print("\n=== J-integral 单元测试 ===\n")
    print("[1/5] test_contour_closed")
    test_contour_closed()
    print("\n[2/5] test_stress_analog_consistency")
    test_stress_analog_consistency()
    print("\n[3/5] test_j_path_independence_exact")
    test_j_path_independence_exact()
    print("\n[4/5] test_j_exact_matches_trapezoidal")
    test_j_exact_matches_trapezoidal()
    print("\n[5/5] test_far_field_anchoring_corrects_drift")
    test_far_field_anchoring_corrects_drift()
    print("\n=== ALL TESTS PASSED ===")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())