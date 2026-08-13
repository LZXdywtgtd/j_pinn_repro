"""
合成 2D 热力场数据生成器

生成逻辑：
- 基础场：两个 log 调和源之差 → 严格调和（∇²T = 0 解析成立）
- 裂纹间断：在 |x| < 0.5 区间叠加 tanh(50y)，造成上/下半平面的温度跳变
- 输出：200×200 网格 → data/synthetic_thermal.npz

存储的 keys：
- x_grid, y_grid: (200, 200) 坐标
- T_grid: (200, 200) 真实温度（包含裂纹间断）
- T_smooth_grid: (200, 200) 仅 log 调和部分（不含间断，用于 ∇²T=0 自检）
- region_id_grid: (200, 200) ∈ {0,1,2,3} = {A, B, C, D}
- is_boundary_grid: (200, 200) bool，外边界标记
- is_crack_grid: (200, 200) bool，裂纹段标记

验证：
- 打印 max|∇²T_smooth| ≈ 0（解析调和性）
- 打印 T 范围、region_id 取值集合
"""
from __future__ import annotations

import os
import sys
import numpy as np

# 让脚本可作为模块导入或独立运行
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ============================================================
# 域与裂纹定义
# ============================================================
DOMAIN = {
    "x_min": -1.0,
    "x_max": 1.0,
    "y_min": -1.0,
    "y_max": 1.0,
}

CRACK = {
    "x_min": -0.5,
    "x_max": 0.5,
    "y_loc": 0.0,
    "tanh_steepness": 50.0,
    "tanh_jump": 2.0,
}

# 热源配置（log 调和极点）
SOURCES = {
    "hot": {"x": -0.6, "y": 0.5, "sign": +1.0},
    "cold": {"x": 0.6, "y": -0.5, "sign": -1.0},
    "offset_eps": 1e-4,  # 防 log(0) 数值溢出
}


# ============================================================
# 解析函数
# ============================================================
def T_smooth_np(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    严格调和的 log 源之差（无裂纹间断）
    T(x, y) = sign_hot * log(r_hot + eps) + sign_cold * log(r_cold + eps)
    """
    hot = SOURCES["hot"]
    cold = SOURCES["cold"]
    eps = SOURCES["offset_eps"]

    r_hot = np.sqrt((x - hot["x"]) ** 2 + (y - hot["y"]) ** 2 + eps)
    r_cold = np.sqrt((x - cold["x"]) ** 2 + (y - cold["y"]) ** 2 + eps)

    return hot["sign"] * np.log(r_hot) + cold["sign"] * np.log(r_cold)


def T_crack_np(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    裂纹间断项（仅在 |x| < 0.5 区间激活）
    crack(x, y) = tanh_jump * tanh(tanh_steepness * y) * 𝟙[|x| < crack_x_max]
    """
    in_crack = (np.abs(x) < CRACK["x_max"]).astype(np.float64)
    return CRACK["tanh_jump"] * np.tanh(CRACK["tanh_steepness"] * y) * in_crack


def T_exact_np(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """完整温度场（包含裂纹间断）"""
    return T_smooth_np(x, y) + T_crack_np(x, y)


def region_id_np(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    4 区域分解（与论文保持一致）：
      A (0): x<0, y>=0
      B (1): x>=0, y>=0
      C (2): x<0, y<0
      D (3): x>=0, y<0
    """
    rid = np.zeros_like(x, dtype=np.int64)
    rid[(x >= 0) & (y >= 0)] = 1
    rid[(x < 0) & (y < 0)] = 2
    rid[(x >= 0) & (y < 0)] = 3
    return rid


def is_boundary_np(x: np.ndarray, y: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """外边界标记（4 条边）"""
    return (
        (np.abs(x - DOMAIN["x_min"]) < tol)
        | (np.abs(x - DOMAIN["x_max"]) < tol)
        | (np.abs(y - DOMAIN["y_min"]) < tol)
        | (np.abs(y - DOMAIN["y_max"]) < tol)
    )


def is_crack_np(x: np.ndarray, y: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """裂纹段标记（y≈0 且 |x| ≤ 0.5）"""
    return (
        (np.abs(y - CRACK["y_loc"]) < tol)
        & (x >= CRACK["x_min"])
        & (x <= CRACK["x_max"])
    )


# ============================================================
# ∇²T 数值自检（验证解析调和性）
# ============================================================
def laplacian_2d_via_torch(x: np.ndarray, y: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """
    通过 PyTorch autograd 计算 ∇²T（更稳健，避开中心差分在源附近的数值爆炸）。
    仅用于自检，不进入训练循环。
    """
    import torch
    xt = torch.as_tensor(x, dtype=torch.float64).reshape(-1).requires_grad_(True)
    yt = torch.as_tensor(y, dtype=torch.float64).reshape(-1).requires_grad_(True)

    r_hot = torch.sqrt((xt - SOURCES["hot"]["x"]) ** 2 + (yt - SOURCES["hot"]["y"]) ** 2 + eps)
    r_cold = torch.sqrt((xt - SOURCES["cold"]["x"]) ** 2 + (yt - SOURCES["cold"]["y"]) ** 2 + eps)
    T = torch.log(r_hot) - torch.log(r_cold)

    dT_dx = torch.autograd.grad(T.sum(), xt, create_graph=True)[0]
    dT_dy = torch.autograd.grad(T.sum(), yt, create_graph=True)[0]
    d2T_dx2 = torch.autograd.grad(dT_dx.sum(), xt, create_graph=True)[0]
    d2T_dy2 = torch.autograd.grad(dT_dy.sum(), yt, create_graph=True)[0]
    return (d2T_dx2 + d2T_dy2).detach().numpy().reshape(x.shape)


# ============================================================
# 主入口
# ============================================================
def main(N: int = 200, out_path: str | None = None) -> None:
    if out_path is None:
        out_path = os.path.join(CURRENT_DIR, "synthetic_thermal.npz")

    x = np.linspace(DOMAIN["x_min"], DOMAIN["x_max"], N, dtype=np.float64)
    y = np.linspace(DOMAIN["y_min"], DOMAIN["y_max"], N, dtype=np.float64)
    X, Y = np.meshgrid(x, y, indexing="xy")  # 注意 indexing='xy' 让 x 沿 axis=1

    # 计算温度场
    T_smooth = T_smooth_np(X, Y)
    T_full = T_exact_np(X, Y)
    region_id = region_id_np(X, Y)
    is_bdy = is_boundary_np(X, Y)
    is_crk = is_crack_np(X, Y)

    # 调和性自检（排除裂纹 + 源附近 ε 内，因解析 ∇²T 严格为 0 但自检需排除 log 源近端）
    lap = laplacian_2d_via_torch(X, Y)
    # 排除：裂纹段、外边界、源周围 0.05 半径内
    source_radius = 0.05
    dist_hot = np.sqrt((X - SOURCES["hot"]["x"]) ** 2 + (Y - SOURCES["hot"]["y"]) ** 2)
    dist_cold = np.sqrt((X - SOURCES["cold"]["x"]) ** 2 + (Y - SOURCES["cold"]["y"]) ** 2)
    interior_mask = ~is_bdy & ~is_crk & (dist_hot > source_radius) & (dist_cold > source_radius)
    lap_max_interior = float(np.max(np.abs(lap[interior_mask]))) if interior_mask.any() else 0.0
    lap_mean_interior = float(np.mean(np.abs(lap[interior_mask]))) if interior_mask.any() else 0.0

    # 保存
    np.savez_compressed(
        out_path,
        x_grid=X,
        y_grid=Y,
        T_grid=T_full,
        T_smooth_grid=T_smooth,
        region_id_grid=region_id,
        is_boundary_grid=is_bdy,
        is_crack_grid=is_crk,
        # 元数据
        meta_x_min=DOMAIN["x_min"],
        meta_x_max=DOMAIN["x_max"],
        meta_y_min=DOMAIN["y_min"],
        meta_y_max=DOMAIN["y_max"],
        meta_crack_x_min=CRACK["x_min"],
        meta_crack_x_max=CRACK["x_max"],
        meta_N=N,
    )

    # 报告（注意：Windows GBK 控制台无法打印 ∇/²/⚠️ 等 Unicode，用 ASCII 替代）
    print(f"== 合成 2D 热力场生成完成 ==")
    print(f"  网格: {N}x{N}  域: [{DOMAIN['x_min']},{DOMAIN['x_max']}] x [{DOMAIN['y_min']},{DOMAIN['y_max']}]")
    print(f"  保存到: {out_path}")
    print(f"  T_full 范围:  [{T_full.min():.4f}, {T_full.max():.4f}]")
    print(f"  T_smooth 范围: [{T_smooth.min():.4f}, {T_smooth.max():.4f}]")
    print(f"  region_id 取值: {sorted(np.unique(region_id).tolist())}")
    print(f"  调和性自检 (Laplacian(T_smooth)): max={lap_max_interior:.3e}  mean={lap_mean_interior:.3e}")
    if lap_max_interior > 1e-6:
        print(f"  [WARN] 调和性偏离较大 (max={lap_max_interior:.3e})，检查解析函数")
    else:
        print(f"  [OK] 调和性自检通过 (Laplacian ~ 0 在非裂纹内部，非源近端)")


if __name__ == "__main__":
    main()