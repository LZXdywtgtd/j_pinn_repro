"""
数据集加载器（统一入口）

支持两种数据源：
- "synthetic": 调 data/generate_synthetic_thermal_data.py 生成的 .npz
- "comsol_png": 调 data/comsol_png_loader.py（暂 stub）

统一接口：
- ThermalDataset.__init__(npz_path, comsol_dir, source, ...)
- ThermalDataset.get_collocation_batch(N_int, N_bc, N_iface, N_crack, device) → dict

返回 batch schema（train.py 消费）：
{
    "interior": (x, y, region_id),                 # 域内配点
    "boundary": {"x", "y", "edge_id", "T_target"}, # 外边界 Dirichlet
    "interface": {
        "A_B": ((x_l, y_l, rid_l), (x_r, y_r, rid_r)),
        "A_C": (...),
        "B_D": (...),
    },
    "crack": {
        "top": (x_t, y_t, rid_t),
        "bot": (x_b, y_b, rid_b),
        "T_jump_value": float,  # 解析裂纹 tanh 跳变值（绝对温标）
        "dT_jump_value": float, # 法向导数跳变估值（用于 Neumann）
    },
}
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from utils import (
    DomainSpec,
    DEFAULT_DOMAIN,
    T_exact_torch,
    normalize_to_unit,
    sample_boundary,
    sample_crack,
    sample_interface,
    sample_interior,
)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


@dataclass
class ThermalDataset:
    npz_path: str = os.path.join(CURRENT_DIR, "synthetic_thermal.npz")
    comsol_dir: str = ""  # source="comsol_png" 时必填
    comsol_colorbar_range: tuple[float, float] = (293.15, 1431.15)  # COMSOL K
    comsol_xy_extent: tuple[float, float, float, float] = (0.0, 0.01, 0.0, 0.01)
    comsol_multiplier: float = 1.0  # COMSOL 色标乘数（如 ×10² → 100）
    source: str = "synthetic"  # "synthetic" | "comsol_png"
    device: torch.device | str = "cpu"
    dtype: torch.dtype = torch.float64

    # 运行时填充（__post_init__）
    T_min: float = 0.0
    T_max: float = 1.0
    spec: DomainSpec = DEFAULT_DOMAIN

    def __post_init__(self) -> None:
        if self.source == "synthetic":
            self._load_synthetic()
        elif self.source == "comsol_png":
            self._load_comsol_png()
        else:
            raise ValueError(f"Unknown source: {self.source}")

    def _load_comsol_png(self) -> None:
        from comsol_png_loader import load_comsol_scan_dir  # 局部导入避免循环
        if not self.comsol_dir:
            raise ValueError(
                "source='comsol_png' 时必须传 comsol_dir"
                "（如 D:/team_project/simulation/参考输入/参数化扫描1）"
            )
        data = load_comsol_scan_dir(
            scan_dir=self.comsol_dir,
            colorbar_range=self.comsol_colorbar_range,
            xy_extent=self.comsol_xy_extent,
        )
        self.T_min = float(data["T_grid"].min())
        self.T_max = float(data["T_grid"].max())
        # 注意：COMSOL 物理域是 [0, 0.01]×[0, 0.01]，不是 [-1, 1]²
        # 但 ThermalDataset.spec 期望 [-1, 1]² 归一化（与 utils.DEFAULT_DOMAIN 一致）
        # 解决：把物理域通过 normalize 映射到 [-1, 1]²
        x_min_phys, x_max_phys = self.comsol_xy_extent[0], self.comsol_xy_extent[1]
        y_min_phys, y_max_phys = self.comsol_xy_extent[2], self.comsol_xy_extent[3]
        self.spec = DomainSpec(
            x_min=x_min_phys, x_max=x_max_phys,
            y_min=y_min_phys, y_max=y_max_phys,
            crack_x_min=-0.5, crack_x_max=0.5,  # 与合成数据一致（论文保形）
        )
        # 但 utils.sample_* 默认用 DEFAULT_DOMAIN 的 [-1, 1]
        # → 需要在 sample 时用此 spec（v0.3 已实现，get_collocation_batch 传 spec=self.spec）
        # 注：v0.3 训练时若要切换到 comsol_png，
        # 还需要把内部坐标归一化为 [-1, 1]²（v0.4 后续）

    def _load_synthetic(self) -> None:
        if not os.path.exists(self.npz_path):
            raise FileNotFoundError(
                f"合成数据 .npz 不存在: {self.npz_path}\n"
                "请先运行：python data/generate_synthetic_thermal_data.py"
            )
        data = np.load(self.npz_path)
        self.T_min = float(data["T_grid"].min())
        self.T_max = float(data["T_grid"].max())
        self.spec = DomainSpec(
            x_min=float(data["meta_x_min"]),
            x_max=float(data["meta_x_max"]),
            y_min=float(data["meta_y_min"]),
            y_max=float(data["meta_y_max"]),
            crack_x_min=float(data["meta_crack_x_min"]),
            crack_x_max=float(data["meta_crack_x_max"]),
        )

    # ============================================================
    # 外边界 Dirichlet 真值（每 epoch 重新计算；与采样点对应）
    # ============================================================
    def boundary_target(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """外边界点对应的真实温度（归一化到 [-1, 1]）"""
        T = T_exact_torch(
            x, y,
            include_crack=True,
            hot_xy=(-0.6, 0.5),
            cold_xy=(0.6, -0.5),
            crack_x_max=self.spec.crack_x_max,
        )
        return normalize_to_unit(T, self.T_min, self.T_max)

    def _to_f64(self, t: torch.Tensor) -> torch.Tensor:
        """防御性 dtype cast：np.load 可能返回非 float64，统一到 self.dtype"""
        return t.to(self.dtype) if t.dtype != self.dtype else t

    # ============================================================
    # 配点 batch（每 epoch 重新采样）
    # ============================================================
    def get_collocation_batch(
        self,
        n_int_per_region: int = 2500,
        n_bc_per_edge: int = 100,
        n_iface_per_seam: int = 50,
        n_crack_per_side: int = 50,
        seed: Optional[int] = None,
    ) -> dict:
        # 内部配点
        xi, yi, rid_i = sample_interior(
            n_int_per_region, spec=self.spec, device=self.device, dtype=self.dtype, seed=seed
        )

        # 外边界
        xb, yb, edge_id = sample_boundary(
            n_bc_per_edge, spec=self.spec, device=self.device, dtype=self.dtype, seed=None if seed is None else seed + 1
        )
        T_bc_target = self.boundary_target(xb, yb)

        # 缝合接口
        ifaces = sample_interface(
            n_iface_per_seam, spec=self.spec, device=self.device, dtype=self.dtype, seed=None if seed is None else seed + 2
        )

        # 裂纹段
        x_ct, y_ct, rid_ct, x_cb, y_cb, rid_cb = sample_crack(
            n_crack_per_side, spec=self.spec, device=self.device, dtype=self.dtype, seed=None if seed is None else seed + 3
        )

        # dtype 数据入口统一（防御性 cast）
        # np.load() 读 .npz 时 numpy.dtype 可能为 float32；
        # 显式 cast 保证与模型 dtype（默认 float64）一致，避免 autograd 精度退化
        xi, yi, xb, yb, x_ct, y_ct, x_cb, y_cb = (
            self._to_f64(xi), self._to_f64(yi), self._to_f64(xb), self._to_f64(yb),
            self._to_f64(x_ct), self._to_f64(y_ct), self._to_f64(x_cb), self._to_f64(y_cb),
        )
        T_bc_target = self._to_f64(T_bc_target)
        ifaces = {k: tuple(self._to_f64(t) for t in v) for k, v in ifaces.items()}

        # 解析裂纹跳跃量（在归一化空间）
        # 裂纹项 = jump * tanh(steepness * y)，跃迁量约为 2 * jump
        # 在归一化 [T_min, T_max] → [-1, 1] 空间，跳跃量为：
        T_jump_physical = 2.0 * 2.0  # jump=2.0 → 上下极限差 4.0
        T_jump_normalized = (2.0 * T_jump_physical) / (self.T_max - self.T_min)

        # 法向导数跳跃估计：d/dy tanh(50y) 在 y=±eps 处近似 50（极大值，在 y=0 间断）；
        # 用更合理的"两侧导数差"= 100/2 = 50（中心差分近似）
        dT_jump_normalized = 50.0

        return {
            "interior": (xi, yi, rid_i),
            "boundary": {
                "x": xb, "y": yb, "edge_id": edge_id, "T_target": T_bc_target,
            },
            "interface": ifaces,
            "crack": {
                "top": (x_ct, y_ct, rid_ct),
                "bot": (x_cb, y_cb, rid_cb),
                "T_jump_value": float(T_jump_normalized),
                "dT_jump_value": float(dT_jump_normalized),
            },
            "meta": {
                "T_min": self.T_min,
                "T_max": self.T_max,
            },
        }