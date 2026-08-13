"""
COMSOL 2D 温度场 PNG 加载器（STUB）

设计目的：
- 与 generate_synthetic_thermal_data.py 的 .npz 输出 schema 对齐
  让 ThermalDataset 可无感切换数据源
- COMSOL 输出的"温度" PNG 是带色标（colorbar）的 2D 图像
  需按色标范围把像素灰度反推为温度 K
- 当前仅提供接口签名 + NotImplementedError 提示
  解析 PNG 像素 + 色标反推留给后续 phase 实现

数据源路径（D:\team_project\simulation\参考输入\参数化扫描*\温度\*.png）：
- 155 张/扫描；每张对应一个时间步
- 像素尺寸通常 640×480（v4 RAW_IMAGE_SIZE）
- 域尺寸 ~5–10mm；色标范围 ~1150–1450K
"""
from __future__ import annotations

import os
from typing import Tuple

import numpy as np


def load_comsol_png(
    png_path: str,
    colorbar_range: Tuple[float, float] = (1150.0, 1450.0),
    crop_ratio: float = 0.70,
    xy_extent: Tuple[float, float, float, float] = (-0.005, 0.005, -0.005, 0.005),
) -> np.ndarray:
    """
    读取一张 COMSOL 温度 PNG，按色标反推温度场。

    Args:
        png_path: PNG 文件绝对路径
        colorbar_range: (T_min, T_max) 单位 K；像素灰度线性映射到此范围
        crop_ratio: 中心裁剪比例（与 v4 dataset_multimodal.py 一致：聚焦物理区域）
        xy_extent: (x_min, x_max, y_min, y_max) 单位 m；与 COMSOL 几何对齐

    Returns:
        T_field: (H', W') 物理温度 K 的二维数组
        meta: dict 含 T_min/T_max 与 xy 范围

    TODO:
        - 解析 PNG：PIL 读 → 灰度 → 中心裁剪
        - 色标反推：min-max normalize 像素 → 线性插值到 colorbar_range
        - xy 映射：与 COMSOL 几何对齐（取决于具体扫描的域尺寸）
    """
    raise NotImplementedError(
        "COMSOL PNG 加载器尚未实现。"
        "需要：(1) PIL 读 PNG + 中心裁剪；"
        "(2) 按 colorbar_range 把像素 → 温度 K；"
        "(3) xy_extent 映射到物理坐标。"
        f"目标文件: {png_path}"
    )


def load_comsol_scan_dir(
    scan_dir: str,
    colorbar_range: Tuple[float, float] = (1150.0, 1450.0),
    crack_x_range: Tuple[float, float] = (-0.5, 0.5),
) -> dict:
    """
    加载一个扫描目录下所有 PNG，归一化到 [-1, 1]^2，与合成数据 schema 对齐。

    Returns:
        dict with keys:
            x_grid, y_grid, T_grid, T_smooth_grid, region_id_grid,
            is_boundary_grid, is_crack_grid
        形状 (H, W)；T_smooth_grid 留 NaN（真实数据无法解析调和分量）
    """
    raise NotImplementedError(
        "COMSOL 扫描目录加载器尚未实现。"
        f"目标目录: {scan_dir}"
    )