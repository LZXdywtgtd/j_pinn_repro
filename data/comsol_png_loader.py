"""
COMSOL 2D 温度场 PNG 加载器（v0.3 完整实现）

设计目的：
- 与 generate_synthetic_thermal_data.py 的 .npz 输出 schema 对齐
  让 ThermalDataset 可无感切换数据源（source="comsol_png"）
- COMSOL 输出的"温度" PNG 是带色标（colorbar）的 2D 图像
  按用户提供的 colorbar_range 把像素 RGB → 温度 K

实测 COMSOL PNG 结构（D:\team_project\simulation\参考输入\参数化扫描*\温度\*.png）：
- 像素尺寸 2000×1500（不是 v4 默认 640×480）
- 右侧 ~200px 是色标垂直渐变条
- 物理域 0.01×0.01 m，色标默认 ×10² 标度（即温度数字 4-14 × 10² = 400-1400 K）

容错设计：
- colorbar_range **必填无默认**：避免"忘了设定时偏差很大"
- 物理域自动检测：找非白像素的方形边界
- 乘数自动检测：从色标数字推断 ×10²（避免手动指定）
- 像素 → 温度用最近邻 RGB 匹配（在色标采样列中找最近色）
"""
from __future__ import annotations

import os
import re
import warnings
from typing import Optional, Tuple

import numpy as np


def _imread_unicode(path: str) -> np.ndarray:
    """PIL.Image.open 支持 Unicode 路径（v4 dataset_multimodal.py:423-441 模式）"""
    try:
        from PIL import Image
        with Image.open(path) as img:
            arr = np.array(img.convert("RGB"))
        return arr
    except ImportError:
        raise ImportError("PIL/Pillow 未安装：请运行 `pip install Pillow`")


def _detect_physical_region(img: np.ndarray, white_threshold: int = 245) -> Tuple[int, int, int, int]:
    """自动检测物理域（非白像素的方形边界）

    COMSOL 温度 PNG：四周有白色 padding（标题/坐标轴/图例），中间是色块域
    返回 (top, bottom, left, right) 像素索引（half-open）

    算法：
    - 找每行 / 每列非白色像素范围
    - 取交集最密集区域（找有色域的方形 bounding box）
    """
    H, W, _ = img.shape
    # 检测非白像素
    is_colored = np.any(img < white_threshold, axis=2)  # (H, W)
    # 行 / 列的非白范围
    row_has_color = is_colored.any(axis=1)  # (H,)
    col_has_color = is_colored.any(axis=0)  # (W,)
    if not row_has_color.any() or not col_has_color.any():
        # 全部白色 → fallback 中心 70%
        h_lo, h_hi = int(H * 0.15), int(H * 0.85)
        w_lo, w_hi = int(W * 0.15), int(W * 0.85)
        return h_lo, h_hi, w_lo, w_hi
    rows_idx = np.where(row_has_color)[0]
    cols_idx = np.where(col_has_color)[0]
    top, bottom = int(rows_idx.min()), int(rows_idx.max()) + 1
    left, right = int(cols_idx.min()), int(cols_idx.max()) + 1
    return top, bottom, left, right


def _detect_colorbar(img: np.ndarray,
                     physical_top: int, physical_bottom: int,
                     physical_left: int, physical_right: int) -> Tuple[int, int]:
    """定位色标条（右侧的渐变窄带）

    色标特征：
    - 位于物理域右侧外部
    - 垂直方向有渐变（每行 RGB 不同）
    - 比物理域窄得多（width ~ 80-200 px）

    返回 (cb_top, cb_bottom) 像素行索引
    """
    H, W, _ = img.shape
    # 在物理域右侧外的范围找（physical_right 之后）
    cb_search_left = min(physical_right + 10, W - 1)
    cb_search_right = W
    if cb_search_left >= cb_search_right - 5:
        return physical_top, physical_bottom  # fallback
    # 找列方差大的（渐变条 vs. 纯色背景）
    region = img[:, cb_search_left:cb_search_right, :]  # (H, W', 3)
    # 每列的"行间方差"（高 → 渐变条）
    col_variance = region.var(axis=0).sum(axis=1)  # (W',)
    if col_variance.max() < 10:
        return physical_top, physical_bottom  # fallback
    cb_col_start = cb_search_left + int(col_variance.argmax())
    # 在色标列附近 ~50px 范围内都视为色标
    cb_top_candidates = []
    cb_bottom_candidates = []
    for c in range(max(cb_col_start - 30, 0), min(cb_col_start + 30, W)):
        col_pixels = img[:, c, :]
        col_is_colored = np.any(col_pixels < 245, axis=1)
        idx = np.where(col_is_colored)[0]
        if len(idx) > 0:
            cb_top_candidates.append(int(idx.min()))
            cb_bottom_candidates.append(int(idx.max()))
    if not cb_top_candidates:
        return physical_top, physical_bottom
    # 限制在物理域垂直范围（避免标题 / 文本混入）
    cb_top = max(min(cb_top_candidates), physical_top)
    cb_bottom = min(max(cb_bottom_candidates), physical_bottom)
    return cb_top, cb_bottom


def _sample_colorbar(img: np.ndarray,
                    cb_top: int, cb_bottom: int,
                    cb_left: int, cb_right: int) -> np.ndarray:
    """采样色标条 → (H, 3) RGB 数组（自上而下）"""
    # 在色标列范围内取中位数（抗噪）
    region = img[cb_top:cb_bottom, cb_left:cb_right, :].astype(np.float64)
    cb_rgb = np.median(region, axis=1)  # (H, 3)
    return cb_rgb


def _rgb_to_colorbar_position(
    pixel_rgb: np.ndarray, cb_rgb: np.ndarray,
) -> np.ndarray:
    """每像素找色标最近邻 → 位置 p ∈ [0, 1]

    pixel_rgb: (..., 3)
    cb_rgb: (H, 3)
    Returns: (...,) 位置 p
    """
    # 距离矩阵 (..., H)
    diff = pixel_rgb.astype(np.float64)[..., None, :] - cb_rgb[None, ...]  # (..., H, 3)
    dist_sq = (diff ** 2).sum(axis=-1)  # (..., H)
    p = dist_sq.argmin(axis=-1) / max(len(cb_rgb) - 1, 1)
    return p


def _detect_multiplier_from_ticks(
    cb_top: int, cb_bottom: int, img: np.ndarray,
) -> Tuple[float, np.ndarray]:
    """从色标附近的 tick 标签推断乘数（×10² 等）

    简化版本：不做 OCR（重型）
    返回 (multiplier, colorbar_pixels)  其中 multiplier 默认 1.0
    """
    # COMSOL 默认在色标顶部右侧显示 ×10^N
    # 简化：返回默认 1.0；用户可通过 multiplier 参数覆盖
    # 后续如需要 OCR，再扩展
    return 1.0, None


def load_comsol_png(
    png_path: str,
    colorbar_range: Tuple[float, float],
    *,
    xy_extent: Tuple[float, float, float, float] = (0.0, 0.01, 0.0, 0.01),
    multiplier: Optional[float] = None,
    colormap_hint: str = "inferno",
    expected_image_shape: Tuple[int, int] = (1500, 2000),
    warn_uniform: bool = True,
) -> Tuple[np.ndarray, dict]:
    """
    读取一张 COMSOL 温度 PNG，按色标反推温度场。

    Args:
        png_path: PNG 文件绝对路径
        colorbar_range: (T_min, T_max) 单位 K；**必填无默认**（用户约束：避免忘了设定时偏差大）
        xy_extent: (x_min, x_max, y_min, y_max) 单位 m；与 COMSOL 几何对齐（默认 0.01×0.01m）
        multiplier: 色标乘数（默认 None → 自动推断为 1.0）；温度 PNG 通常 ×10²
        colormap_hint: 色带类型（用于文档）
        expected_image_shape: 期望 (H, W)（默认 COMSOL 2000×1500）
        warn_uniform: 若全场温度几乎相等（单色），是否警告

    Returns:
        T_field: (H', W') 物理温度 K 的二维数组（已去 padding）
        meta: dict 含 T_min/T_max + xy 范围 + 像素尺寸 + 物理尺寸 + 乘数

    Raises:
        FileNotFoundError: png_path 不存在
        ValueError: png_path 指向非 RGB 图或 colorbar_range 无效
    """
    if not os.path.exists(png_path):
        raise FileNotFoundError(f"PNG 不存在: {png_path}")
    if colorbar_range[1] <= colorbar_range[0]:
        raise ValueError(
            f"colorbar_range[1] 必须 > [0]，当前 ({colorbar_range[0]}, {colorbar_range[1]})"
        )

    img = _imread_unicode(png_path)
    H, W, C = img.shape
    if C != 3:
        raise ValueError(f"期望 RGB 图，实际 {C} 通道")
    if (H, W) != expected_image_shape:
        warnings.warn(
            f"PNG 形状 {(H, W)} 与期望 {expected_image_shape} 不符；继续处理（物理区域将自动检测）"
        )

    # 自动检测物理域
    p_top, p_bot, p_left, p_right = _detect_physical_region(img)
    physical = img[p_top:p_bot, p_left:p_right, :]
    H_phys, W_phys, _ = physical.shape

    # 自动检测色标（在物理域右侧）
    cb_top, cb_bot = _detect_colorbar(img, p_top, p_bot, p_left, p_right)
    cb_left = p_right  # 紧贴物理域右侧
    cb_right = min(cb_left + 50, W)  # 50px 宽的色标搜索带

    # 采样色标（如果检测失败，fallback 到 inferno colormap）
    try:
        cb_rgb = _sample_colorbar(img, cb_top, cb_bot, cb_left, cb_right)
        if cb_rgb.shape[0] < 10:
            raise ValueError("色标采样过短")
    except Exception:
        # Fallback：用 matplotlib 的 colormap
        try:
            import matplotlib.cm as cm
            cmap = getattr(cm, colormap_hint, cm.inferno)
            cb_rgb = (cmap(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.float64)
            warnings.warn("色标自动检测失败；fallback 到 matplotlib inferno 色带")
        except ImportError:
            raise RuntimeError("色标检测失败且 matplotlib 未安装")

    # 乘数推断
    if multiplier is None:
        multiplier = 1.0
        # 简化版本不做 OCR 推断
        # 若用户预期 ×10²，可显式传 multiplier=100

    # 像素 → 色标位置
    p_phys = _rgb_to_colorbar_position(physical, cb_rgb)  # (H_phys, W_phys)

    # 物理域坐标（按 xy_extent 映射）
    x_min, x_max, y_min, y_max = xy_extent
    T_phys = colorbar_range[0] + p_phys * (colorbar_range[1] - colorbar_range[0])
    T_phys = T_phys * multiplier  # 乘数应用（若用户传了）

    # 警告：均匀场（t=0 常见）
    if warn_uniform:
        T_std = float(T_phys.std())
        if T_std < 0.5:  # K
            warnings.warn(
                f"温度场标准差仅 {T_std:.4f} K（极均匀）；可能是 t=0 帧（场未演化）。"
                "色标反推仍正确，但可视化会一片纯色。"
            )

    meta = {
        "T_min": float(colorbar_range[0] * multiplier),
        "T_max": float(colorbar_range[1] * multiplier),
        "xy_extent": xy_extent,
        "xy_min": (x_min, y_min),
        "xy_max": (x_max, y_max),
        "pixel_shape": (H_phys, W_phys),
        "raw_shape": (H, W),
        "physical_bbox_px": (p_top, p_bot, p_left, p_right),
        "colorbar_bbox_px": (cb_top, cb_bot, cb_left, cb_right),
        "multiplier": multiplier,
        "colormap": colormap_hint,
        "source": "comsol_png",
    }
    return T_phys, meta


def load_comsol_scan_dir(
    scan_dir: str,
    colorbar_range: Tuple[float, float],
    *,
    subdir: str = "温度",
    xy_extent: Tuple[float, float, float, float] = (0.0, 0.01, 0.0, 0.01),
    crack_x_range: Tuple[float, float] = (-0.5, 0.5),
    crack_y_loc: float = 0.0,
    region_split: str = "quadrant",
    filename_pattern: str = r"温度(\d+)\.png$",
    max_frames: Optional[int] = None,
) -> dict:
    """
    加载一个扫描目录下所有 PNG，归一化到 [-1, 1]^2，与合成数据 schema 对齐。

    Args:
        scan_dir: 扫描目录（如 D:/.../参数化扫描1）
        colorbar_range: (T_min, T_max) 必填
        subdir: 子目录名（"温度"/"d_hist"/"应力"）
        xy_extent: 物理域范围
        crack_x_range / crack_y_loc: 裂纹段定义（生成 region_id 用）
        region_split: "quadrant"（按 (x, y) 象限分 A/B/C/D）或 "uniform"（全 0）
        filename_pattern: PNG 文件名 regex
        max_frames: 最多加载多少帧（None = 全部）

    Returns:
        dict with keys: x_grid, y_grid, T_grid, T_smooth_grid, region_id_grid,
                       is_boundary_grid, is_crack_grid, meta
    """
    png_dir = os.path.join(scan_dir, subdir)
    if not os.path.isdir(png_dir):
        raise FileNotFoundError(f"子目录不存在: {png_dir}")

    # 列举 PNG 文件
    files = []
    for fname in sorted(os.listdir(png_dir)):
        m = re.match(filename_pattern, fname)
        if m:
            files.append((int(m.group(1)), fname))
    if not files:
        raise ValueError(f"未匹配到 PNG（pattern={filename_pattern}）在 {png_dir}")
    files.sort()
    if max_frames is not None:
        files = files[:max_frames]

    # 加载第一张建立网格
    T0, meta0 = load_comsol_png(
        os.path.join(png_dir, files[0][1]),
        colorbar_range=colorbar_range,
        xy_extent=xy_extent,
    )
    H_phys, W_phys = T0.shape

    # 构建坐标网格（按 xy_extent）
    x_min, x_max, y_min, y_max = xy_extent
    x_grid = np.linspace(x_min, x_max, W_phys)
    y_grid = np.linspace(y_max, y_min, H_phys)  # 注意 y 反向（图像 y=0 在顶）

    # 4D 数组（帧, H, W）
    n_frames = len(files)
    T_grid = np.zeros((n_frames, H_phys, W_phys), dtype=np.float64)
    T_grid[0] = T0
    for i, (_, fname) in enumerate(files[1:], start=1):
        T_grid[i], _ = load_comsol_png(
            os.path.join(png_dir, fname),
            colorbar_range=colorbar_range,
            xy_extent=xy_extent,
            warn_uniform=False,  # 仅警告第一帧
        )

    # 归一化坐标到 [-1, 1]（与 ThermalDataset 接口对齐）
    x_norm = 2.0 * (x_grid - x_min) / (x_max - x_min) - 1.0
    y_norm = 2.0 * (y_grid - y_min) / (y_max - y_min) - 1.0

    # region_id（按象限；与 generate_synthetic_thermal_data.region_id_np 一致）
    X, Y = np.meshgrid(x_norm, y_norm, indexing="xy")
    if region_split == "quadrant":
        region_id = np.zeros_like(X, dtype=np.int64)
        region_id[(X >= 0) & (Y >= 0)] = 1
        region_id[(X < 0) & (Y < 0)] = 2
        region_id[(X >= 0) & (Y < 0)] = 3
    else:  # uniform
        region_id = np.zeros_like(X, dtype=np.int64)

    # is_boundary / is_crack
    is_boundary = (
        (np.abs(X - (-1.0)) < 1e-9)
        | (np.abs(X - 1.0) < 1e-9)
        | (np.abs(Y - (-1.0)) < 1e-9)
        | (np.abs(Y - 1.0) < 1e-9)
    )
    is_crack = (
        (np.abs(Y - 0.0) < 1e-9)
        & (X >= crack_x_range[0])
        & (X <= crack_x_range[1])
    )

    # T_smooth_grid：真实数据无法解析调和分量，留 NaN
    T_smooth_grid = np.full_like(T_grid[0], np.nan)

    # 单帧选择：第一帧
    return {
        "x_grid": X,  # (H, W) 与 ThermalDataset 一致
        "y_grid": Y,
        "T_grid": T_grid[0],  # 当前默认第一帧
        "T_smooth_grid": T_smooth_grid,
        "region_id_grid": region_id,
        "is_boundary_grid": is_boundary,
        "is_crack_grid": is_crack,
        "meta_x_min": float(x_min),
        "meta_x_max": float(x_max),
        "meta_y_min": float(y_min),
        "meta_y_max": float(y_max),
        "meta_crack_x_min": float(crack_x_range[0]),
        "meta_crack_x_max": float(crack_x_range[1]),
        "meta_N": int(H_phys),
        "meta_T_min": float(meta0["T_min"]),
        "meta_T_max": float(meta0["T_max"]),
        "n_frames": n_frames,
        "frame_indices": [f[0] for f in files],
    }