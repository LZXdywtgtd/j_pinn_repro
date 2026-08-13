"""P11 COMSOL PNG 加载器单元测试

测试目标（不依赖真实 PNG；用合成的 RGB 数组模拟）：
1. test_physical_region_detection: 自动检测非白像素的方形边界
2. test_colorbar_sampling: 沿色标列采样 RGB 渐变
3. test_rgb_to_colorbar_position: 像素→色标位置 (NN) 正确
4. test_load_comsol_png_basic: 完整 PNG 加载 + meta 字段
5. test_uniform_field_warning: 单色场触发警告
6. test_colorbar_range_required: 必填无默认（type error）
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.comsol_png_loader import (
    _detect_physical_region,
    _rgb_to_colorbar_position,
    _sample_colorbar,
    load_comsol_png,
)


def _make_synthetic_png(
    H: int = 1500, W: int = 2000,
    physical_bbox: tuple = (300, 1100, 400, 1300),  # (top, bot, left, right)
    colorbar_bbox: tuple = (300, 1100, 1320, 1340),
    T_min: float = 100.0, T_max: float = 200.0,
    uniform: bool = False,
) -> np.ndarray:
    """合成一张仿真 COMSOL PNG（H, W, 3）uint8

    物理域内：若 uniform=True，全单色；否则按 [x, y] → 温度梯度
    """
    img = np.full((H, W, 3), 255, dtype=np.uint8)  # 全白
    top, bot, left, right = physical_bbox
    cb_top, cb_bot, cb_left, cb_right = colorbar_bbox
    # 色标：垂直 inferno-like 渐变（蓝→红）
    cb_height = cb_bot - cb_top
    for r in range(cb_top, cb_bot):
        frac = (r - cb_top) / max(cb_height - 1, 1)
        # 简化的 inferno: 蓝(0,0,80) → 黄(255,255,0) → 红(180,0,0)
        if frac < 0.5:
            R = int(frac * 2 * 255)
            G = int(frac * 2 * 255)
            B = int(255 - frac * 2 * 175)
        else:
            R = int(255 - (frac - 0.5) * 2 * 75)
            G = int(255 - (frac - 0.5) * 2 * 255)
            B = int(80 - (frac - 0.5) * 2 * 80)
        for c in range(cb_left, cb_right):
            img[r, c] = [R, G, B]
    # 物理域：按 (x, y) → 温度 → 像素颜色（按色标插值）
    for r in range(top, bot):
        for c in range(left, right):
            if uniform:
                img[r, c] = [200, 100, 50]  # 全单色
            else:
                # y 归一化 [top, bot] → [0, 1]
                y_frac = (r - top) / max(bot - top - 1, 1)
                # x 归一化 [left, right] → [0, 1]
                x_frac = (c - left) / max(right - left - 1, 1)
                # 温度按 y 线性（底冷顶热）
                T_frac = y_frac
                if T_frac < 0.5:
                    R = int(T_frac * 2 * 255)
                    G = int(T_frac * 2 * 255)
                    B = int(255 - T_frac * 2 * 175)
                else:
                    R = int(255 - (T_frac - 0.5) * 2 * 75)
                    G = int(255 - (T_frac - 0.5) * 2 * 255)
                    B = int(80 - (T_frac - 0.5) * 2 * 80)
                img[r, c] = [R, G, B]
    return img


def test_physical_region_detection():
    """自动检测物理域边界"""
    img = _make_synthetic_png()
    top, bot, left, right = _detect_physical_region(img)
    # 应检出 300, 1100, 400, 1300（±2 像素误差，因边界归一化）
    assert abs(top - 300) <= 2, f"top={top}, expected 300"
    assert abs(bot - 1100) <= 2, f"bot={bot}, expected 1100"
    assert abs(left - 400) <= 2, f"left={left}, expected 400"
    assert abs(right - 1300) <= 2, f"right={right}, expected 1300"
    print(f"  ✓ 物理域检测 (top={top}, bot={bot}, left={left}, right={right})")


def test_colorbar_sampling():
    """沿色标列采样 RGB 渐变"""
    img = _make_synthetic_png()
    cb_top, cb_bot, cb_left, cb_right = 300, 1100, 1320, 1340
    cb_rgb = _sample_colorbar(img, cb_top, cb_bot, cb_left, cb_right)
    # cb_rgb 形状 (H, 3)
    H = cb_bot - cb_top
    assert cb_rgb.shape == (H, 3), f"shape={cb_rgb.shape}"
    # 顶部（蓝）→ 底部（红）：R 应递增
    R_top = cb_rgb[0, 0]
    R_bot = cb_rgb[-1, 0]
    assert R_bot > R_top, f"R 应从顶到底递增：top={R_top}, bot={R_bot}"
    print(f"  ✓ 色标采样（顶部 R={R_top:.0f}，底部 R={R_bot:.0f}）")


def test_rgb_to_colorbar_position():
    """像素 → 色标位置 NN 匹配"""
    # 色标：蓝(0,0,80) → 黄(255,255,0) → 红(180,0,0)
    cb_rgb = np.array([
        [0, 0, 80],
        [128, 128, 40],
        [255, 255, 0],
        [200, 50,  0],
        [180, 0,   0],
    ], dtype=np.float64)
    # 像素：纯蓝 → 应映射到位置 0
    pixel_blue = np.array([[[0, 0, 80]]], dtype=np.float64)
    p = _rgb_to_colorbar_position(pixel_blue, cb_rgb)
    assert p.shape == (1, 1)
    assert abs(p[0, 0] - 0.0) < 0.1, f"蓝色应映射到 0，实际 {p[0,0]}"
    # 像素：纯红 → 应映射到位置 1
    pixel_red = np.array([[[180, 0, 0]]], dtype=np.float64)
    p = _rgb_to_colorbar_position(pixel_red, cb_rgb)
    assert abs(p[0, 0] - 1.0) < 0.1, f"红色应映射到 1，实际 {p[0,0]}"
    # 像素：黄 → 应映射到位置 ~0.5
    pixel_yellow = np.array([[[255, 255, 0]]], dtype=np.float64)
    p = _rgb_to_colorbar_position(pixel_yellow, cb_rgb)
    assert abs(p[0, 0] - 0.5) < 0.15, f"黄色应映射到 0.5，实际 {p[0,0]}"
    print(f"  ✓ RGB → 色标位置（蓝=0、黄=0.5、红=1）")


def test_load_comsol_png_basic(tmp_path):
    """完整加载（用合成的 PNG）"""
    import imageio.v2 as imageio
    png_path = tmp_path / "test_thermal.png"
    img = _make_synthetic_png(uniform=False)
    imageio.imwrite(str(png_path), img)
    T_field, meta = load_comsol_png(
        str(png_path),
        colorbar_range=(100.0, 200.0),
        xy_extent=(0.0, 0.01, 0.0, 0.01),
        warn_uniform=False,  # 测试用 gradient 场
    )
    # 形状：物理域尺寸 (1100-300, 1300-400) = (800, 900)
    assert T_field.shape == (800, 900), f"T_field shape={T_field.shape}"
    # T 范围应在 [100, 200]
    assert T_field.min() >= 99.0, f"T_min={T_field.min()}"
    assert T_field.max() <= 201.0, f"T_max={T_field.max()}"
    # meta 字段
    assert meta["T_min"] == 100.0
    assert meta["T_max"] == 200.0
    assert meta["source"] == "comsol_png"
    assert meta["pixel_shape"] == (800, 900)
    assert meta["multiplier"] == 1.0
    print(f"  ✓ 完整加载：T_field {T_field.shape}，T ∈ [{T_field.min():.1f}, {T_field.max():.1f}]")


def test_uniform_field_warning(tmp_path):
    """单色场触发警告"""
    import warnings
    import imageio.v2 as imageio
    png_path = tmp_path / "test_uniform.png"
    img = _make_synthetic_png(uniform=True)
    imageio.imwrite(str(png_path), img)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        T_field, _ = load_comsol_png(
            str(png_path),
            colorbar_range=(100.0, 200.0),
            warn_uniform=True,
        )
        assert any("均匀" in str(wi.message) for wi in w), "应触发均匀场警告"
    print(f"  ✓ 单色场警告（标准差={T_field.std():.4f}）")


def test_colorbar_range_required():
    """colorbar_range 必填：传 None 或缺省应报 TypeError 或 ValueError"""
    import imageio.v2 as imageio
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        png_path = Path(td) / "test.png"
        img = _make_synthetic_png()
        imageio.imwrite(str(png_path), img)
        # 缺省：函数签名 colorbar_range 无默认值 → 必报错
        try:
            load_comsol_png(str(png_path))  # 不传 colorbar_range
            assert False, "应报错"
        except TypeError:
            print(f"  ✓ colorbar_range 缺省 → TypeError")


def main():
    print("\n=== P11 COMSOL PNG 加载器单元测试 ===\n")
    tests = [
        ("test_physical_region_detection", test_physical_region_detection),
        ("test_colorbar_sampling", test_colorbar_sampling),
        ("test_rgb_to_colorbar_position", test_rgb_to_colorbar_position),
        ("test_colorbar_range_required", test_colorbar_range_required),
    ]
    for name, fn in tests:
        print(f"[{name}]")
        fn()
        print()
    # 需要 tmp_path 的测试
    print("[test_load_comsol_png_basic]")
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        test_load_comsol_png_basic(Path(td))
    print()
    print("[test_uniform_field_warning]")
    with tempfile.TemporaryDirectory() as td:
        test_uniform_field_warning(Path(td))
    print()
    print("=== ALL TESTS PASSED ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())