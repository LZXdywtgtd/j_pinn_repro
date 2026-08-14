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
    colorbar_bbox: tuple = (300, 1100, 1360, 1380),  # 与物理域留 60px 间隔
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


def _make_comsol_style_png(
    H: int = 1500, W: int = 2000,
    physical_bbox: tuple = (101, 1440, 135, 1725),  # (top, bot, left, right) 真实 COMSOL 图结构
    colorbar_bbox: tuple = (149, 1424, 1783, 1855),  # 与物理域之间有 ~58px 白间隙
) -> np.ndarray:
    """合成一张**复现真实失败模式**的 COMSOL 图（v0.8 回归测试用）

    真实 COMSOL 温度 PNG 的两个关键特征（旧算法因此失败）：
    1. 物理域内部大片**近白区**（高温区在 inferno 顶端渲染成 [255,255,255]）
       → 每列非白像素占比仅 ~20%（旧算法"列密度 > 0.5*height"阈值被击穿）
    2. 色标条密集渐变（每列 ~57% 有色）且与物理域之间有白间隙
       → 旧算法把色标误判为物理域；旧采样带假设"色标紧贴物理域右侧"也失效
    """
    img = np.full((H, W, 3), 255, dtype=np.uint8)  # 全白
    top, bot, left, right = physical_bbox
    cb_top, cb_bot, cb_left, cb_right = colorbar_bbox

    # 色标条：密集垂直渐变（模拟真实色标：~57% 行非白）
    cb_height = cb_bot - cb_top
    for r in range(cb_top, cb_bot):
        frac = (r - cb_top) / max(cb_height - 1, 1)
        # 简化 inferno：蓝(0,0,80) → 黄(255,255,0) → 红(180,0,0)
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

    # 物理域：每行散布 ~20% 有色像素（模拟真实图：高温区近白但裂纹线/边界线散布全高）
    # 真实实测：物理域每列非白像素 6~1353 个（中位数 33），每行几乎都有非白像素
    rng = np.random.default_rng(42)
    n_rows = bot - top
    n_cols = right - left
    scatter_mask = rng.random((n_rows, n_cols)) < 0.20
    # 有色像素：蓝色系（低温区颜色），其余保持纯白（高温近白区）
    for r in range(n_rows):
        frac = r / max(n_rows - 1, 1)
        R = int(frac * 60)
        G = int(frac * 120)
        B = int(255 - frac * 175)
        cols = np.where(scatter_mask[r])[0]
        img[top + r, left + cols, 0] = R
        img[top + r, left + cols, 1] = G
        img[top + r, left + cols, 2] = B
    return img


def test_physical_region_detection_comsol_style():
    """回归测试（v0.8）：物理域含大片近白区 + 色标有间隙 → 应检出物理域而非色标"""
    img = _make_comsol_style_png()
    top, bot, left, right = _detect_physical_region(img)
    # 期望检出物理域 (101, 1440, 135, 1725)，而非色标 (149, 1424, 1783, 1855)
    assert abs(top - 101) <= 3, f"top={top}，期望 101（物理域顶，非标题/色标顶）"
    assert abs(bot - 1440) <= 3, f"bot={bot}，期望 1440（物理域底）"
    assert abs(left - 135) <= 3, f"left={left}，期望 135（物理域左）"
    assert abs(right - 1725) <= 3, f"right={right}，期望 1725（物理域右）"
    width = right - left
    assert width > 1500, f"物理域宽={width}，期望 ~1590（远大于色标 72）"
    print(f"  ✓ COMSOL 风格物理域检测（含近白区 + 色标间隙）：({top},{bot},{left},{right}) 宽={width}")


def test_load_comsol_png_comsol_style(tmp_path):
    """回归测试（v0.8）：真实失败模式端到端——读出的 T 范围应覆盖 colorbar_range"""
    import imageio.v2 as imageio
    png_path = tmp_path / "comsol_style.png"
    img = _make_comsol_style_png()
    imageio.imwrite(str(png_path), img)
    T_field, meta = load_comsol_png(
        str(png_path),
        colorbar_range=(293.15, 1431.15),
        xy_extent=(0.0, 0.01, 0.0, 0.01),
        warn_uniform=False,
    )
    # 形状：物理域 (1440-101, 1725-135) = (1339, 1590)
    assert T_field.shape == (1339, 1590), f"T_field shape={T_field.shape}，期望 (1339, 1590)"
    # T 不应是常数（旧 bug 读出全底色 293.15）
    t_min, t_max = T_field.min(), T_field.max()
    assert t_max - t_min > 100.0, f"T 范围 {t_min:.1f}~{t_max:.1f} 过窄，疑似读数失败（旧 bug 全 293.15）"
    # 色标 bbox 应为检测到的完整 bbox（非紧贴物理域的假设带）
    cb = meta["colorbar_bbox_px"]
    assert cb[2] > 1700, f"色标左缘 {cb[2]} 应 > 1700（真实图有间隙），疑似仍用旧假设带"
    print(f"  ✓ COMSOL 风格端到端：T ∈ [{t_min:.1f}, {t_max:.1f}]，色标 bbox={cb}")


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
    cb_top, cb_bot, cb_left, cb_right = 300, 1100, 1360, 1380
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


# ============================================================
# D1 多扫描 batch loader 单元测试（v0.5 新增）
# ============================================================
def _make_scan_dir(tmp_path, n_frames: int = 3, subdir_name: str = "温度",
                   field_pattern: str = "温度", uniform: bool = False) -> Path:
    """合成一个扫描目录：<tmp>/<subdir_name>/温度001.png ..."""
    import imageio.v2 as imageio
    scan_dir = Path(tmp_path) / "scan"
    sub = scan_dir / subdir_name
    sub.mkdir(parents=True, exist_ok=True)
    for i in range(1, n_frames + 1):
        img = _make_synthetic_png(uniform=uniform)
        imageio.imwrite(str(sub / f"{field_pattern}{i:03d}.png"), img)
    return scan_dir


def test_scan_dir_numeric_sort(tmp_path):
    """文件名数字排序（001 < 002 < 010）"""
    import imageio.v2 as imageio
    scan_dir = Path(tmp_path) / "scan"
    sub = scan_dir / "温度"
    sub.mkdir(parents=True, exist_ok=True)
    # 故意乱序文件名
    for name in ["温度010.png", "温度002.png", "温度001.png"]:
        img = _make_synthetic_png()
        imageio.imwrite(str(sub / name), img)
    from data.comsol_png_loader import load_comsol_scan_dir
    data = load_comsol_scan_dir(str(scan_dir), colorbar_range=(100.0, 200.0))
    assert data["frame_indices"] == [1, 2, 10], f"frame_indices={data['frame_indices']}"
    print(f"  ✓ 数字排序：{data['frame_indices']}")


def test_scan_dir_returns_4d_volume(tmp_path):
    """返回 (N_frames, H, W) float64"""
    from data.comsol_png_loader import load_comsol_scan_dir
    scan_dir = _make_scan_dir(tmp_path, n_frames=3)
    data = load_comsol_scan_dir(str(scan_dir), colorbar_range=(100.0, 200.0))
    vol = data["T_grid_volume"]
    assert vol.shape[0] == 3, f"n_frames={vol.shape[0]}"
    assert vol.dtype == np.float64, f"dtype={vol.dtype}"
    assert len(vol.shape) == 3, f"shape={vol.shape}"
    print(f"  ✓ 4D volume: {vol.shape} dtype={vol.dtype}")


def test_scan_dir_first_frame_matches_single_loader(tmp_path):
    """scan_dir 第一帧 == load_comsol_png 单帧结果"""
    from data.comsol_png_loader import load_comsol_scan_dir, load_comsol_png
    scan_dir = _make_scan_dir(tmp_path, n_frames=3)
    data = load_comsol_scan_dir(str(scan_dir), colorbar_range=(100.0, 200.0))
    # 单帧加载同一 PNG
    single, _ = load_comsol_png(
        str(scan_dir / "温度" / "温度001.png"),
        colorbar_range=(100.0, 200.0),
    )
    assert np.allclose(data["T_grid"], single, atol=1e-10), "首帧与单帧加载不一致"
    print(f"  ✓ 首帧与单帧加载一致（max|Δ|={np.abs(data['T_grid'] - single).max():.2e}）")


def test_scan_dir_field_auto_detect(tmp_path):
    """field auto-detect：有 温度/ 和 应力/ 时按 field 命中"""
    import imageio.v2 as imageio
    from data.comsol_png_loader import load_comsol_scan_dir
    scan_dir = Path(tmp_path) / "scan"
    for sub in ("温度", "应力", "d_hist"):
        (scan_dir / sub).mkdir(parents=True, exist_ok=True)
        for i in range(1, 3):
            img = _make_synthetic_png(uniform=True)
            imageio.imwrite(str(scan_dir / sub / f"{sub}{i:03d}.png"), img)
    # 用 field="应力" → 应命中 应力 子目录
    data = load_comsol_scan_dir(str(scan_dir), colorbar_range=(100.0, 200.0), field="应力")
    assert data["field"] == "应力", f"field={data['field']}"
    print(f"  ✓ field auto-detect：field={data['field']}，n_frames={data['n_frames']}")


def test_scan_dir_max_frames(tmp_path):
    """max_frames 截断"""
    from data.comsol_png_loader import load_comsol_scan_dir
    scan_dir = _make_scan_dir(tmp_path, n_frames=5)
    data = load_comsol_scan_dir(str(scan_dir), colorbar_range=(100.0, 200.0), max_frames=3)
    assert data["n_frames"] == 3, f"n_frames={data['n_frames']}"
    assert data["T_grid_volume"].shape[0] == 3
    print(f"  ✓ max_frames=3 → n_frames=3")


def test_scan_dir_as_array_wrapper(tmp_path):
    """load_comsol_scan_dir_as_array 返回 (T_volume, loaded_field_name, meta)"""
    from data.comsol_png_loader import load_comsol_scan_dir_as_array
    scan_dir = _make_scan_dir(tmp_path, n_frames=3)
    T_vol, field_name, meta = load_comsol_scan_dir_as_array(str(scan_dir), colorbar_range=(100.0, 200.0))
    assert T_vol.shape == (3, 800, 900), f"T_vol.shape={T_vol.shape}"
    assert T_vol.dtype == np.float64
    assert field_name == "温度", f"loaded_field_name={field_name}"
    assert meta["n_frames"] == 3
    assert meta["field"] == "温度"
    assert "frame_indices" in meta
    assert "colorbar_range" in meta
    print(f"  ✓ as_array wrapper: T_vol {T_vol.shape} dtype={T_vol.dtype} field={field_name}")


def test_scan_dir_no_matching_files(tmp_path):
    """空子目录 → FileNotFoundError（用户 v0.5 决策：0 子目录即报错）"""
    from data.comsol_png_loader import load_comsol_scan_dir
    scan_dir = Path(tmp_path) / "scan"
    scan_dir.mkdir(parents=True, exist_ok=True)
    try:
        load_comsol_scan_dir(str(scan_dir), colorbar_range=(100.0, 200.0))
        assert False, "应报 FileNotFoundError"
    except FileNotFoundError:
        print(f"  ✓ 空子目录 → FileNotFoundError")


def test_scan_dir_missing_subdir(tmp_path):
    """field 指定不存在 → FileNotFoundError（用户 v0.5 决策：严格报错）"""
    from data.comsol_png_loader import load_comsol_scan_dir
    scan_dir = Path(tmp_path) / "scan"
    (scan_dir / "温度").mkdir(parents=True, exist_ok=True)
    try:
        load_comsol_scan_dir(str(scan_dir), colorbar_range=(100.0, 200.0), field="d_hist")
        assert False, "应报 FileNotFoundError"
    except FileNotFoundError as e:
        assert "d_hist" in str(e), f"错误应列出指定字段: {e}"
        print(f"  ✓ field='d_hist' 不存在 → FileNotFoundError（列出实际）")


def test_scan_dir_auto_detect_multiple(tmp_path):
    """多个子目录 → 字母序选第一个 + [INFO] 提示"""
    import imageio.v2 as imageio
    from data.comsol_png_loader import load_comsol_scan_dir
    scan_dir = Path(tmp_path) / "scan"
    # 构造两个子目录（d_hist 字母序在 温度 前）
    for sub in ("温度", "d_hist"):
        (scan_dir / sub).mkdir(parents=True, exist_ok=True)
        for i in range(1, 3):
            img = _make_synthetic_png(uniform=True)
            imageio.imwrite(str(scan_dir / sub / f"{sub}{i:03d}.png"), img)
    data = load_comsol_scan_dir(str(scan_dir), colorbar_range=(100.0, 200.0))
    # sorted(['d_hist', '温度']) 按字符序：d_hist 的 'd'(0x64) < '温'(0x6E29)
    # Python sorted 对中文字符用 Unicode 码点：'d'(100) < '温'(28201) → d_hist 在前
    assert data["field"] == "d_hist", f"自动选第一个 field={data['field']}"
    print(f"  ✓ 多子目录自动选第一个：{data['field']}（字母序）")


def test_scan_dir_uniform_field_suppressed(tmp_path):
    """3 张单色 PNG 只触发 1 个均匀场警告（仅首帧）"""
    import warnings
    from data.comsol_png_loader import load_comsol_scan_dir
    scan_dir = _make_scan_dir(tmp_path, n_frames=3, uniform=True)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        load_comsol_scan_dir(str(scan_dir), colorbar_range=(100.0, 200.0))
        uniform_warns = [wi for wi in w if "均匀" in str(wi.message)]
        assert len(uniform_warns) == 1, f"应只有 1 个均匀场警告，实际 {len(uniform_warns)}"
    print(f"  ✓ 3 张单色 PNG 只 1 个警告")


def main():
    print("\n=== P11 COMSOL PNG 加载器单元测试 ===\n")
    tests = [
        ("test_physical_region_detection", test_physical_region_detection),
        ("test_physical_region_detection_comsol_style", test_physical_region_detection_comsol_style),
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
    print("[test_load_comsol_png_comsol_style]")
    with tempfile.TemporaryDirectory() as td:
        test_load_comsol_png_comsol_style(Path(td))
    print()
    # D1 多扫描测试
    print("\n=== D1 多扫描 batch loader 测试 ===\n")
    d1_tests = [
        ("test_scan_dir_numeric_sort", test_scan_dir_numeric_sort),
        ("test_scan_dir_returns_4d_volume", test_scan_dir_returns_4d_volume),
        ("test_scan_dir_first_frame_matches_single_loader", test_scan_dir_first_frame_matches_single_loader),
        ("test_scan_dir_field_auto_detect", test_scan_dir_field_auto_detect),
        ("test_scan_dir_max_frames", test_scan_dir_max_frames),
        ("test_scan_dir_as_array_wrapper", test_scan_dir_as_array_wrapper),
        ("test_scan_dir_no_matching_files", test_scan_dir_no_matching_files),
        ("test_scan_dir_missing_subdir", test_scan_dir_missing_subdir),
        ("test_scan_dir_auto_detect_multiple", test_scan_dir_auto_detect_multiple),
        ("test_scan_dir_uniform_field_suppressed", test_scan_dir_uniform_field_suppressed),
    ]
    for name, fn in d1_tests:
        print(f"[{name}]")
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            fn(Path(td))
        print()
    print("=== ALL TESTS PASSED ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())