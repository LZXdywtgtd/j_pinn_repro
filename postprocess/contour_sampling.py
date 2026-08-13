"""
矩形轮廓采样（论文 §4.4 Fig.12 风格）

每个轮廓是一个 4 段的矩形：
- 左竖线：x = x_lig, y ∈ [y_min, y_max]（韧带侧 / uncracked）
- 上横线：y = y_max, x ∈ [x_lig, x_wake]
- 右竖线：x = x_wake, y ∈ [y_max, y_min]（尾迹侧 / 裂纹面 / traction-free）
- 下横线：y = y_min, x ∈ [x_wake, x_lig]

外法向 n 在每段上分别为 (-1, 0) / (0, +1) / (+1, 0) / (0, -1)。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import torch


@dataclass(frozen=True)
class RectContour:
    """矩形轮廓（绕裂纹尖端 x_tip, y_tip）"""
    x_tip: float = 0.0
    y_tip: float = 0.0
    x_lig: float = -0.3      # 韧带侧 x 坐标（uncracked material）
    x_wake: float = +0.3     # 尾迹侧 x 坐标（traction-free crack faces）
    y_min: float = -1.0
    y_max: float = +1.0
    n_per_side: int = 200    # 每段采样点
    crack_x_min: float = -0.5
    crack_x_max: float = +0.5

    def sample(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """返回 (x_path, y_path, n_x_path, n_y_path)

        4 段顺序：左竖（y_min→y_max）→ 上横（x_lig→x_wake）→
                  右竖（y_max→y_min）→ 下横（x_wake→x_lig）
        每段含 n_per_side 个点；首点=(x_lig, y_min)，末点=(x_lig, y_min)
        """
        n = self.n_per_side
        # 段 1：左竖（x=x_lig, y 从 y_min 升到 y_max）
        yt = np.linspace(self.y_min, self.y_max, n, dtype=np.float64)
        x1 = np.full(n, self.x_lig, dtype=np.float64)
        # 段 2：上横（y=y_max, x 从 x_lig 增到 x_wake）
        xt = np.linspace(self.x_lig, self.x_wake, n, dtype=np.float64)
        x2 = xt
        y2 = np.full(n, self.y_max, dtype=np.float64)
        # 段 3：右竖（x=x_wake, y 从 y_max 降到 y_min）
        yt3 = np.linspace(self.y_max, self.y_min, n, dtype=np.float64)
        x3 = np.full(n, self.x_wake, dtype=np.float64)
        y3 = yt3
        # 段 4：下横（y=y_min, x 从 x_wake 减到 x_lig）
        xt4 = np.linspace(self.x_wake, self.x_lig, n, dtype=np.float64)
        x4 = xt4
        y4 = np.full(n, self.y_min, dtype=np.float64)

        x = np.concatenate([x1, x2, x3, x4])
        y = np.concatenate([yt, y2, y3, y4])

        # 外法向：段1=(-1,0), 段2=(0,+1), 段3=(+1,0), 段4=(0,-1)
        n_x = np.concatenate([
            np.full(n, -1.0),
            np.zeros(n),
            np.full(n, +1.0),
            np.zeros(n),
        ])
        n_y = np.concatenate([
            np.zeros(n),
            np.full(n, +1.0),
            np.zeros(n),
            np.full(n, -1.0),
        ])
        return x, y, n_x, n_y

    def height(self) -> float:
        return self.y_max - self.y_min


def sweep_contours(
    x_lig_values: Tuple[float, ...] = (-0.9, -0.7, -0.5, -0.3, -0.1),
    x_wake_values: Tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 0.9),
    y_range: Tuple[float, float] = (-1.0, 1.0),
    n_per_side: int = 200,
    crack_x_range: Tuple[float, float] = (-0.5, 0.5),
) -> List[RectContour]:
    """生成 (N_lig × N_wake) 个矩形 contour"""
    contours: List[RectContour] = []
    for x_lig in x_lig_values:
        for x_wake in x_wake_values:
            contours.append(RectContour(
                x_lig=x_lig,
                x_wake=x_wake,
                y_min=y_range[0],
                y_max=y_range[1],
                n_per_side=n_per_side,
                crack_x_min=crack_x_range[0],
                crack_x_max=crack_x_range[1],
            ))
    return contours


def contour_to_tensor(
    contour: RectContour,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """RectContour.sample() → 4 个 tensor（GPU/CPU 可）"""
    x, y, n_x, n_y = contour.sample()
    return (
        torch.as_tensor(x, dtype=dtype, device=device),
        torch.as_tensor(y, dtype=dtype, device=device),
        torch.as_tensor(n_x, dtype=dtype, device=device),
        torch.as_tensor(n_y, dtype=dtype, device=device),
    )