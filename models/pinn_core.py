"""
J-PINN 模型核心：4 区域分解 MLP（论文 §2.3 / §4.2）

架构（论文选定配置）：
- 每个子网络：4 隐藏层 × 64 神经元 + SiLU 激活 + LayerNorm
- 输入：归一化坐标 (x̃, ỹ) ∈ [-1, 1]^2
- 输出：归一化温度 T̃ ∈ [-1, 1]
- 总参数量 ≈ 7-9 万（论文 §2.3 报告 71,712；本实现含 LayerNorm 略偏多）

设计要点：
- 4 个独立 MLP（mlp_A/B/C/D），无参数共享
- Forward 按 region_id 路由到对应子网络
- 支持消融变体：SingleRegionPINN（1 个 MLP）+ TwoRegionPINN（上下分割）

参考实现：
- v4 MLP block 模式：projects/pe_mmnet/project_v4/models/pe_tsnet_multimodal.py:1017-1027
  （Linear → LayerNorm → Activation → Dropout），但用 SiLU 替 ReLU
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


# ============================================================
# 单个 FNN 子网络（4 隐藏层 × 64 神经元）
# ============================================================
class MLPBlock(nn.Module):
    """
    论文 §4.2 选定的 FNN 结构：
    - 4 个隐藏层 × 64 神经元（论文 "4 layers" 指隐藏层数）
    - SiLU 激活（论文对比 Tanh 后选择）
    - Linear → LayerNorm → SiLU → Dropout 模式
    - 最后一层无激活（线性输出 T̃）

    注：论文参数 ~17,928/子网 已按 6 个 Linear 层核算（1 输入投影 + 4 隐藏 + 1 输出），
    与论文 §2.3 报告值一致。
    """

    def __init__(
        self,
        in_dim: int = 2,
        hidden: int = 64,
        n_hidden_layers: int = 4,  # 改名为 n_hidden_layers 避免歧义
        out_dim: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if n_hidden_layers < 1:
            raise ValueError("n_hidden_layers 必须 >= 1")

        layers: list[nn.Module] = []
        # 输入投影：in_dim → hidden（不计为隐藏层）
        layers += [
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        # 隐藏层：hidden → hidden，共 n_hidden_layers 个
        for _ in range(n_hidden_layers):
            layers += [
                nn.Linear(hidden, hidden),
                nn.LayerNorm(hidden),
                nn.SiLU(),
            ]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        # 输出层：hidden → out_dim（无激活）
        layers.append(nn.Linear(hidden, out_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        """
        Args:
            xy: (N, in_dim) 坐标（已归一化）
        Returns:
            (N, out_dim) 输出
        """
        return self.net(xy)


# ============================================================
# 4 区域 JPINN（论文主模型）
# ============================================================
class JPINN(nn.Module):
    """
    论文 §2.3 的 J-PINN 完整架构：4 个独立 FNN + 区域路由。

    Forward 路由策略：
    - 输入 (x, y, region_id) 其中 region_id ∈ {0,1,2,3}
    - 按 region_id 用布尔掩码切分到对应子网络
    - 输出 (N,) 归一化温度
    """

    def __init__(
        self,
        in_dim: int = 2,
        hidden: int = 64,
        n_hidden_layers: int = 4,
        dropout: float = 0.0,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        self.mlp_A = MLPBlock(in_dim, hidden, n_hidden_layers, 1, dropout)
        self.mlp_B = MLPBlock(in_dim, hidden, n_hidden_layers, 1, dropout)
        self.mlp_C = MLPBlock(in_dim, hidden, n_hidden_layers, 1, dropout)
        self.mlp_D = MLPBlock(in_dim, hidden, n_hidden_layers, 1, dropout)
        self.sub_nets = [self.mlp_A, self.mlp_B, self.mlp_C, self.mlp_D]

        # 整体默认 float64（论文 §4.5 注：裂纹尖端需要更高精度）
        self.to(dtype)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        region_id: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: (N,) x 坐标（归一化）
            y: (N,) y 坐标（归一化）
            region_id: (N,) ∈ {0,1,2,3}
        Returns:
            T_pred: (N,) 归一化温度预测
        """
        if x.shape != y.shape or x.shape != region_id.shape:
            raise ValueError(
                f"x/y/region_id 形状不一致: {x.shape} / {y.shape} / {region_id.shape}"
            )

        xy = torch.stack([x, y], dim=-1)  # (N, 2)
        out = torch.empty_like(x)
        for rid, mlp in enumerate(self.sub_nets):
            mask = region_id == rid
            if mask.any():
                out[mask] = mlp(xy[mask]).squeeze(-1)
        return out

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================
# 消融变体：单区域 PINN（论文 §4.6）
# ============================================================
class SingleRegionPINN(nn.Module):
    """单个 MLP 覆盖整个域（无区域分解基线）"""

    def __init__(
        self,
        in_dim: int = 2,
        hidden: int = 128,  # 单区域给更大容量以做公平对比
        n_hidden_layers: int = 5,
        dropout: float = 0.0,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        self.mlp = MLPBlock(in_dim, hidden, n_hidden_layers, 1, dropout)
        self.to(dtype)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        region_id: torch.Tensor | None = None,  # 忽略
    ) -> torch.Tensor:
        xy = torch.stack([x, y], dim=-1)
        return self.mlp(xy).squeeze(-1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def num_networks(self) -> int:
        return 2


# ============================================================
# 消融变体：上下分割（2 区域）
# ============================================================
class TwoRegionPINN(nn.Module):
    """上下分割的 2 区域 PINN（论文 §4.6 中间基线）"""

    def __init__(
        self,
        in_dim: int = 2,
        hidden: int = 64,
        n_hidden_layers: int = 4,
        dropout: float = 0.0,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        # region_id {0,1} → 上半（y>=0）；{2,3} → 下半（y<0）
        # 强制重映射：A/B → 0, C/D → 1
        self.mlp_upper = MLPBlock(in_dim, hidden, n_hidden_layers, 1, dropout)
        self.mlp_lower = MLPBlock(in_dim, hidden, n_hidden_layers, 1, dropout)
        self.to(dtype)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        region_id: torch.Tensor,
    ) -> torch.Tensor:
        xy = torch.stack([x, y], dim=-1)
        out = torch.empty_like(x)
        # 上半
        upper_mask = (region_id == 0) | (region_id == 1)
        if upper_mask.any():
            out[upper_mask] = self.mlp_upper(xy[upper_mask]).squeeze(-1)
        # 下半
        lower_mask = (region_id == 2) | (region_id == 3)
        if lower_mask.any():
            out[lower_mask] = self.mlp_lower(xy[lower_mask]).squeeze(-1)
        return out

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def num_networks(self) -> int:
        return 1


# ============================================================
# 模型工厂
# ============================================================
def build_model(ablation: str = "full", dtype: torch.dtype = torch.float64) -> nn.Module:
    """
    按消融标签构建模型。
    ablation ∈ {"full", "two", "single"} → 分别 JPINN / TwoRegionPINN / SingleRegionPINN
    """
    if ablation == "full":
        return JPINN(dtype=dtype)
    if ablation == "two":
        return TwoRegionPINN(dtype=dtype)
    if ablation == "single":
        return SingleRegionPINN(dtype=dtype)
    raise ValueError(f"Unknown ablation: {ablation}")