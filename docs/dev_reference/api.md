# API 参考文档

> J-PINN 公开 API | 版本 0.1
>
> 所有签名与默认值与源码一致。

---

## 1. 数据模块 (`data`)

### 1.1 `data.generate_synthetic_thermal_data`

```python
def T_exact_np(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """完整温度场（含裂纹间断）"""
def T_smooth_np(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """严格调和部分（∇²T=0 解析成立）"""
def T_crack_np(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """裂纹间断项"""
def region_id_np(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """4 区域 ID {0=A, 1=B, 2=C, 3=D}"""
def main(N: int = 200, out_path: str | None = None) -> None:
    """主入口：生成 N×N 网格 → .npz"""
```

### 1.2 `data.dataset`

```python
@dataclass
class ThermalDataset:
    npz_path: str = "data/synthetic_thermal.npz"
    source: str = "synthetic"   # "synthetic" | "comsol_png"
    device: torch.device | str = "cpu"
    dtype: torch.dtype = torch.float64
    T_min: float = 0.0
    T_max: float = 1.0
    spec: DomainSpec = DEFAULT_DOMAIN

    def __post_init__(self) -> None: ...
    def boundary_target(self, x, y) -> torch.Tensor:
        """外边界点对应的真实温度（归一化 [-1, 1]）"""
    def get_collocation_batch(
        self,
        n_int_per_region: int = 2500,
        n_bc_per_edge: int = 100,
        n_iface_per_seam: int = 50,
        n_crack_per_side: int = 50,
        seed: int | None = None,
    ) -> dict:
        """每 epoch 重新采样的配点 batch"""
```

**batch schema**：
```python
{
    "interior": (x, y, region_id),                 # 域内
    "boundary": {"x", "y", "edge_id", "T_target"}, # 外边界
    "interface": {
        "A_B": (x_l, y_l, rid_l, x_r, y_r, rid_r),  # 3 条缝合
        "A_C": ...,
        "B_D": ...,
    },
    "crack": {
        "top": (x, y, rid),                        # 裂纹段上侧
        "bot": (x, y, rid),                        # 裂纹段下侧
        "T_jump_value": float,
        "dT_jump_value": float,
    },
    "meta": {"T_min", "T_max"},
}
```

### 1.3 `data.comsol_png_loader`（v0.5 完整实现）

```python
def load_comsol_png(
    png_path: str,
    colorbar_range: Tuple[float, float],          # 必填无默认（用户约束）
    *,
    xy_extent: Tuple[float, float, float, float] = (0.0, 0.01, 0.0, 0.01),
    multiplier: Optional[float] = None,
    colormap_hint: str = "inferno",
    expected_image_shape: Tuple[int, int] = (1500, 2000),
    warn_uniform: bool = True,
) -> Tuple[np.ndarray, dict]:
    """单帧加载：读 PNG → 自动检测物理域/色标 → 像素 RGB → 温度 K。
    Returns: (T_field (H', W') float64, meta dict)"""

def load_comsol_scan_dir(
    scan_dir: str,
    colorbar_range: Tuple[float, float],
    *,
    field: Optional[str] = None,                 # None = 动态发现（用户 v0.5 决策）
    subdir: Optional[str] = None,
    file_pattern: Optional[str] = None,          # None = natural sort
    xy_extent: Tuple[float, float, float, float] = (0.0, 0.01, 0.0, 0.01),
    crack_x_range: Tuple[float, float] = (-0.5, 0.5),
    crack_y_loc: float = 0.0,
    region_split: str = "quadrant",
    max_frames: Optional[int] = None,
    multiplier: Optional[float] = None,
    colormap_hint: str = "inferno",
) -> dict:
    """多扫描加载：读整个 温度/ 子目录所有 PNG → (N_frames, H, W) 序列。
    返回 dict：x_grid, y_grid, T_grid(首帧2D), T_grid_volume(4D), T_smooth_grid,
      region_id_grid, is_boundary_grid, is_crack_grid, meta_*, n_frames, frame_indices,
      scan_dir, field"""

def load_comsol_scan_dir_as_array(
    scan_dir: str,
    colorbar_range: Tuple[float, float],
    *,
    field: Optional[str] = None,
    file_pattern: Optional[str] = None,
    xy_extent: Tuple[float, float, float, float] = (0.0, 0.01, 0.0, 0.01),
    multiplier: Optional[float] = None,
    colormap_hint: str = "inferno",
    max_frames: Optional[int] = None,
) -> Tuple[np.ndarray, str, dict]:
    """简洁形式：返回 (T_volume (N,H,W) float64, loaded_field_name, meta dict)"""
```

**字段自动检测**（v0.5 用户决策，无硬编码优先级）：
- `field=None`：扫描子目录数 0 → FileNotFoundError；1 个 → 加载该目录；多个 → 按字母序取第一个 + [INFO] 提示
- `field="温度"`：严格查找；不存在 → FileNotFoundError

---

## 2. 模型模块 (`models.pinn_core`)

### 2.1 `MLPBlock`

```python
class MLPBlock(nn.Module):
    def __init__(
        self,
        in_dim: int = 2,
        hidden: int = 64,
        n_hidden_layers: int = 4,  # 论文 "4 layers" 实指隐藏层
        out_dim: int = 1,
        dropout: float = 0.0,
    ) -> None: ...

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        """输入 (N, in_dim) → 输出 (N, out_dim)"""
```

参数量（含 LayerNorm）：约 21,500/子网。

### 2.2 `JPINN`

```python
class JPINN(nn.Module):
    def __init__(
        self,
        in_dim: int = 2,
        hidden: int = 64,
        n_hidden_layers: int = 4,
        dropout: float = 0.0,
        dtype: torch.dtype = torch.float64,
    ) -> None: ...

    def forward(self, x, y, region_id) -> torch.Tensor:
        """输入 (x, y, region_id) 各 (N,) → 输出 (N,) 归一化温度"""

    def count_parameters(self) -> int:
        """返回可训练参数总数（约 70,148）"""
```

### 2.3 `TwoRegionPINN` / `SingleRegionPINN`

```python
class TwoRegionPINN(nn.Module):
    """上下分割 2 区域；region {0,1}→upper, {2,3}→lower"""
    def __init__(self, in_dim=2, hidden=64, n_hidden_layers=4,
                 dropout=0.0, dtype=torch.float64): ...
    def forward(self, x, y, region_id) -> torch.Tensor: ...
    def count_parameters(self) -> int: ...   # ~35,074

class SingleRegionPINN(nn.Module):
    """单 MLP（hidden=128, 5 隐藏层）"""
    def __init__(self, in_dim=2, hidden=128, n_hidden_layers=5,
                 dropout=0.0, dtype=torch.float64): ...
    def forward(self, x, y, region_id=None) -> torch.Tensor: ...
    def count_parameters(self) -> int: ...   # ~84,609
```

### 2.4 `build_model`

```python
def build_model(ablation: str = "full", dtype: torch.dtype = torch.float64) -> nn.Module:
    """ablation ∈ {"full", "two", "single"} → 对应 JPINN/TwoRegionPINN/SingleRegionPINN"""
```

---

## 3. 损失模块 (`losses`)

### 3.1 损失函数

```python
def pde_residual(model, x, y, region_id) -> torch.Tensor:
    """∇²T = ∂²T/∂x² + ∂²T/∂y²；每点残差 (N,)"""

def pde_loss(model, x, y, region_id) -> torch.Tensor:
    """L_pde = mean(r²)"""

def interface_loss(model, ifaces: dict) -> torch.Tensor:
    """3 条缝合 seam 温度连续 MSE"""
    # ifaces: dict[key] = (x_l, y_l, rid_l, x_r, y_r, rid_r)

def bc_loss_dirichlet(
    model, x_b, y_b, region_id_b, T_target, huber_beta=0.1
) -> torch.Tensor:
    """外边界 Huber"""

def neumann_crack_loss(
    model, x_top, y_top, rid_top, x_bot, y_bot, rid_bot,
    dT_jump_value: float, eps: float = 1e-3, huber_beta: float = 0.05,
) -> torch.Tensor:
    """裂纹段法向跳跃 Huber"""

def smoothness_loss(model, x, y, region_id) -> torch.Tensor:
    """Sobolev Hessian Frobenius 范数"""
```

### 3.2 `LossWeights`

```python
@dataclass
class LossWeights:
    lambda_pde: float = 100.0              # 主导 PDE 收敛
    lambda_interface: float = 10.0
    lambda_bc: float = 1.0
    lambda_neumann_crack: float = 0.05     # Neumann 跳跃值大，调小避免主导
    lambda_smooth: float = 0.0             # 默认关闭
```

### 3.3 `LossAggregator`

```python
class LossAggregator:
    def __init__(
        self,
        weights: LossWeights | None = None,
        outlier_cfg: OutlierConfig | None = None,  # v0.5 P2
        device: torch.device | str = "cpu",
    ) -> None: ...
    def __call__(
        self, model, batch: dict, current_epoch: int = 0
    ) -> tuple[torch.Tensor, dict]:
        """返回 (total, components_dict)；components_dict 字段:
        'pde'/'iface'/'tnormal'/'bc'/'neumann'/'smooth'
        + 'bc_active_frac'/'bc_n_outliers' (P2 启用时)
        + 'total' (均为 float)"""
```

### 3.4 `outlier`（P2，v0.5 新增）

```python
@dataclass
class OutlierConfig:
    enabled: bool = True
    burnin_epochs: int = 100      # Γ 宽限期
    delta: float = 3.0            # δ Z-score 阈值
    ema_alpha: float = 0.1        # 残差平方 EMA 平滑系数
    min_active_per_edge: int = 5  # 每条边最少保留点数
    n_edges: int = 4

class BoundaryOutlierTracker:
    def __init__(self, cfg, device="cpu", dtype=torch.float64) -> None: ...
    def update(self, residuals, x, y, edge_id, epoch) -> torch.Tensor:
        """返回 active_mask (N,) bool；Z-score 测试（MAD 稳健估计）"""
    def get_active_count_per_edge(self) -> Dict[int, int]: ...
    def state_dict(self) -> dict: ...
    def load_state_dict(self, state) -> None: ...
```

**Z-score 算法**（论文 §3.3 Eq.19-20）：
- 每点维护残差平方 `(T_pred - T_target)²` 的 EMA（α 平滑）
- burn-in（Γ=100）后，每 edge 内用中位数 + MAD（median absolute deviation）估算 μ/σ
- `Z_i = |EMA_i - μ_edge| / (σ_edge + ε)`，`Z_i > δ=3.0` 标记 outlier
- L_bc 除以 `N_total / N_active` 归一化（论文 Table 5：88% 损失下降）

---

## 4. 工具模块 (`utils`)

### 4.1 `DomainSpec`

```python
@dataclass(frozen=True)
class DomainSpec:
    x_min: float = -1.0
    x_max: float = 1.0
    y_min: float = -1.0
    y_max: float = 1.0
    crack_x_min: float = -0.5
    crack_x_max: float = 0.5
    crack_y_loc: float = 0.0
    crack_offset: float = 1e-3  # 缝合镜像偏移
```

### 4.2 区域 / 采样 / 归一化

```python
def region_id(x, y) -> torch.Tensor:
    """论文 §2.3 象限映射 → {0=A, 1=B, 2=C, 3=D}"""

def sample_interior(n_per_region, spec=DEFAULT_DOMAIN, device="cpu",
                    dtype=torch.float64, seed=None) -> (x, y, region_id):
    """拉丁超立方采样 4 区域"""

def sample_boundary(n_per_edge, spec=DEFAULT_DOMAIN, device="cpu",
                    dtype=torch.float64, seed=None) -> (x, y, edge_id):
    """4 条外边均匀采样"""

def sample_interface(n_per_seam, spec=DEFAULT_DOMAIN, device="cpu",
                     dtype=torch.float64, seed=None) -> dict:
    """3 条缝合 seam，返回 {seam_name: (x_l, y_l, rid_l, x_r, y_r, rid_r)}"""

def sample_crack(n_per_side, spec=DEFAULT_DOMAIN, device="cpu",
                 dtype=torch.float64, seed=None) -> (x_t, y_t, rid_t, x_b, y_b, rid_b):
    """裂纹段上下两侧采样"""

def normalize_to_unit(v, v_min, v_max) -> torch.Tensor:
    """Min-max → [-1, 1]"""

def denormalize_from_unit(v_norm, v_min, v_max) -> torch.Tensor:
    """[-1, 1] → 物理值"""

def T_exact_torch(x, y, include_crack=True, eps=1e-4,
                  hot_xy=(-0.6, 0.5), cold_xy=(0.6, -0.5),
                  crack_x_max=0.5, crack_jump=2.0, crack_steepness=50.0) -> torch.Tensor:
    """解析场 torch 版本"""
```

---

## 5. 训练入口 (`train`)

### 5.1 命令行参数

```
--epochs          int   None      # 训练总 epoch（默认 5000；--resume 未传时沿用 checkpoint target）
--lr              float 1e-3      # 初始学习率
--device          str   cpu        # cpu/cuda
--data            str   data/synthetic_thermal.npz
--N_int           int   2500      # 每区域内部配点数
--N_bc            int   100       # 每条外边配点数
--N_iface         int   50        # 每条缝合边配点数
--N_crack         int   50        # 裂纹段每侧配点数
--ablation        str   full      # full/two/single
--out             str   checkpoints/jpinn.pt
--log             str   logs/train_history.csv
--seed            int   42
--print_every     int   500
--log_plain       flag            # 禁用 ANSI 颜色 + Tee 日志（CI/重定向场景）
--resume          str   None     # 续训 checkpoint 路径；恢复 model + optimizer + scheduler + RNG + outlier
--boundary_strategy str resample # 外边界采样：resample（每 epoch 重新采样）/ fixed（固定点，P2 用）
# P2 Z-score 边界去噪（v0.5）
--outlier_enabled  flag            # 启用 Z-score 边界去噪（论文 §3.3 Eq.19-20）
--outlier_burnin   int   100       # burn-in 宽限期 Γ
--outlier_delta    float 3.0       # Z-score 阈值 δ
--outlier_ema_alpha float 0.1      # 残差平方 EMA 平滑系数 α
--lambda_pde             float 100.0
--lambda_interface       float 10.0
--lambda_interface_normal float 1.0  # L_traction（缝合边法向连续）；v0.2 新增
--lambda_bc              float 1.0
--lambda_neumann_crack   float 0.05
--lambda_smooth          float 0.0
```

**续训行为**（v0.4 P4-core + v0.5 调整）：
- `--resume <ckpt>` 时，checkpoint 内的 `args["seed"]` 覆盖 CLI（保证可复现）
- `--epochs` 语义：显式传入时优先 CLI（允许续训延长目标）；未传（None）时沿用 checkpoint 的 `target_epochs`
- 从 `checkpoint["epoch"]` (completed_epoch) +1 开始训练
- 若 checkpoint 无 `optimizer/scheduler state`（旧版），跳过对应 load
- P2 outlier tracker 状态随 checkpoint 保存/恢复（`outlier_state_dict`）
- 兼容旧版格式：若 `completed_epoch >= target_epochs` 视为旧版（存的是 target），自动修正

### 5.2 Checkpoint 格式

```python
torch.save({
    # 模型
    "model_state_dict": state,
    # 训练进度（v0.4：区分 completed 与 target）
    "epoch": completed_epoch,        # v0.4 改为"已完成 epoch 数"
    "target_epochs": args.epochs,    # v0.4 新增：目标 epoch 数（用于 scheduler 对齐）
    # 元数据
    "ablation": "full",
    "loss_weights": {...},
    "n_params": 70148,
    "best_loss": 0.226,
    "ds_meta": {"T_min", "T_max", "spec"},
    "args": {...},                    # 完整 argparse vars()（续训时覆盖 CLI）
    # v0.4 新增：续训所需 state
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": scheduler.state_dict(),
    "rng_state": torch.get_rng_state(),
    "numpy_rng_state": np.random.get_state(),
}, "checkpoints/jpinn.pt")
```

---

## 6. 可视化 (`visualize`)

### 6.1 命令行参数

```
--checkpoint      str   单 ckpt 模式
--data            str   data/synthetic_thermal.npz
--out_dir         str   logs/figures
--compare         list  多 ckpt 消融对比模式
--compare_labels  list  标签
```

### 6.2 输出

| 文件 | 内容 |
|---|---|
| `pred_vs_true_heatmap.png` | 三联：预测 / 真值 / \|相对误差\| |
| `loss_curves.png` | 6 子图：total + 5 类损失分量 |
| `per_region_2x2.png` | 4 区域独立预测 |
| `ablation_compare.png` | 多 ckpt E₂ 柱状图 |

---

## 7. 测试 (`tests.smoke_test`)

`python tests/smoke_test.py` — 8 步端到端冒烟（30s），输出 `✓ ALL SMOKE TESTS PASSED`。