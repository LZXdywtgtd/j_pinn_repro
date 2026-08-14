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

### 1.3 `data.comsol_png_loader`（v0.5 完整实现，v0.8 修复检测算法）

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

**物理域/色标检测算法**（v0.8 修复）：
- **物理域** = 最宽的连续非白**列块**（[left, right]）+ 该区间内最高的连续非白**行块**（[top, bottom]）
  - 旧算法用「每列非白密度 > 0.5×height」阈值——真实 COMSOL 图高温区渲染成近白色（inferno 顶端 [255,255,255]），物理域列密度仅 ~20%，反而色标条 ~57% 被误判为物理域（bug：读出全底色常数）
- **色标条** = 物理域右侧第一个连续非白列块，返回**完整 bbox 含水平位置**
  - 旧版 `_detect_colorbar` 只返回垂直范围 + 调用点假设「色标紧贴物理域右侧」，但真实图物理域与色标间有 ~58px 白间隙，采样带落在空白区
- `_largest_block(mask)`：返回最长连续 True 块（start, end 半开区间）

**字段自动检测**（v0.5 用户决策，无硬编码优先级）：
- `field=None`：扫描子目录数 0 → FileNotFoundError；1 个 → 加载该目录；多个 → 按字母序取第一个 + [INFO] 提示
- `field="温度"`：严格查找；不存在 → FileNotFoundError

**真实 COMSOL 数据实测**（2026-08-14，`D:/team_project/simulation/参考输入/参数化扫描1/`）：
- 温度 001：bbox (105,1440,135,1725) 宽1590，T ∈ [293.1, 1431.2] K
- d 001：d ∈ [0.0000, 0.9921]（归一化损伤）
- 应力 001：σ ∈ [0, 1e9] Pa

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
    """裂纹段 ∂T/∂y 连续性约束（论文 Eq.17 traction continuity）

    v0.7 语义变化（B9）：
    - 旧版：有限差分 (T_top - T_bot)/(2*eps) ≈ dT_jump_value（强制 tanh 跳跃）
    - 新版：autograd 求 ∂T/∂y_top 与 ∂T/∂y_bot，约束其差接近 0
    - dT_jump_value / eps 参数保留仅兼容 CLI 签名（不再使用）
    """

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
        bc_loss_type: str = "mse",     # v0.7 阶段 1：默认 mse（论文 Eq.18）
    ) -> None: ...
    def __call__(
        self, model, batch: dict, current_epoch: int = 0
    ) -> tuple[torch.Tensor, dict]:
        """返回 (total, components_dict)；components_dict 字段:
        'pde'/'iface'/'tnormal'/'bc'/'neumann'/'smooth'
        + 'bc_active_frac'/'bc_n_outliers' (P2 启用时)
        + 'total' (均为 float)

        v0.7 阶段 5 (B2)：L_bc 不再乘 n_total/n_active；mask 后直接 re-evaluate
        v0.7 阶段 5 (B8)：边界 region_id 用 utils.region_id 统一（替代 Python 短路）
        """
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
- v0.7 阶段 5 (B2)：L_bc 不再乘 n_total/n_active；mask 后直接重算（论文 Eq.19 `1/|A| Σ`）

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

## 4.5 调度器 (`schedulers`)（v0.6 P17 新增）

```python
class LossProportionalLR(torch.optim.lr_scheduler._LRScheduler):
    """lr = base_lr * clip(loss / loss_ref, lr_min/base_lr, 1.0)"""
    def __init__(self, optimizer, loss_ref=None, lr_min=1e-6, last_epoch=-1): ...
    def step(self, loss=None, epoch=None):
        """loss 必传；loss=None 时 no-op（基类 _initial_step 调用）"""
    def state_dict(self) -> dict: ...       # 含 loss_ref/lr_min/base_lrs
    def load_state_dict(self, state) -> None: ...
```

## 4.6 日志 writer (`logging_utils`)（v0.6 D3 新增）

```python
def make_writer(logger_type: str, log_dir: str):
    """none → None；tensorboard → SummaryWriter；wandb → wandb.run"""
def log_metrics(writer, epoch: int, comps: dict) -> None:
    """发射 loss 各分量（tensorboard 用 add_scalars；wandb 用 log）"""
```

## 4.7 集成工具（v0.6 P9 新增，v0.8 阶段 7 更新）

```python
# run_ensemble.py
def run_restarts(n_seeds, base_args, cwd, seed_base=0) -> list[Path]:
    """循环 seeds 跑 train.py 子进程，返回 checkpoint 路径"""
def average_models(ckpt_paths, out_path) -> dict:
    """torch.stack 参数平均（保持 float64 dtype）"""

# collect_pareto.py
def collect_pareto(sweep_dir, out_csv, n_params_map=None) -> None:
    """读 sweep CSV 输出 best_loss 表格 + Top5"""

# jpinn_core/utils_tee_eta.py（v0.8 阶段 7 更新）
def estimate_training_time(model, ds, agg, optimizer, device, args,
                           scheduler=None, n_warmup_epochs=2,
                           overhead_factor=1.2, early_stop_factor=0.85,
                           use_real_batch=True) -> tuple:
    """干跑估算单 epoch 耗时与总训练时长
    v0.8 阶段 7：新增 use_real_batch（默认 True）
      - True：用 args.N_int/N_bc/N_iface/N_crack 真实配点规模（旧版缩小 batch
        导致预估偏差 9.3×；修复后 1.2×）
      - False：旧行为（128/16/8/8 缩小 batch，保留兼容调试）
    """
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
--hidden          int   64        # 每子网隐藏层宽度（P6 sweep）
--n_hidden_layers int   4         # 每子网隐藏层数（P6 sweep）
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
# v0.6 新增
--bc_loss_type    str   mse      # 外边界损失：mse（论文 Eq.18 合成数据默认）/ huber（Eq.19 含噪 DIC）
--scheduler       str   loss_prop # LR 调度器：loss_prop（论文 §3.4 损失比例默认）/ cosine
--lr_min          float 1e-6     # LR 下限（loss_prop 用）
--logger          str   none     # 训练日志：none / tensorboard / wandb
# v0.7 阶段 5（B7）远场锚定（postprocess CLI，非 train.py）
# --anchor_mode extremes | residual_min  # 详见 postprocess/run_j_integral.py
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

---

## 8. 后处理模块 (`postprocess`)（v0.3 + v0.7 阶段 4）

### 8.1 J-integral 数值积分

```python
from postprocess.j_integral import (
    j_integral_one_contour,         # 单条 contour 的标量 J
    j_integral_surface,             # 批量：返回 (x_lig_grid, x_wake_grid, J_grid)
    j_integral_exact_for_surface,   # 解析 J（用于对比 PINN 输出）
    j_integral_mode_decomposed,     # ⚠️ v0.7 Mode I/II 分解（论文 §2.2 Eq.9-10）
)

from postprocess.far_field_anchoring import (
    fit_linear_drift,               # 最小二乘拟合 J = a + b_lig·x_lig + b_wake·x_wake
    select_far_field_anchor,         # v0.7 阶段 5 (B7)：选择远场锚点 contour
    compensate_j_surface,           # 远场锚定补偿（论文 §4.4 公式）
    path_independence_metric,       # 路径无关度 = std/mean
    relative_error,                 # 相对误差 = |J_pred - J_exact| / max(|J_exact|)
)

def select_far_field_anchor(
    x_lig: np.ndarray, x_wake: np.ndarray, J: np.ndarray,
    mode: str = "extremes",         # "extremes" | "residual_min"
) -> Tuple[float, float]:
    """B7: 选远场锚定 contour 参数
    - extremes (默认): min(x_lig) + max(x_wake)
    - residual_min: |J_raw - J_fit| 最小 contour（论文 §4.4）
    """
```

def j_integral_one_contour(
    model, contour: RectContour, T_min: float, T_max: float, spec=None,
) -> float:
    """J = ∮_Γ (W·n_x - σ·n·∂T/∂x) ds（沿 ligament x₁）"""

def j_integral_mode_decomposed(
    model, contour: RectContour, T_min: float, T_max: float, spec=None,
) -> tuple[float, float]:
    """Rigby-Aliabadi 对称/反对称分解 → (J_I, J_II)

    ┌──────────────────────────────────────────────────────────────────┐
    │ ⚠️⚠️⚠️ 热场降维下的物理意义限制（v0.7 B5+C2）⚠️⚠️⚠️                │
    │                                                                  │
    │ 论文 Mode I/II 描述弹性力学裂纹模式：                              │
    │   Mode I（张开型）：裂纹面法向被拉开（σ_22 拉伸）                  │
    │   Mode II（滑开型）：裂纹面切向相对滑动（σ_12 剪切）              │
    │                                                                  │
    │ 本项目 Laplace ∇²T=0 降维后：                                     │
    │   - 标量温度场没有"方向性断裂模式"概念                             │
    │   - σ_ij = ∇T·∇T^T 是几何外积（不是 Hooke 律应力）                │
    │   - 本函数 J_I / J_II 数值不应解读为 Mode 贡献                     │
    │                                                                  │
    │ 本函数保留用途：                                                   │
    │   1. 验证 Rigby-Aliabadi 分解算法的数学正确性（正交性、路径无关）  │
    │   2. 准备未来反转 ADR-0001 时直接复用（拿到 DIC 后）              │
    │                                                                  │
    │ 反转条件（ADR-0001 §11）：                                         │
    │   - 获得论文原版 DIC 全场位移数据                                  │
    │   - 切换 Navier-Cauchy PDE（恢复混合二阶导 + Lamé 参数）           │
    │   - 届时 J_I / J_II 数值将恢复物理意义                             │
    └──────────────────────────────────────────────────────────────────┘

    算法（Rigby-Aliabadi 对称/反对称分解）：
    1. 同时评估 T(x,y) 与 T(x,-y)（裂纹面对称）
       T_sym = 0.5 * (T(x,y) + T(x,-y))
       T_asym = 0.5 * (T(x,y) - T(x,-y))
    2. 对 T_sym 求梯度 → σ_sym = ∇T_sym · ∇T_sym^T
       对 T_asym 求梯度 → σ_asym = ∇T_asym · ∇T_asym^T
    3. J_I = ∮ (W_sym·n_x - σ_sym·n·∂T_asym/∂x) ds
       J_II = ∮ (W_asym·n_x - σ_asym·n·∂T_sym/∂x) ds
    4. 交叉项正交抵消（数学基础）：J ≈ J_I + J_II
    """
```

### 8.2 应力类比（v0.7 B6 文档化，v0.8 阶段 7 更新）

```python
from postprocess.stress_from_T import (
    grad_T, grad_T_physical, stress_analog, strain_energy_W,
)

def grad_T(model, x, y, region_id) -> tuple:
    """∂T/∂x, ∂T/∂y via autograd（在归一化空间）

    v0.8 阶段 7 修复：autograd.grad 加 retain_graph=True
    （旧版 False 会在第二次 backward 时报 "Trying to backward through
    the graph a second time"，因第一次 ∂T/∂x 已释放中间梯度）
    """

def stress_analog(dT_dx, dT_dy) -> tuple[σ_xx, σ_yy, σ_xy]:
    """σ_ij = ∇T·∇T^T（论文弹性 σ_ij = λ tr(ε) δ_ij + 2μ ε_ij 的降维类比）

    ⚠️ σ_ij = ∇T·∇T^T 是维度简化下的几何外积
       - 论文 Hooke 律 σ_ij = λ tr(ε) δ_ij + 2μ ε_ij（铝 λ=27070, μ=27481 MPa）
       - 本项目类比 σ_ij = (∂T/∂x_i)(∂T/∂x_j)（无 λ/μ，纯几何）
       - 需反转 ADR-0001 切换 Navier-Cauchy 才能恢复物理应力
    """
```

**v0.8 阶段 7 后处理链路修复**（`postprocess/j_integral.py`）：
- `_j_integral_pinn_one` 移除 `@torch.no_grad()` 装饰器——它与 `grad_T` 的
  `requires_grad_(True)` 互斥，导致 `RuntimeError: element 0 of tensors does not
  require grad`（装饰器在 grad 前包裹整段，让 autograd 图无法建立）
- `run_j_integral.py` 中 `J_pinn_grid` 扁平化 + `x_lig_flat`/`x_wake_flat`
  用 `np.tile`/`np.repeat` 扩展到与 J 等长——旧版 unique 值（5 个）与
  `J_pinn_grid (5,5)` 不匹配导致 `LinAlgError: Incompatible dimensions`

### 8.3 CLI 入口

```bash
python -m postprocess.run_j_integral \
    --checkpoint checkpoints/jpinn.pt \
    --data data/synthetic_thermal.npz \
    --out_dir logs/j_integral \
    --n_per_side 200 \
    --anchor_mode extremes         # v0.7 阶段 5 (B7)；residual_min 可选
```

**远场锚定模式**（B7）：
- `extremes`（默认）：取 `min(x_lig)` + `max(x_wake)` contour（原行为）
- `residual_min`：取 `|J_raw - J_fit|` 最小 contour（论文 §4.4「误差最小处」）