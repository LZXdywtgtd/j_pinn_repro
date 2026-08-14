# J-integral 后处理模块设计（论文 §4.4 / §5.3）

> 版本 0.3 | 更新：2026-08-13

---

## 一、背景

J-integral（应变能释放率 G）是 J-PINN 论文的**头条应用案例**（§4.4 + §5.3）。它用矩形 contour 围住裂纹尖端，积分能量释放率，证明 PINN 训练后的模型**可工程化用于断裂评估**。

本项目 v0.1/v0.2 完全没有这个模块。P3 目标：在合成 Laplace 场上**抽象复现**论文 J-integral 算法（视 T 为位移 u，∇T 为应变），并实现远场锚定误差补偿。

---

## 二、数学映射（弹性 → Laplace 降维）

| 弹性（J-PINN 论文）| Laplace（本项目降维）|
|---|---|
| 位移 u_j | 温度 T |
| 应变 ε_ij = ½(∂u_i/∂x_j + ∂u_j/∂x_i) | ∇T（梯度向量）|
| 应力 σ_ij = λ tr(ε) δ_ij + 2μ ε_ij | σ_ij = (∂T/∂x_i)(∂T/∂x_j)（梯度积，对称 rank-2 张量）|
| 应变能密度 W = ½ σ_ij ε_ij | W = ½ |∇T|² |
| J-integral | J = ∮_Γ (W·n₁ − σ_ij·n_j·∂u_i/∂x₁) ds |

**关键性质**：对任意 ∇·σ = 0 的张量场，J 路径无关（Rice 1968）。本项目的 σ_ij = (∂T/∂x_i)(∂T/∂x_j) 在 ∇²T=0 时 ∇·σ=0 自动成立。

---

## 三、算法流程（论文 §4.4 + 附录 A）

### 3.1 单条 contour 的 J 积分

```
对每条矩形 contour（参数 x_lig, x_wake, y_min, y_max）：
  1. 在 4 段（左/上/右/下）均匀采样 N 个点 + 外法向
  2. region_id = region_id(x, y)
  3. T = JPINN.forward(x_norm, y_norm, region_id)
  4. autograd: ∂T/∂x_norm, ∂T/∂y_norm
  5. 链式法则: ∂T/∂x_phys = ∂T/∂x_norm × chain_T × chain_x
  6. σ_xx = (∂T/∂x)², σ_yy = (∂T/∂y)², σ_xy = (∂T/∂x)(∂T/∂y)
  7. W = ½(σ_xx + σ_yy)
  8. integrand = W·n_x − (σ_xx·n_x + σ_xy·n_y)·∂T/∂x
  9. J = ∮ integrand ds = torch.trapz(integrand, ds_index)
```

### 3.2 路径无关性 + 远场锚定补偿

```
对 (N_lig × N_wake) contour 网格：
  1. J_raw(x_lig, x_wake) → 5×5 = 25 个值
  2. 拟合双线性平面: J_raw ≈ a + b_lig·x_lig + b_wake·x_wake
     （论文 §4.4 Eq. A.24，残留体力导致线性漂移）
  3. 远场锚定: 在最远的 contour (x_lig_far, x_wake_far)：
     J_far = a + b_lig·x_lig_far + b_wake·x_wake_far
  4. 补偿: J_corrected = J_raw − b_lig·(x_lig − x_lig_far)
                            − b_wake·(x_wake − x_wake_far) + J_far
  5. 验证: std(J_corrected)/|mean(J_corrected)| < 5%（论文 §4.4 准则）
```

---

## 四、文件结构

```
j_pinn_repro/
├── postprocess/
│   ├── __init__.py               # 5 行：说明模块用途
│   ├── contour_sampling.py       # 80 行：RectContour + sweep_contours + contour_to_tensor
│   ├── stress_from_T.py          # 60 行：grad_T + stress_analog + strain_energy_W
│   ├── j_integral.py             # 130 行：_j_integral_pinn_one + j_integral_surface
│   ├── far_field_anchoring.py    # 80 行：fit_linear_drift + compensate_j_surface
│   ├── analytic_J.py             # 100 行：j_integral_exact + j_integral_exact_surface
│   └── run_j_integral.py         # 180 行：CLI + 可视化
├── tests/
│   └── test_j_integral.py        # 200 行：5 个单元测试
└── docs/postprocess/J_integral_design.md  # 本文件
```

**总计 ~835 行**（不含测试 200 行）。所有代码**仅新增**，不修改现有 train.py / losses.py / pinn_core.py / utils.py / visualize.py / dataset.py。

---

## 五、复用现有模块

| 复用项 | 来源 |
|---|---|
| autograd 模式（detach + requires_grad + grad）| [jpinn_core/losses.py:47-69](../../jpinn_core/losses.py) |
| region_id 推断 | [jpinn_core/utils.py:region_id](../../jpinn_core/utils.py) |
| 模型加载 + checkpoint 解析 | [visualize.py:load_model](../visualize.py) |
| ThermalDataset（T_min/T_max/spec）| [data/dataset.py:ThermalDataset](../data/dataset.py) |
| 4 区域 JPINN（4 子网络 + 路由）| [models/pinn_core.py:JPINN](../models/pinn_core.py) |

---

## 六、验证计划

### 6.1 单元测试（`tests/test_j_integral.py`）

1. **test_contour_closed**：RectContour.sample() 首尾点重合 + 段间连续
2. **test_stress_analog_consistency**：对 T = log(r)，验证 σ_xx = (∂T/∂x)² 与解析导数一致
3. **test_j_path_independence_exact**：解析场 J 路径无关（std/mean < 1e-4）
4. **test_j_exact_matches_trapezoidal**：梯形积分 O(h²) 收敛性
5. **test_far_field_anchoring_corrects_drift**：注入线性漂移 → 补偿 → 曲面变平

### 6.2 端到端验证（`python -m postprocess.run_j_integral`）

v0.9 起 `--checkpoint` 默认 None：裸命令自动解析 `outputs/latest.json` 中 full 的最新任务，输出到其 `j_integral/` 子目录；latest 无记录则报错（需先训练或显式传 `--checkpoint`）。

```bash
# v0.9 默认方式
python -m postprocess.run_j_integral \
    --n_per_side 200 \
    --x_lig_values -0.9 -0.7 -0.5 -0.3 -0.1 \
    --x_wake_values 0.1 0.3 0.5 0.7 0.9

# 显式路径（旧方式，仍兼容）
python -m postprocess.run_j_integral \
    --checkpoint checkpoints/jpinn.pt \
    --data data/synthetic_thermal.npz \
    --out_dir logs/j_integral \
    --n_per_side 200 \
    --x_lig_values -0.9 -0.7 -0.5 -0.3 -0.1 \
    --x_wake_values 0.1 0.3 0.5 0.7 0.9
```

**期望输出**（默认输出到 checkpoint 同目录 `j_integral/`；显式 `--out_dir` 时输出到指定目录）：
- `j_integral/J_raw.png`：PINN 原始 J 曲面（可能含线性漂移）
- `j_integral/J_corrected.png`：补偿后 J 曲面（应平坦，std/mean < 5%）
- `j_integral/J_exact.png`：解析场 J 曲面（参考）
- `j_integral/relative_error.png`：PINN vs 解析 误差
- `j_integral/metrics.json`：路径无关度 + 相对误差

**质量门控**（CLI 返回码）：
- rc=0：path_indep_corrected < 10% 且 relative_error_corrected < 50%
- rc=1：路径无关性偏差
- rc=2：相对误差过大

---

## 七、设计决策与权衡

| 决策 | 选择 | 理由 |
|---|---|---|
| J-integral 物理单位 | K²/m（无量纲对比）| 论文 N/mm；跨 PDE 类型无强约束 |
| 裂纹尖端位置 | 固定 (0, 0) | 论文 §4.4 默认；与本项目裂纹段中心一致 |
| 路径无关性图维度 | 3D surface | 论文 Fig.12 风格；直观展示误差补偿 |
| Checkpoint 依赖 | 必须存在（v0.9：`--checkpoint` 默认从 outputs/latest.json 解析；无则报错） | 随机初始模型精度差，验证质量没意义 |
| 积分方法 | 梯形法 | 论文未指定；rectangle 是分段线性，梯形足够 |
| 远场 contour 选择 | (x_lig_min, x_wake_max) | 论文 §4.4 推荐用域边界附近 |

---

## 九、参考

- Rice, J. R. (1968). "A path-independent integral and the approximate analysis of strain concentration by notches and cracks." *Journal of Applied Mechanics*.
- 论文 §4.4 + 附录 A
- 论文 §5.3（实验验证 DCB 案例 G_I = 0.0466 N/mm vs MBT 解析 0.048 N/mm）