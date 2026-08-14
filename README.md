# J-PINN Reproduction (2D Thermal Field)

复现论文《J-PINN: A domain-decomposed physics-informed neural network framework for kinematic field reconstruction in fracture mechanics》(Materials & Design 2026, j.matdes.2026.116567) 的核心架构与算法思想。

## 论文 → 降维映射

| 论文术语 | 本实现替代 | 理由 |
|---|---|---|
| 2D 弹性力学 Navier-Cauchy PDE | **2D Laplace 方程 ∇²T = 0** | 缺少论文原版 DIC 数据；热传导保留"全场 PDE 残差监督"框架特性 |
| 4 区域 FNN（弹性体裂纹分解） | **4 区域 MLP（热力场象限分解）** | 完整保留 J-PINN 的核心架构贡献 |
| Interface stitching（位移/牵引力连续） | **温度连续 + 裂纹段 Neumann 跳跃** | 类比缝合损失 |
| L_smooth (Sobolev Hessian) | 同款正则 | 防谱偏置 |
| L_bc Huber + Z-score 离群剔除 | 同款 Huber | 对边界噪点鲁棒 |

## 关键决策

- **架构**：4 个独立 MLP，每个 4 隐藏层 × 64 神经元 + SiLU；总参数量 ≈ 71,712（与论文 §4.2 一致）
- **优化器**：Adam + LossProportionalLR（v0.7 B11 改默认；论文 §3.4 损失比例；可通过 `--scheduler cosine` 切回）
- **精度**：float64（论文注：float32 在裂纹尖端损失精度）
- **设备**：CPU 默认（论文明确 CPU ~10× faster than GPU for ~72K 参数规模）
- **数据**：合成 2D 热力场（log 调和函数 + tanh 裂纹间断）；COMSOL PNG 加载器留 stub

## 快速上手

### 0. 创建 conda 环境（推荐）
```bash
conda create -n jpinn python=3.10 -y
conda activate jpinn
pip install -r requirements.txt
```

### 1. 冒烟测试（验证 pipeline）
```bash
python tests/smoke_test.py
```
应打印 `✓ ALL SMOKE TESTS PASSED`。

### 2. 统一 CLI 入口（v0.8 推荐）

```bash
# 查看所有子命令
python jpinn.py --help

# 训练（v0.9：不带 --out 自动归档到 outputs/full/<task_id>/）
python jpinn.py train --epochs 5000

# 可视化（v0.9：不带 --checkpoint 自动解析 outputs/latest.json）
python jpinn.py visualize
# 显式路径仍兼容：
python jpinn.py visualize --checkpoint checkpoints/jpinn.pt

# J-integral 后处理（同 visualize：不带 --checkpoint 自动解析 outputs/latest.json）
python jpinn.py j_integral
# 显式路径仍兼容：
python jpinn.py j_integral --checkpoint checkpoints/jpinn.pt

# 生成合成数据
python jpinn.py generate_data

# 消融 / 集成 / Pareto（编排）
python jpinn.py ablations --dry-run
python jpinn.py ensemble --n_restarts 3
python jpinn.py pareto --sweep_dir logs/sweep
```

**兼容**：旧 CLI 入口仍可独立运行（`python train.py` / `python visualize.py` / `python -m postprocess.run_j_integral` 等）。

### 3. 完整 pipeline（生成 → 训练 → 可视化）
```bash
# 1) 生成合成数据（200x200 网格，保存为 data/synthetic_thermal.npz）
python jpinn.py generate_data

# 2) 训练（5000 epoch，CPU，约 8 分钟；v0.9 自动归档到 outputs/full/<task_id>/）
python jpinn.py train --epochs 5000

# 3) 可视化（v0.9 裸命令自动解析 outputs/latest.json，输出到 task 目录 figures/，
#    含 pred_vs_true_heatmap / loss_curves / training_growth_panel / per_region_2x2）
python jpinn.py visualize

# 4) 消融：单区域 vs 双区域 vs 全 4 区域
#    显式路径（旧方式，仍兼容）；路径已存在时需加 --force 才允许覆盖
python jpinn.py train --ablation single --epochs 5000 --out checkpoints/single.pt
python jpinn.py train --ablation two    --epochs 5000 --out checkpoints/two.pt
python jpinn.py visualize --compare checkpoints/single.pt checkpoints/two.pt checkpoints/jpinn.pt \
                         --compare_labels Single Two FourRegions
```

## 文件结构

```
j_pinn_repro/
├── README.md
├── requirements.txt
├── CHANGELOG.md
├── jpinn.py                     # 统一 CLI 入口（v0.8；7 子命令：train/visualize/j_integral/...）
├── train.py                     # 训练入口（仍兼容）
├── visualize.py                 # 可视化入口（仍兼容）
├── jpinn_core/                  # 内部核心包（v0.8 收纳）
│   ├── losses.py                # PDE/Interface/BC/Neumann/Smoothness 5 类损失
│   ├── outlier.py               # P2 Z-score 边界去噪
│   ├── schedulers.py            # P17 LossProportionalLR
│   ├── logging_utils.py         # D3 TB/W&B writer
│   ├── utils.py                 # 区域掩码 + 拉丁超立方采样 + 归一化
│   ├── utils_console.py         # 彩色控制台
│   └── utils_tee_eta.py         # Tee + ETAEstimator
├── data/
│   ├── generate_synthetic_thermal_data.py
│   ├── comsol_png_loader.py
│   ├── synthetic_thermal.npz    # 生成产物（被 .gitignore）
│   └── dataset.py               # ThermalDataset 统一入口
├── models/
│   └── pinn_core.py             # MLPBlock + JPINN（核心架构）
├── postprocess/                 # J-integral 后处理（已用包级相对导入）
│   ├── __init__.py
│   ├── analytic_J.py
│   ├── contour_sampling.py
│   ├── stress_from_T.py
│   ├── far_field_anchoring.py
│   ├── j_integral.py
│   └── run_j_integral.py
├── tests/
│   └── smoke_test.py            # 端到端冒烟测试（11 步）
├── tasks/                       # 配方 JSON（用户可编辑）
│   ├── ablation_recipes.json
│   └── architecture_sweep_recipes.json
├── checkpoints/                 # 显式路径训练产物（被 .gitignore）
├── logs/                        # 显式路径训练历史 + 图像（被 .gitignore）
├── outputs/                     # v0.9 结果管理系统：<ablation>/<task_id>/ 自动归档 + latest.json（被 .gitignore）
├── docs/                        # 文档（8 个子目录）
│   ├── audit/                   # 审计报告 + CODE_BUGS.md
│   ├── architecture/            # 架构设计文档
│   ├── DECISIONS/               # ADR 决策日志
│   ├── dev_reference/           # API + 开发人员 + 流程文档
│   ├── collaboration/           # 消融实验指南
│   ├── experiment_reports/      # 调参 + 实验设计
│   ├── postprocess/             # J_integral 设计
│   └── user_guides/             # 快速配置指南
└── .github/workflows/
    └── feishu-notify.yml        # 推送通知（沿用 v4）
```

## 复现 v4 风格

本项目提交风格、`.gitignore` 结构、workflow 文件格式均参考 `projects/pe_mmnet/project_v4`。
具体规范见 `CHANGELOG.md` 与 `.gitignore` 头部注释。

## 引用

```bibtex
@article{stamatelatos2026jpinn,
  title={J-PINN: A domain-decomposed physics-informed neural network framework for kinematic field reconstruction in fracture mechanics},
  author={Stamatelatos, K. and Kotsinis, G. and Binsfeld, L. and Loutas, T.},
  journal={Materials \& Design},
  year={2026},
  doi={10.1016/j.matdes.2026.116567}
}
```