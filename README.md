# J-PINN Reproduction (2D Thermal Field)

复现论文《J-PINN: Adaptive domain-decomposed physics-informed neural network…》(Materials & Design 2026, j.matdes.2026.116567) 的核心架构与算法思想。

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
- **优化器**：Adam + CosineAnnealingLR（沿用 v4 `run_train.py:1494-1500` 写法）
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

### 2. 生成合成数据 + 训练 + 可视化
```bash
# 1) 生成合成数据（200x200 网格，保存为 data/synthetic_thermal.npz）
python data/generate_synthetic_thermal_data.py

# 2) 训练（5000 epoch，CPU，约 8 分钟）
python train.py --epochs 5000

# 3) 可视化（生成 logs/figures/ 三张图）
python visualize.py --checkpoint checkpoints/jpinn.pt

# 4) 消融：单区域 vs 双区域 vs 全 4 区域
python train.py --ablation single --epochs 5000 --out checkpoints/single.pt
python train.py --ablation two    --epochs 5000 --out checkpoints/two.pt
python visualize.py --compare checkpoints/single.pt checkpoints/two.pt checkpoints/jpinn.pt \
                     --compare_labels Single Two FourRegions
```

## 文件结构

```
j_pinn_repro/
├── README.md
├── requirements.txt
├── CHANGELOG.md
├── train.py                    # 训练入口
├── losses.py                   # PDE/Interface/BC/Neumann/Smoothness 5 类损失
├── utils.py                    # 区域掩码 + 拉丁超立方采样 + 归一化
├── visualize.py                # 加载 checkpoint 出图
├── data/
│   ├── generate_synthetic_thermal_data.py
│   ├── comsol_png_loader.py    # 预留接口（STUB）
│   ├── synthetic_thermal.npz   # 生成产物（被 .gitignore）
│   └── dataset.py              # ThermalDataset 统一入口
├── models/
│   └── pinn_core.py            # MLPBlock + JPINN（核心架构）
├── tests/
│   └── smoke_test.py           # 端到端冒烟测试（8 步）
├── checkpoints/                # 训练产物（被 .gitignore）
├── logs/                       # 训练历史 + 图像（被 .gitignore）
└── .github/workflows/
    └── feishu-notify.yml       # 推送通知（沿用 v4）
```

## 复现 v4 风格

本项目提交风格、`.gitignore` 结构、workflow 文件格式均参考 `projects/pe_mmnet/project_v4`。
具体规范见 `CHANGELOG.md` 与 `.gitignore` 头部注释。

## 引用

```bibtex
@article{stamatelatos2026jpinn,
  title={J-PINN: Adaptive domain-decomposed physics-informed neural network for fracture assessment},
  author={Stamatelatos, K. and Kotsinis, G. and Binsfeld, L. and Loutas, T.},
  journal={Materials \& Design},
  year={2026},
  doi={10.1016/j.matdes.2026.116567}
}
```