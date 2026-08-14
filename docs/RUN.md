# J-PINN 完整运行清单

> **本清单的"重写"承诺**：每个命令的每个参数都对照源码 `p.add_argument` 逐项映射，不再凭印象。

---

## 0. 强化审计流程（v0.8 阶段 7+）

### 0.1 审计盲区教训

v0.8 阶段 7 暴露的 `run_j_integral` 三连 bug（`atexit` / `requires_grad` / 维度不对齐）**全部在两次历史审计（v0.6 / v0.7）中漏掉**。根因不是流程设计错，是流程**执行不到位**：

| 错误做法 | 正确做法 |
|---|---|
| 写文档时凭印象描述函数行为 | 写文档时**重新打开被描述的源文件**，逐行核对数据流 |
| "测试通过"= smoke_test 11 步通过 | smoke_test 仅覆盖主训练路径；CLI 入口需独立端到端测试 |
| 4 维度核对只检查"自洽" | 必须检查"未覆盖路径"——文档里显式列出来 |

### 0.2 强化审计 5 条铁律

**铁律 1：写文档必须重读源码**
- 不准凭印象。每写一句"X 函数调用 Y 函数"必须 grep 验证。
- CLI 命令清单必须逐项对源码 `p.add_argument`。

**铁律 2：每条 CLI 都要有端到端测试**
- 用户可调用的 CLI 入口必须有 `tests/test_*_cli.py` 跑 subprocess。
- 测试矩阵覆盖：默认参数 + 关键 flag 组合 + 边界值。

**铁律 3：审计记录必须显式标注"未覆盖路径"**
- 每份审计报告加一栏"未覆盖路径"，未来维护者一眼能看到测试盲区。

**铁律 4：错误处理不再"凭印象假设"**
- 每次修复必须附最小复现命令 + 实际报错截图/堆栈。
- 不准说"应该没问题"——必须跑过。

**铁律 5：跨 shell 命令必须三环境实测**
- 文档命令必须能在 Git Bash + PowerShell + cmd 三环境跑通。
- 验证用 `chcp 65001` 切到 UTF-8，**禁止** `PYTHONIOENCODING=utf-8` 前缀（Git Bash 会当作命令名）。

---

## 1. 强化审计追溯：本清单的"文档→源码"映射表

| 文档声明 | 源码位置 | 验证命令 |
|---|---|---|
| 训练入口支持 `--epochs` | [train.py:59](train.py:59) | `grep 'p.add_argument("--epochs"' train.py` |
| 训练默认 `--ablation full` | [train.py:72](train.py:72) | `grep -A2 '"--ablation"' train.py` |
| 数据生成入口在 `data/generate_synthetic_thermal_data.py` | [generate_synthetic_thermal_data.py:151](data/generate_synthetic_thermal_data.py:151) | `python data/generate_synthetic_thermal_data.py` |
| J 积分后处理入口 | [postprocess/run_j_integral.py:213](postprocess/run_j_integral.py:213) | `python -m postprocess.run_j_integral --help` |
| J 积分支持 `--anchor_mode` | [postprocess/run_j_integral.py:226](postprocess/run_j_integral.py:226) | 同上 |
| 可视化支持 `--compare` | [visualize.py:52](visualize.py:52) | `python visualize.py --help` |
| JPINN 统一 CLI 入口 | [jpinn.py:61](jpinn.py:61) | `python jpinn.py --help` |
| 消融编排器 | [run_ablations.py:124](run_ablations.py:124) | `python run_ablations.py --help` |
| 多 restart 集成 | [run_ensemble.py:82](run_ensemble.py:82) | `python run_ensemble.py --help` |

**未覆盖路径**（v0.8 阶段 8 待补）：
- `run_j_integral` 端到端 CLI 测试（`tests/test_run_j_integral_cli.py`）— **本次 bug 直接触发**
- `visualize --compare` 多 ckpt 对比模式
- `run_ablations` 编排器
- `run_ensemble` 集成
- `collect_pareto` Pareto 前沿收集

---

## 2. 前置：环境（首次）

### 验证：所有命令依赖 jpinn conda 环境

源码依赖（`train.py:35-43` 等）：
```python
from data.dataset import ThermalDataset
from jpinn_core.losses import LossAggregator, LossWeights
from models.pinn_core import build_model
```

这些 import 在 `jpinn` 环境里。**未激活 jpinn = import 失败**。

### Git Bash (Windows / macOS / Linux)

```bash
conda activate jpinn
cd /d/team_project/j_pinn_repro
```

### PowerShell (Windows)

```powershell
conda activate jpinn
chcp 65001
cd D:\team_project\j_pinn_repro
```

### cmd (Windows)

```
conda activate jpinn
chcp 65001
cd D:\team_project\j_pinn_repro
```

---

## 3. A. 验证 pipeline（必跑，~1 分钟）

**目的**：用 `tests/smoke_test.py` 的 11 步确认整条主训练路径通畅。

**Git Bash / PowerShell / cmd 通用**（实测命令）：
```bash
python tests/smoke_test.py
```

期望输出最后一行：`✓ ALL SMOKE TESTS PASSED`（11 步）。

**未覆盖的测试**（v0.8 阶段 8 待补）：

| 模块 | 测试文件 | 状态 |
|---|---|---|
| train.py 主路径 | `tests/smoke_test.py` | ✅ 11/11 |
| `postprocess/j_integral` 单元 | 无 | ❌ 缺失 |
| `run_j_integral` CLI 端到端 | 无 | ❌ 缺失 |
| `visualize` CLI | 无 | ❌ 缺失 |
| `data.generate_synthetic_thermal_data` | `smoke_test.py:Step 1` 覆盖 | ✅ |
| `comsol_png_loader` 单元 | `test_comsol_png_loader.py` | ✅ 18 个 |

---

## 4. B. 训练合成数据 + 完整产出

**目的**：从合成数据生成到 J 积分后处理，验证全链路。

**实测耗时**（基于你跑完的 71.54 分钟实测）：
- 数据生成：~30 秒
- 训练 5000 epoch：~71.5 分钟（4 区域，CPU）
- 可视化：~10 秒
- J 积分后处理：~30 秒

### 4.1 单行命令（Git Bash / PowerShell / cmd 三环境通用）

```bash
python data/generate_synthetic_thermal_data.py
python train.py --epochs 5000 --ablation full
python visualize.py --checkpoint checkpoints/jpinn.pt
python -m postprocess.run_j_integral --checkpoint checkpoints/jpinn.pt --anchor_mode extremes
```

**禁用反斜杠续行**（cmd 不识别 `\`）。如要传多个 flag，**全写在一行**，用空格分隔。

### 4.2 关键 flag 映射表

| 命令 | 关键 flag | 默认值 | 源码 |
|---|---|---|---|
| `train.py` | `--epochs` | None（兜底 5000） | [train.py:59](train.py:59) |
| `train.py` | `--ablation` | `full` | [train.py:72](train.py:72) |
| `train.py` | `--out` | `checkpoints/jpinn.pt` | [train.py:78](train.py:78) |
| `train.py` | `--log` | `logs/train_history.csv` | [train.py:79](train.py:79) |
| `visualize.py` | `--checkpoint` | `checkpoints/jpinn.pt` | [visualize.py:49](visualize.py:49) |
| `visualize.py` | `--out_dir` | `logs/figures` | [visualize.py:51](visualize.py:51) |
| `run_j_integral` | `--checkpoint` | `checkpoints/jpinn.pt` | [run_j_integral.py:215](postprocess/run_j_integral.py:215) |
| `run_j_integral` | `--out_dir` | `logs/j_integral` | [run_j_integral.py:217](postprocess/run_j_integral.py:217) |
| `run_j_integral` | `--n_per_side` | 200 | [run_j_integral.py:218](postprocess/run_j_integral.py:218) |
| `run_j_integral` | `--anchor_mode` | `extremes` | [run_j_integral.py:226](postprocess/run_j_integral.py:226) |

### 4.3 产出清单

| 文件 | 来源 |
|---|---|
| `data/synthetic_thermal.npz` | `generate_synthetic_thermal_data.main()` [L151](data/generate_synthetic_thermal_data.py:151) |
| `checkpoints/jpinn.pt` | `train.py` save |
| `logs/train_history.csv` | Tee 日志 |
| `logs/figures/{pred_vs_true_heatmap,loss_curves,per_region_2x2}.png` | visualize.py 单 ckpt 模式 |
| `logs/j_integral/{J_raw,J_corrected,J_exact,relative_error}.png` + `metrics.json` | run_j_integral.py |

---

## 5. C. 消融对比（可选，~3.5 小时）

**目的**：复现论文 §4.6 消融研究（Table 4）。

Git Bash（推荐后台跑）：
```bash
python train.py --ablation single --epochs 5000 --out checkpoints/single.pt
python train.py --ablation two --epochs 5000 --out checkpoints/two.pt
python train.py --ablation full --epochs 5000 --out checkpoints/jpinn.pt
python visualize.py --compare checkpoints/single.pt checkpoints/two.pt checkpoints/jpinn.pt --compare_labels Single Two FourRegions
```

cmd 等价（路径用 `\` 单独写每条命令）：
```
python train.py --ablation single --epochs 5000 --out checkpoints\single.pt
python train.py --ablation two --epochs 5000 --out checkpoints\two.pt
python train.py --ablation full --epochs 5000 --out checkpoints\jpinn.pt
python visualize.py --compare checkpoints\single.pt checkpoints\two.pt checkpoints\jpinn.pt --compare_labels Single Two FourRegions
```

---

## 6. D. COMSOL 真实数据（验证修复后的 loader）

**实测命令**（基于 v0.8 阶段 6 修复后的真实 COMSOL 数据）：

### Git Bash
```bash
python -c "import sys; sys.path.insert(0,'.'); from data.comsol_png_loader import load_comsol_png; T, meta = load_comsol_png('D:/team_project/simulation/参考输入/参数化扫描1/温度/温度001.png', colorbar_range=(293.15, 1431.15)); print('T:', T.shape, '范围', T.min(), T.max(), 'K')"
```

### cmd（路径用 `\\`）
```
python -c "import sys; sys.path.insert(0,'.'); from data.comsol_png_loader import load_comsol_png; T, meta = load_comsol_png('D:\\team_project\\simulation\\参考输入\\参数化扫描1\\温度\\温度001.png', colorbar_range=(293.15, 1431.15)); print('T:', T.shape, '范围', T.min(), T.max(), 'K')"
```

期望输出：`T: (1335, 1590) 范围 293.1 ~ 1431.2 K`（修复后真实数据正确读出）。

---

## 7. E. 重新编译论文中文译本

**条件**：仅当你改了 `docs/论文中文译本/sections/*.tex` 时。

```bash
cd docs/论文中文译本
xelatex -interaction=nonstopmode main.tex
xelatex -interaction=nonstopmode main.tex
```

**两遍原因**：第二遍解析交叉引用（章节/图/表编号）。

---

## 8. 常见报错及对应指令（按"实测复现→修复"组织）

| 报错 | 实测复现命令 | 根因 | 修复 |
|---|---|---|---|
| `python: 找不到命令` | Git Bash 输入 `python` | Windows 把 python.exe 别名指到 Microsoft Store | 用 `C:/Users/LZXdywtgtd/.conda/envs/jpinn/python.exe` 完整路径 |
| `PYTHONIOENCODING 不是内部或外部命令` | `PYTHONIOENCODING=utf-8 python train.py` | Git Bash 把 `VAR=value` 当作命令名 | 用 `chcp 65001`（cmd）或直接不加前缀 |
| 中文乱码 | cmd 直接 `python train.py` | cmd 默认 GBK | 先 `chcp 65001` |
| 反斜杠续行被截断 | `python train.py --epochs 5000 \ --ablation full`（cmd） | cmd 不识别 `\` 续行 | 命令单行写 |
| `ModuleNotFoundError: jpinn_core` | `python train.py` | 未激活 conda 环境 | `conda activate jpinn` |
| `NameError: name 'atexit' is not defined` | `python train.py --epochs 5000` | `train.py` 漏 `import atexit` | ✅ 已修（v0.8 阶段 7） |
| `RuntimeError: element 0 of tensors does not require grad` | `python -m postprocess.run_j_integral --checkpoint checkpoints/jpinn.pt` | `_j_integral_pinn_one` 用 `@torch.no_grad()` 与 `grad_T` 互斥 | ✅ 已修（v0.8 阶段 7，移除装饰器 + `retain_graph=True`） |
| `LinAlgError: Incompatible dimensions` | `python -m postprocess.run_j_integral --anchor_mode extremes` | `x_lig_arr` 5 长与 `J_pinn_grid` (5,5) 不匹配 | ✅ 已修（v0.8 阶段 7，扁平化 + tile/repeat） |
| 训练发散（NaN） | `python train.py --epochs 5000` | λ_pde 太大 | `--lambda_pde 50` |
| PDE 残差不收敛 | 训练后看 logs/train_history.csv pde 列 | λ_pde 太小 | `--lambda_pde 200` |
| 显存/内存溢出 | 大 batch | N_int 太大 | `--N_int 1000` |

---

## 9. 最关键的 3 条命令（任何 shell）

### Git Bash
```bash
conda activate jpinn && cd /d/team_project/j_pinn_repro && python tests/smoke_test.py
```

### PowerShell
```powershell
conda activate jpinn; cd D:\team_project\j_pinn_repro; python tests/smoke_test.py
```

### cmd
```
conda activate jpinn && chcp 65001 && cd D:\team_project\j_pinn_repro && python tests\smoke_test.py
```

期望最后一行：`✓ ALL SMOKE TESTS PASSED`。

---

## 10. 完整最短路径（从零到 J 积分后处理产出）

**总耗时：~75 分钟**（1 分钟 smoke + 30 秒数据 + 71.5 分钟训练 + 10 秒可视化 + 30 秒 J 积分）

### Git Bash
```bash
conda activate jpinn
cd /d/team_project/j_pinn_repro
python tests/smoke_test.py
python data/generate_synthetic_thermal_data.py
python train.py --epochs 5000 --ablation full
python visualize.py --checkpoint checkpoints/jpinn.pt
python -m postprocess.run_j_integral --checkpoint checkpoints/jpinn.pt --anchor_mode extremes
```

### cmd
```
conda activate jpinn
chcp 65001
cd D:\team_project\j_pinn_repro
python tests\smoke_test.py
python data\generate_synthetic_thermal_data.py
python train.py --epochs 5000 --ablation full
python visualize.py --checkpoint checkpoints/jpinn.pt
python -m postprocess.run_j_integral --checkpoint checkpoints/jpinn.pt --anchor_mode extremes
```

---

## 11. v0.8 阶段 7 修复总结 + 历史漏洞教训

### 11.1 本次修复内容

| Bug | 触发条件 | 修复 |
|---|---|---|
| `train.py: NameError: atexit` | 直接 `python train.py`（smoke_test 用 subprocess 跳过此路径） | `train.py:20` 加 `import atexit` |
| `j_integral: requires_grad 错` | `python -m postprocess.run_j_integral` | `stress_from_T.py` `grad_T` 移除 `@torch.no_grad()`（装饰器间接问题）+ 加 `retain_graph=True` |
| `j_integral: Incompatible dimensions` | `python -m postprocess.run_j_integral --anchor_mode extremes` | `run_j_integral.py` `J_pinn_flat = J_pinn_grid.ravel()` + `x_lig_flat = np.tile+repeat` 扩展到 25 长 |
| ETA 估算偏差 9.3× | 任意 `--epochs` | `jpinn_core/utils_tee_eta.py` 干跑默认 `use_real_batch=True`（旧行为保留为 `False`） |

### 11.2 历史漏洞教训（关键）

**两次审计（v0.6 / v0.7）都漏掉了以上 4 个 bug**，因为：

1. **smoke_test 不覆盖 CLI 入口端到端路径**——只测 `train.py` 的 subprocess 调用，跳过了 `atexit` 注册时机
2. **`run_j_integral` 完全没有自动化测试**——所有用户首次跑这个 CLI 才会发现 bug
3. **审计自检时"凭印象"写文档，没回去逐行 grep 核对源码**——本应是审计的护栏，反而成了漏洞的伪装

### 11.3 v0.8 阶段 8 待补

| 任务 | 优先级 |
|---|---|
| `tests/test_run_j_integral_cli.py`（端到端 CLI 烟测，覆盖 4 个 bug 的回归场景） | **高** |
| `tests/test_visualize_cli.py`（`--compare` 模式） | 中 |
| `tests/test_run_ensemble_cli.py` | 中 |
| `tests/test_run_ablations_cli.py` | 中 |
| `tests/test_collect_pareto_cli.py` | 低 |
| 把 0.2 节的 5 条铁律正式写进 `docs/dev_reference/审计流程.md` | 中 |