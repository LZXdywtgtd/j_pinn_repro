# 更新日志 (CHANGELOG)

所有重大更改将记录在此文件中。

格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)。

---

## [v0.7] - 2026-08-14

### Changed
- `--bc_loss_type` 默认 `huber` → `mse`（B1：贴合论文 Eq.18 合成场景）
- `--scheduler` 默认 `cosine` → `loss_prop`（B11：贴合论文 §3.4 损失比例）

### Added
- `docs/DECISIONS/0001-laplace-substitute.md`（ADR-0001）— 维度简化决策完整溯源
- ADR 模板（§12）— 后续决策按此规范记录

### Changed
- ADR-0001 补全§9 当时背景 + §10 局限性 + §11 冲突检测与重新决策机制（用户提醒）

### Changed（Stage 2 - B14）
- `losses.py:244-289` LossWeights docstring 扩展：论文 Table 2 vs 本项目映射（含 λ_pde=1e-6 vs 100.0 量级偏差 + 反转条件）
- `docs/experiment_reports/调参与算法工程指导文档.md` 新增 §8「论文 Table 2 权重映射」5 小节（论文值/本项目值/量级偏差原因/调参区间/反转条件）

### Changed（Stage 3 - B9 + C3）
- `losses.py:186-249` `neumann_crack_loss` 重写（对齐论文 Eq.17 traction continuity）
  - **旧逻辑**：有限差分 (T_top - T_bot)/(2*eps) ≈ dT_jump_value（强制 tanh 跳跃 = 数学替代）
  - **新逻辑**：autograd 求 ∂T/∂y_top 与 ∂T/∂y_bot，约束差接近 0（∂T/∂y 跨裂纹连续 = 物理约束）
  - **签名兼容**：`dT_jump_value` / `eps` 参数保留但不再使用（下游 CLI 兼容）
  - **测试**：常数场 → loss ≈ 0；线性场（slope=3）两侧 → loss ≈ 0
  - **影响**：训练行为改变——之前学 tanh 跳跃，现在学梯度连续；旧 checkpoint 续训不受影响（loss function 替换独立）

### Added（Stage 4 - B5 + C2）
- `postprocess/j_integral.py:158-274` 新增 `j_integral_mode_decomposed` 函数（Rigby-Aliabadi 对称/反对称分解）
  - **论文** §2.2 Eq.9-10：σ_sym / σ_asym 构造 → J_I / J_II 分解
  - **算法**：T(x,y) + T(x,-y) → T_sym/T_asym → σ_sym/σ_asym → J_I/J_II 积分
  - **⚠️ 物理意义限制**：热场降维后标量场无 Mode 概念，J_I/J_II 仅为形式分解
  - **保留用途**：验证分解算法正确性 + 准备反转 ADR-0001 时复用
  - **测试**：线性场 T=a*x+0.5*y → J_I=1.0, J_II=0.17；JPINN 随机初始化 → J_I+J_II=5.50
  - **import 兼容**：try `from utils` / except `from ..utils`（v0.3 路径兼容）
- `docs/dev_reference/api.md §8` 新增后处理模块章节
  - §8.1 j_integral_mode_decomposed（含 ⚠️⚠️⚠️ 警告 + ADR-0001 反转条件）
  - §8.2 stress_analog σ_ij = ∇T·∇T^T 文档化（B6）
  - §8.3 CLI 入口示例
- `postprocess/j_integral.py` 文件顶部 docstring 加 ⚠️⚠️⚠️ 物理意义限制段

### Fixed（Stage 5 - B2 + B7 + B8）
- `losses.py:402-409` B2：移除 `L_b = L_b_raw * (n_total / n_active)` 冗余
  - 旧逻辑：mask 后 re-evaluate 再放大 → 等价于未 mask 的 L_b_raw，违反论文 Eq.19 `1/|A| Σ`
  - 新逻辑：mask 后直接用 `L_b_raw`（|A| 已隐式归一化）
  - 影响：L_bc 数值约减小 N_total/N_active 倍（≈ 1.2x）；与论文 Eq.19 公式对齐
- `postprocess/far_field_anchoring.py` B7：新增 `select_far_field_anchor(x_lig, x_wake, J, mode)`
  - `mode="extremes"`（默认）：`min(x_lig) + max(x_wake)`（原行为）
  - `mode="residual_min"`：拟合线性平面 → 取 `|J_raw - J_fit|` 最小 contour（论文 §4.4 「误差最小处」）
- `postprocess/run_j_integral.py` 接入 B7 + 新增 `--anchor_mode {extremes, residual_min}` CLI flag
- `losses.py:365-369` B8：边界 region_id 改用 `utils.region_id` 统一（替代 Python 短路布尔加法）
  - 旧逻辑：4 个互斥 `(x<0)&(y>=0)` 短路 + long 加法（角点 (0,0) 路由到 1，与内部配点不一致）
  - 新逻辑：复用 `utils.region_id`（与 `sample_interior` / `sample_interface` / `sample_crack` 一致）
  - 行为差异：内部配点采样 + 边界点现在都用同一 routing 函数，避免未来 corner 处理不一致

### Changed（Stage 5）
- `docs/dev_reference/api.md` 同步：§3.3 LossAggregator 注释 B2/B8 + §8.1 select_far_field_anchor + §8.3 --anchor_mode CLI 表

---

## [v0.6] - 2026-08-14

### Added
- **D1-fix 干跑状态回滚**：`train.py` 干跑无条件回滚 model/optimizer/scheduler（清 CODE_BUGS N-5）
- **P7 时间估算 + ETA 修复**：`estimate_training_time`（仿 v4:954-1063）+ 修复 ETA 死代码
- **P17 损失比例 LR**：`schedulers.LossProportionalLR`（论文 §3.4）+ `--scheduler/--lr_min`
- **P12 BC loss 切换**：`bc_loss_dirichlet(loss_type)` + `--bc_loss_type {mse,huber}`
- **P6 架构 sweep**：`build_model(hidden, n_hidden_layers)` + 20 架构配方 + `collect_pareto.py`
- **P9 多 restart 集成**：`run_ensemble.py`（多 seed + 参数平均）
- **D3 TB/W&B**：`logging_utils.make_writer/log_metrics` + `--logger`

### Fixed
- ETA 死代码（ETAEstimator 从未 update）
- `logging_utils` tensorboard 双前缀（`loss/loss/total`）
- `schedulers` 违反 `_LRScheduler` 基类契约
- `run_ensemble.average_models` float64 精度损失（`.float()` 降 float32）
- `collect_pareto` params 恒为 0（未读 checkpoint n_params）

### Changed
- `train.py` 新增 9 个 CLI flag
- `docs/dev_reference/api.md`：CLI 表 + 4.5/4.6/4.7 新模块 API

---

## [v0.5] - 2026-08-13

### Added
- **D1 COMSOL 多扫描 batch loader**：
  - `load_comsol_scan_dir_as_array`：返回 `(T_volume (N,H,W) float64, loaded_field_name, meta)`（用户 v0.5 决策）
  - `load_comsol_scan_dir` 保留完整 `T_grid_volume` 4D 序列（不再只返回首帧）
  - 字段动态检测：`field=None` 时按子目录数自动选择（0→报错，1→加载，多→字母序第一个 + [INFO]）
  - `field` 指定时严格查找，不存在抛 FileNotFoundError
  - `_natural_sort_key`：按文件名数字 token 排序（温度001 < 温度010）
  - `_auto_detect_subdir`：无硬编码优先级，基于实际目录结构
- **测试**：`tests/test_comsol_png_loader.py` 扩展 10 个 D1 测试（数字排序/4D volume/首帧一致性/field 检测/max_frames/as_array/空目录/缺字段/多目录自动选/单色抑制）
- **P2 Z-score 边界去噪**（论文 §3.3 Eq.19-20）：
  - `outlier.py`：OutlierConfig + BoundaryOutlierTracker（EMA + MAD 稳健 Z-score）
  - 用中位数 + MAD 而非 mean/std（测试暴露 mean/std 对稀疏 outlier 不敏感）
  - L_bc 除以 `N_total / N_active` 归一化（论文 Table 5：88% 损失下降）
  - `--boundary_strategy fixed` 支撑持久 mask
  - smoke_test Step 11 端到端验证（active_frac=0.838 检出 13 outlier）
  - `tests/test_outlier.py` 5 单元测试（EMA/spike/burn-in/min_active/normalization）

### Fixed
- **P11 遗留性能 bug**：`_rgb_to_colorbar_position` 构造 `(H*W, H_cb, 3)` 距离矩阵 → 2000×1500 图 OOM（单帧 35.6s）
  - 修复：色标下采样到 `min(n_levels, len(cb_rgb))`（默认 128）+ argmin 向量化
- **P11 遗留检测 bug**：`_detect_physical_region` 把色标条纳入物理域（right 扩到色标右缘）
  - 修复：用"最宽连续厚列块"（计数 > 0.5*height）区分物理域与窄色标带
- **v0.3 遗留 utils_console.py 孤立三引号**：line 22 多余 `"""` 导致 tokenizer 报全角括号 SyntaxError
  - 修复：删掉孤立 `"""`
- **v0.3 遗留 GBK 编码**：`generate_synthetic_thermal_data.py` 打印 `∇²T` Unicode 在 Windows GBK 控制台报 UnicodeEncodeError
  - 修复：改用 ASCII（Laplacian / [OK] / [WARN]）
- **v0.3 遗留 datetime 双调用**：`train.py` CSV 写入 `datetime_now().isoformat()` 而 `datetime_now()` 已返回字符串
  - 修复：改为 `datetime_now()`（去重 `.isoformat()`）
- **v0.4 续训 epochs 语义**：`--resume` 强制用 checkpoint 的 target_epochs，无法延长训练
  - 修复：`--epochs` 默认 None；显式传时优先 CLI，未传时沿用 checkpoint

### Known Limitations
- 训练结果（checkpoints/*.pt, logs/*.csv）不入仓
- 测试耗时 ~116s（imageio 写 2000×1500 PNG 慢；可减小合成 PNG 尺寸加速）
- `_detect_multiplier_from_ticks` 简化版返回 1.0 不 OCR（需用户传 multiplier 覆盖 ×10²）

---

## [v0.4] - 2026-08-13

### Added
- **P4-core Checkpoint Resume**：`--resume <ckpt>` 支持续训
  - 恢复 model + optimizer + scheduler + RNG state（torch + numpy）
  - checkpoint 新增 `target_epochs` 字段（区别于 `epoch: completed_epochs`）
  - 兼容旧版 checkpoint（无 optimizer/scheduler state）
  - smoke_test Step 10：subprocess 端到端续训验证（50 epoch → resume → 100 epoch）
- **P4-extension 友好化日志**：Tee（实时落盘无 ANSI）+ ETAEstimator（α=0.3 指数移动平均 + 完成时间）
  - train.py CSV 扩展到 14 列（新增 timestamp/task_id/ablation/epoch/eta_seconds/ema_s）
  - 控制台颜色：NaN 黄、错误红、最佳 loss 青高亮
- **P3 J-integral 后处理**：论文 §4.4 / §5.3 头条算法
  - 矩形轮廓积分 + 远场锚定误差补偿
  - 标量场抽象映射 + 解析 J 对比验证
- **P5 多 ablation 编排器**：`run_ablations.py` 一键跑 3 个 ablation
  - 拓扑排序 + `--only` / `--dry-run` / `--visualize-only`
  - 配方表 `tasks/ablation_recipes.json`
- **P11 COMSOL PNG 加载器**：替换 stub
  - 必需 `colorbar_range` 参数（避免忘了设定时偏差大）
  - 像素区域检测 + 色标 RGB 反推温度
- **P1 L_traction** 缝合边法向连续损失
- **`docs/全流程.md` 重写**：加"代码 → 审计 → 自动修复 → 再审计 → 写文档 → 再审计 → commit"闭环
- **`docs/audit/` 目录**：每次审计报告单独存档（`YYYY-MM-DD-*-审计报告.md`）

### Changed
- `docs/api.md`：`--resume` flag + checkpoint schema 同步更新
- 消融公平性：`num_networks` 属性（full=4, two=2, single=1）
- argparse `--lambda_pde` 默认值 1.0 → 100.0（修复默认值不一致 bug）
- argparse `--lambda_neumann_crack` 默认值 0.5 → 0.05

### Known Limitations
- 训练结果（checkpoints/*.pt, logs/*.csv）不入仓
- COMSOL 加载器 v0.4 实现单帧解析，多扫描 batch 处理留 v0.5

---

## [v0.3] - 2026-08-13

### Added
- 合成 2D 热力场数据生成器（解析 log 调和函数 + tanh 裂纹间断）
- 4 区域 J-PINN 模型（~71,712 参数，SiLU 激活）
- 5 类损失函数：PDE / Interface / Dirichlet BC / Neumann Crack / Sobolev Smoothness
- 训练脚本 `train.py`：Adam + CosineAnnealingLR + 干跑 + 梯度裁剪
- 可视化脚本 `visualize.py`：预测 vs 真值热图 + 损失曲线 + 4 区域子图
- 消融支持：`--ablation {single, two, full}`
- COMSOL PNG 加载器预留接口（`data/comsol_png_loader.py`，stub）
- 飞书推送通知 workflow（沿用 v4 `feishu-notify.yml`）
- 端到端冒烟测试 `tests/smoke_test.py`（8 步：生成数据→加载→模型→batch→forward→backward→10 epoch→checkpoint）

### Notes
- 本地未做推送（无远端 + 等待用户决策；详见会话记录）
- Python 环境：建议 conda 创建独立环境 `jpinn`，激活后再 `pip install -r requirements.txt`
- 冒烟测试需在本地 Python ≥ 3.10 环境运行