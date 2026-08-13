# 更新日志 (CHANGELOG)

所有重大更改将记录在此文件中。

格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)。

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