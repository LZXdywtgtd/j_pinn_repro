# 更新日志 (CHANGELOG)

所有重大更改将记录在此文件中。

格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)。

---

## [Unreleased]

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
- 冒烟测试需在本地 Python ≥ 3.10 环境运行；当前环境无 python.exe 可用，未执行