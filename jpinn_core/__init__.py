"""jpinn_core —— J-PINN 内部核心包

v0.8 收纳自根目录的 7 个内部核心模块：
  - losses          损失函数 + 聚合器
  - outlier         P2 Z-score 边界去噪
  - schedulers      P17 LossProportionalLR
  - logging_utils   D3 TensorBoard/W&B writer
  - utils           区域 / 采样 / 归一化基础
  - utils_console   彩色控制台输出
  - utils_tee_eta   Tee 日志 + ETAEstimator

外部使用者应通过 `from jpinn_core.X import Y` 访问。
"""
