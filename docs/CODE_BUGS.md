# CODE_BUGS — 已知 bug 清单

> J-PINN 项目已知 bug + 修复状态
> 风格仿 [v4 CODE_BUGS.md](../../projects/pe_mmnet/project_v4/docs/CODE_BUGS.md)

本文件**入版本控制**，作为团队"已知未修 bug + 历史 bug 库"的可视化清单。每次开发循环都应更新。

---

## 已知未修 bug

### Bug N-5：干跑污染 Adam optimizer 状态（优先级：低，来源 v0.4 P4-core 审计）

**症状**：干跑验证后 optimizer 的 Adam 动量 m/v 已"污染"（基于初始样本更新）
**根因**：干跑后 `optimizer.step()` 没回滚（v4 用 `dry_run_state` 快照恢复）
**影响**：正式训练从 step 1 开始时 Adam 已有虚假动量；可能影响收敛稳定性
**修复方案**：v4:1538-1682 模式——`optimizer_state = {k: v.clone() for k, v in optimizer.state_dict().items()}`后回滚
**决策**：低优先级，影响有限；v0.5+ 再修

### Bug N-6：Step 8 rng_state 覆盖加强（优先级：低，来源 v0.4 P4-core 审计）

**症状**：Step 8 checkpoint roundtrip 只验证 key 存在，未验证值有效性
**根因**：smoke test 偷懒
**影响**：极小（Step 9 已覆盖）
**修复方案**：Step 8 加 `assert torch.equal(saved_rng, current_rng)` 等值断言
**决策**：低优先级

### Bug N-1：消融对比不公平（优先级：中）

**症状**：Full 用 λ_pde=1 训练，Single/Two 用 λ_pde=100 训练，E₂ 数字不严格可比
**根因**：完整训练顺序：先训 full（旧权重），后改 λ_pde=100 才跑 single/two；full 的 checkpoint 没重训
**影响**：消融对比图（`ablation_compare.png`）当前 Single=20.9%, Two=30.1%, Full=22.4% 无说服力
**修复方案**：`python train.py --ablation full --epochs 5000 --out checkpoints/jpinn.pt`（~64 min），重训后再跑对比
**决策**：建议修复（论文 §4.6 关键证据），但非阻塞——已有 best_loss 数据可定性证明 J-PINN 价值

### Bug N-2：Neumann 跳跃 dT_jump 仅近似（优先级：低）

**症状**：解析场 `2·tanh(50y)` 在 y=±1e-3 处 `∂T/∂y ≈ 50`，但精确值 `2·50·sech²(50·1e-3) ≈ 99.95`
**根因**：中心差分近似 `(T_top - T_bot)/(2ε)` 在 ε=1e-3 时不精确
**影响**：Neumann 损失不能 100% 拟合，残差 ~0.01
**修复方案**：用 torch.autograd 在 y_top/y_bot 处求 ∂T/∂y，跳过中心差分
**决策**：低优先级，当前精度已满足定性验证

### Bug N-3：JPINN.region_id 路由 O(N) 布尔掩码慢（优先级：低）

**症状**：每个 MLP 都做完整 forward，但仅掩码部分点参与计算；理论上 4 个 MLP 可并行
**根因**：当前实现 `out[mask] = mlp(xy[mask]).squeeze(-1)`，未优化
**影响**：训练时长 64 min；若优化可缩短至 ~45 min（理论）
**修复方案**：把 batch 按 region_id 预排序分块，每块送对应 MLP；可完全并行
**决策**：低优先级，64 min 可接受

### Bug N-4：train.py 打印 print_every 硬编码（优先级：极低）

**症状**：默认 print_every=500，但用户需在 --print_every 自定义
**根因**：argparse 默认值固定
**影响**：打印密度不灵活
**修复方案**：自动根据 epochs 调整（如 epochs<1000 时 100/epochs 打印）
**决策**：低优先级

---

## 已修复 bug 历史

### Bug F-1：调和性自检偏差大 [✅ 修复于 commit 3dfa53e]

**症状**：max|∇²T| = 9.9e+03（远超 1e-6 阈值）
**根因**：首版用中心差分 `h=1e-3`，log 调和源附近梯度爆炸
**修复**：改用 `torch.autograd.grad` 二次求导；并排除源 0.05 半径内
**验证**：max|∇²T| < 1e-6 ✓（非源近端）

### Bug F-2：参数量 35,844 ≠ 论文 71,712 [✅ 修复于 commit b0f5b11]

**症状**：4 区域 JPINN 实际参数 ~36K，仅论文一半
**根因**：首版 `n_layers=4` 误实现为"1 输入 + 2 中间 + 1 输出 = 4 Linear"
**修复**：改为 `n_hidden_layers=4`（1 输入 + 4 隐藏 + 1 输出 = 6 Linear）
**验证**：参数量 70,148 ≈ 论文 71,712 ✓

### Bug F-3：denormalize NameError [✅ 修复于未提交 commit]

**症状**：`NameError: name 'v' is not defined`
**根因**：`utils.py:69` 函数体误写 `v` 而非 `v_norm`
**修复**：改为 `0.5 * (v_norm + 1.0) * (v_max - v_min) + v_min`
**验证**：可视化正常运行 ✓

### Bug F-4：PDE 残差不收敛 [✅ 修复于未提交 commit]

**症状**：5000 epoch 后 L_pde 停在 1.7e-3（验证清单要求 < 1e-4）
**根因**：`λ_pde=1.0` 不够主导（论文对 Navier-Cauchy 用 1e-6 是因量级不同）
**修复**：`λ_pde` 改 100，Neumann 项调小（λ=0.05，dT_jump=50）
**验证**：L_pde 主导训练 ✓

### Bug F-5：消融对比换行符 `\` 导致空路径 [✅ 修复于未提交 commit]

**症状**：`FileNotFoundError: [Errno 2] No such file or directory: '\\'`
**根因**：Windows 命令行 `\n` 续行被 argparse 解析为空字符串 `\\`
**修复**：`visualize.py` 过滤空字符串 token
**验证**：ablation_compare.png 正常生成 ✓

### Bug F-6：loss_curves log scale warning [✅ 修复于未提交 commit]

**症状**：`UserWarning: Data has no positive values, and therefore cannot be log-scaled.`
**根因**：当 L_smooth=0 长期 = 0 时，log scale 不允许 ≤0
**修复**：自动检测，全 0/负值时改 `symlog`
**验证**：无 warning ✓

---

## 模板（新 bug 追加用）

```
### Bug N-X：[短描述]（优先级：高/中/低/极低）

**症状**：[用户可见的现象]
**根因**：[代码引用 + 触发逻辑]
**影响**：[对训练/可视化/精度的具体影响]
**修复方案**：[具体改动 + 涉及文件]
**决策**：[是否阻塞 + 建议]
```