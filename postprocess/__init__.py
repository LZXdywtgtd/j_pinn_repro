"""postprocess 模块：J-integral / SERR 后处理（论文 §4.4 / §5.3 头条）

抽象映射（论文弹性 → 本项目 Laplace）：
  位移 u_j     →  温度 T
  应变 ε_ij    →  ∇T（梯度向量）
  应力 σ_ij    →  σ_ij = (∂T/∂x_i)(∂T/∂x_j)（梯度积，对称 rank-2 张量）
  应变能 W     →  W = ½ |∇T|²
  J-integral    →  J = ∮_Γ (W·n − σ·∇T) ds

对 ∇²T=0（调和场），∇·σ=0 自动成立，J 路径无关（Rice 1968）。
"""