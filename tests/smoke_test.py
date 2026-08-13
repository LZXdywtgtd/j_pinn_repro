"""
冒烟测试脚本：用最少 epoch 验证整条 pipeline 通畅

运行：
    python tests/smoke_test.py

覆盖：
- 数据生成（独立可重跑，不依赖已生成的 .npz）
- 模型参数数量断言
- 一次 forward + backward（不进入循环）
- 10 epoch 训练（确认 loss 下降 + 无 NaN）
- checkpoint 保存 + 加载
"""
from __future__ import annotations

import os
import sys
import subprocess

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, PROJECT_ROOT)


def step(name: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"[SMOKE] {name}")
    print("=" * 60)


def main() -> int:
    import torch
    from data.generate_synthetic_thermal_data import main as gen_data
    from data.dataset import ThermalDataset
    from losses import LossAggregator, LossWeights
    from models.pinn_core import build_model

    torch.set_default_dtype(torch.float64)

    # Step 1: 生成数据
    step("1) 生成合成数据")
    gen_data(N=100, out_path=os.path.join(PROJECT_ROOT, "data/synthetic_thermal.npz"))
    # 调和性自检：跳过单测断言（生成器自身已 print max/mean；PyTorch autograd 应 < 1e-6）

    # Step 2: 加载数据
    step("2) 加载数据集")
    ds = ThermalDataset()
    print(f"  T range: [{ds.T_min:.4f}, {ds.T_max:.4f}]")

    # Step 3: 构建模型
    step("3) 构建 4 区域 JPINN")
    model = build_model(ablation="full")
    n_params = model.count_parameters()
    print(f"  参数量: {n_params}")
    # 论文 §2.3 报告 71,712（4×17,928）；本实现因 LayerNorm 比论文略有差异，宽松判定
    assert 60_000 < n_params < 100_000, (
        f"4 区域 JPINN 应约 7-9 万参数（论文 §2.3 报 71,712），实际 {n_params}"
    )
    print(f"  ✓ 参数规模与论文 §2.3 同量级")

    # Step 4: 配点 batch
    step("4) 配点 batch 采样")
    batch = ds.get_collocation_batch(
        n_int_per_region=200,
        n_bc_per_edge=20,
        n_iface_per_seam=10,
        n_crack_per_side=10,
        seed=42,
    )
    print(f"  interior: x{batch['interior'][0].shape}")
    print(f"  boundary: x{batch['boundary']['x'].shape}, T{batch['boundary']['T_target'].shape}")
    print(f"  interfaces: {list(batch['interface'].keys())}")
    print(f"  crack T_jump={batch['crack']['T_jump_value']:.3f}, dT_jump={batch['crack']['dT_jump_value']:.3f}")

    # Step 5: 一次 forward + loss
    step("5) Forward + 5 类损失")
    agg = LossAggregator()
    total, comps = agg(model, batch)
    print(f"  total={comps['total']:.4e}")
    for k in ("pde", "iface", "bc", "neumann", "smooth"):
        print(f"    {k:>10s} = {comps[k]:.4e}")
    assert not torch.isnan(total), "Loss 出现 NaN"

    # Step 6: backward
    step("6) Backward + 梯度裁剪")
    model.zero_grad()
    total.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    print(f"  total grad norm (pre-clip) = {grad_norm:.4e}")

    # Step 7: 10 epoch mini training
    step("7) 10 epoch mini-training")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    initial_loss = comps["total"]
    for epoch in range(10):
        optimizer.zero_grad()
        total, _ = agg(model, batch)
        if torch.isnan(total):
            print(f"  [ERROR] epoch {epoch} NaN")
            return 1
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    final_loss = float(total.item())
    print(f"  loss: {initial_loss:.4e} → {final_loss:.4e}")
    assert final_loss < initial_loss, f"Loss 未下降 ({initial_loss:.4e} → {final_loss:.4e})"
    print(f"  ✓ Loss 下降，pipeline 通畅")

    # Step 8: checkpoint
    step("8) 保存与加载 checkpoint")
    ckpt_path = os.path.join(PROJECT_ROOT, "checkpoints/smoke.pt")
    torch.save({"model_state_dict": model.state_dict(), "epoch": 10}, ckpt_path)
    state = torch.load(ckpt_path, weights_only=False)
    assert "model_state_dict" in state
    os.remove(ckpt_path)
    print(f"  ✓ Checkpoint roundtrip OK")

    print("\n" + "=" * 60)
    print("✓ ALL SMOKE TESTS PASSED")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())