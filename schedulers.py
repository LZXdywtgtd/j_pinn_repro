"""
P17 损失比例 LR 调度器（论文 §3.4）

论文 §3.4："learning rate annealing, decaying proportionally to the total loss,
was found critical for maintaining stability."

实现：
- lr = base_lr * (loss / loss_ref)，clamp 到 [lr_min, base_lr]
- loss_ref 在第一次 step(loss) 时若为 None 则设为当前 loss（每进程独立，sweep 自然对齐）
- 每次 step 需显式传入当前 total loss

用法（train.py）：
    scheduler = LossProportionalLR(optimizer, loss_ref=None, lr_min=1e-6)
    for epoch in ...:
        total, comps = agg(model, batch)
        ...
        scheduler.step(comps["total"])
"""
from __future__ import annotations

import torch
from torch.optim.lr_scheduler import _LRScheduler


class LossProportionalLR(_LRScheduler):
    """
    lr = base_lr * clip(loss / loss_ref, lr_min/base_lr, 1.0)

    - loss_ref 首次 step(loss) 时自动设为当前 loss（如未显式传入）
    - loss 下降 → lr 单调下降；loss 上升 → lr 最多到 base_lr（cap）
    - lr_min 兜底（默认 1e-6）

    Args:
        optimizer: 已创建的 optimizer
        loss_ref: 参考 loss（None = 首次 step 时设为当前 loss）
        lr_min: 学习率下限
        last_epoch: 继承 _LRScheduler（-1 = 未开始）
    """

    def __init__(
        self,
        optimizer,
        loss_ref: float | None = None,
        lr_min: float = 1e-6,
        last_epoch: int = -1,
    ) -> None:
        self.loss_ref = loss_ref
        self.lr_min = lr_min
        # base_lr 来自 optimizer 初始 param_groups
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """由 step() 设置 lr 后此方法不再被调用；保留兼容返回当前 lr"""
        return [group["lr"] for group in self.optimizer.param_groups]

    def step(self, loss=None, epoch=None):
        """loss 必传（比例 LR 需要）；epoch 可选（兼容 _LRScheduler 签名）。

        注意：_LRScheduler.__init__ 会调一次 step()（无参数）初始化 last_epoch，
        此时 loss=None 应 no-op（不抛错），真正训练时 loss 才必传。
        """
        if loss is None:
            # 基类 _initial_step() 调用：no-op（仅推进 last_epoch）
            if epoch is None:
                self.last_epoch += 1
            else:
                self.last_epoch = epoch
            self._last_lr = [group["lr"] for group in self.optimizer.param_groups]
            return
        loss_val = float(loss) if isinstance(loss, torch.Tensor) else float(loss)
        # 首次 step：设定 loss_ref
        if self.loss_ref is None:
            self.loss_ref = max(loss_val, 1e-12)
        # 比例，clamp 到 [lr_min/base_lr, 1.0]
        ratio = max(
            self.lr_min / max(self.base_lrs[0], 1e-12),
            min(loss_val / max(self.loss_ref, 1e-12), 1.0),
        )
        for i, group in enumerate(self.optimizer.param_groups):
            base_lr = self.base_lrs[i] if i < len(self.base_lrs) else self.base_lrs[0]
            group["lr"] = base_lr * ratio

        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]
        # 更新 last_epoch（用显式 epoch 或自增）
        if epoch is None:
            self.last_epoch += 1
        else:
            self.last_epoch = epoch

    def state_dict(self):
        """扩展 state_dict 保存 loss_ref / lr_min（续训用）"""
        state = super().state_dict()
        state["loss_ref"] = self.loss_ref
        state["lr_min"] = self.lr_min
        state["base_lrs"] = self.base_lrs
        return state

    def load_state_dict(self, state_dict):
        """恢复 state_dict（含 loss_ref / lr_min）"""
        if "loss_ref" in state_dict:
            self.loss_ref = state_dict.pop("loss_ref")
        if "lr_min" in state_dict:
            self.lr_min = state_dict.pop("lr_min")
        if "base_lrs" in state_dict:
            self.base_lrs = state_dict.pop("base_lrs")
        super().load_state_dict(state_dict)