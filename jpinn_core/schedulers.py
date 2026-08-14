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
        self._current_loss: float | None = None
        # base_lr 来自 optimizer 初始 param_groups
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """按当前 loss_ref / last_epoch 计算各 param_group 的 lr。

        遵循 _LRScheduler 契约：step() 会先更新内部状态再调用本方法，
        返回的 list 会写入每个 param_group["lr"]。
        """
        if self.loss_ref is None:
            # 尚未有 loss_ref（未 step(loss)）→ 保持 base_lr
            return self.base_lrs
        # 比例，clamp 到 [lr_min/base_lr, 1.0]
        ratio = max(
            self.lr_min / max(self.base_lrs[0], 1e-12),
            min(self._current_loss / max(self.loss_ref, 1e-12), 1.0),
        )
        return [base_lr * ratio for base_lr in self.base_lrs]

    def step(self, loss=None, epoch=None):
        """loss 必传（比例 LR 需要）；epoch 可选（兼容 _LRScheduler 签名）。

        注意：_LRScheduler.__init__ 会调一次 step()（无参数）初始化 last_epoch，
        此时 loss=None 应 no-op（不抛错），真正训练时 loss 才必传。
        """
        if loss is None:
            # 基类 _initial_step() 调用：no-op（仅推进 last_epoch）
            self._current_loss = None
            if epoch is None:
                self.last_epoch += 1
            else:
                self.last_epoch = epoch
            self._last_lr = self.get_lr()
            return
        loss_val = float(loss) if isinstance(loss, torch.Tensor) else float(loss)
        self._current_loss = loss_val
        # 首次 step：设定 loss_ref
        if self.loss_ref is None:
            self.loss_ref = max(loss_val, 1e-12)
        # 更新 last_epoch（用显式 epoch 或自增）
        if epoch is None:
            self.last_epoch += 1
        else:
            self.last_epoch = epoch
        # 用 get_lr() 计算并写入（遵循基类契约）
        lrs = self.get_lr()
        for group, lr in zip(self.optimizer.param_groups, lrs):
            group["lr"] = lr
        self._last_lr = lrs

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