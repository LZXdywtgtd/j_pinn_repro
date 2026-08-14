"""
J-PINN 友好化日志 + ETA（仿 v4 run_train.py:106-136 Tee + 886-951 ETAEstimator）

两个组件：
1. Tee：同时输出到控制台（带 ANSI 颜色）和日志文件（去除颜色码）
2. ETAEstimator：指数移动平均估算剩余时间

设计：合并到一个文件 `utils_tee_eta.py`，避免子目录膨胀（j_pinn_repro/
  当前是单 utils.py，无 utils/ 子包）。
"""
from __future__ import annotations

import atexit
import re
import statistics
import sys
import time
from datetime import datetime, timedelta


# ============================================================
# Tee（控制台 + 文件双输出）
# ============================================================
class Tee:
    """同时输出到控制台和日志文件；颜色码自动去除后写入文件

    用法：
        tee = Tee("logs/train_20260813_143022.log")
        sys.stdout = tee
        sys.stderr = tee
        atexit.register(tee.close)
    """

    def __init__(self, filename: str) -> None:
        self.file = open(filename, "w", encoding="utf-8")
        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr
        self._ansi = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]")
        self._closed = False

    def write(self, text: str) -> None:
        if self._closed:
            self._orig_stdout.write(text)
            return
        self._orig_stdout.write(text)
        plain = self._ansi.sub("", text)
        self.file.write(plain)
        self.file.flush()

    def flush(self) -> None:
        if self._closed:
            self._orig_stdout.flush()
            return
        self._orig_stdout.flush()
        self.file.flush()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            sys.stdout = self._orig_stdout
            sys.stderr = self._orig_stderr
        except Exception:
            pass
        try:
            self.file.close()
        except Exception:
            pass


# ============================================================
# ETAEstimator（指数移动平均）
# ============================================================
class ETAEstimator:
    """指数移动平均 ETA 估算器（论文 v4 同款）

    用法：
        eta = ETAEstimator(total_epochs=5000, alpha=0.3)
        for epoch in range(1, 5001):
            eta.start_epoch()
            # ... 训练一个 epoch ...
            eta.end_epoch()
            info = eta.get_eta(epoch)
            print(f"ETA: {info['eta_str']} ± {info['confidence']}")
    """

    def __init__(self, total: int, alpha: float = 0.3) -> None:
        self.total = total
        self.alpha = alpha  # 近期权重
        self.epoch_times: list[float] = []
        self.ema: float | None = None
        self._epoch_start: float | None = None

    def start_epoch(self) -> None:
        self._epoch_start = time.time()

    def end_epoch(self) -> float:
        if self._epoch_start is None:
            return 0.0
        elapsed = time.time() - self._epoch_start
        self._epoch_start = None
        self.update(elapsed)
        return elapsed

    def update(self, epoch_time: float) -> None:
        self.epoch_times.append(epoch_time)
        if self.ema is None:
            self.ema = epoch_time
        else:
            self.ema = self.alpha * epoch_time + (1.0 - self.alpha) * self.ema

    def get_eta(self, current: int) -> dict:
        """返回 dict 含 ema/eta_seconds/eta_str/finish_time/confidence/remaining_epochs"""
        if self.ema is None:
            return {
                "ema": 0.0, "eta_seconds": 0.0, "eta_str": "?",
                "finish_time": "?", "confidence": "",
                "remaining_epochs": max(self.total - current, 0),
            }
        remaining = max(self.total - current, 0)
        eta_seconds = self.ema * remaining
        finish = datetime.now() + timedelta(seconds=eta_seconds)
        # 置信区间（最近 10 个 epoch std）
        if len(self.epoch_times) >= 5:
            recent = self.epoch_times[-min(10, len(self.epoch_times)):]
            std = statistics.stdev(recent) if len(recent) > 1 else 0.0
            confidence = f"±{std:.1f}s"
        else:
            confidence = ""
        # 时间格式
        if eta_seconds >= 3600:
            eta_str = f"{eta_seconds / 3600:.1f}h"
        elif eta_seconds >= 60:
            eta_str = f"{int(eta_seconds // 60)}m{int(eta_seconds % 60)}s"
        else:
            eta_str = f"{eta_seconds:.0f}s"
        return {
            "ema": float(self.ema),
            "eta_seconds": float(eta_seconds),
            "eta_str": eta_str,
            "finish_time": finish.strftime("%H:%M"),
            "confidence": confidence,
            "remaining_epochs": remaining,
        }


# =============================================================================
# P7 训练时间估算器（仿 v4 run_train.py:954-1063）
# =============================================================================
def estimate_training_time(
    model, ds, agg, optimizer, device, args, scheduler=None,
    n_warmup_epochs: int = 2,
    overhead_factor: float = 1.2,
    early_stop_factor: float = 0.85,
) -> tuple:
    """
    跑 n_warmup_epochs 个小 batch epoch 估算单 epoch 耗时与总训练时长。

    快照并恢复 model/optimizer/scheduler 状态（估算不应污染正式训练）。

    Args:
        model, ds, agg, optimizer, device, args: 与主循环相同的对象
        scheduler: 可选，有则快照/恢复
        n_warmup_epochs: 预跑 epoch 数
        overhead_factor / early_stop_factor: v4 同款系数

    Returns:
        (avg_time_s, estimated_total_minutes)
    """
    import copy
    import time as _time

    # 快照状态
    model_state = {k: v.clone() for k, v in model.state_dict().items()}
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    scheduler_state = copy.deepcopy(scheduler.state_dict()) if scheduler is not None else None

    try:
        times = []
        for _ in range(n_warmup_epochs):
            t0 = _time.time()
            batch = ds.get_collocation_batch(
                n_int_per_region=128,
                n_bc_per_edge=16,
                n_iface_per_seam=8,
                n_crack_per_side=8,
                seed=args.seed,
            )
            optimizer.zero_grad()
            total, _ = agg(model, batch, current_epoch=0)
            total.backward()
            import torch
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            times.append(_time.time() - t0)

        avg_time = times[-1] if times else 0.0
        estimated_total_s = avg_time * args.epochs * overhead_factor * early_stop_factor
        return avg_time, estimated_total_s / 60.0
    finally:
        # 恢复状态（估算不污染正式训练）
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)
        if scheduler is not None and scheduler_state is not None:
            scheduler.load_state_dict(scheduler_state)