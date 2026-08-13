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