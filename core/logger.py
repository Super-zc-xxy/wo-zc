import logging
import queue
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

try:
    import winsound
except ImportError:
    winsound = None


class QueueLogHandler(logging.Handler):
    """线程安全的日志队列处理器，供桌面 GUI 实时显示日志。"""

    def __init__(self, log_queue: "queue.Queue[tuple[str, str]]", max_items: int = 2000) -> None:
        super().__init__()
        self.log_queue = log_queue
        self.max_items = max_items

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            if self.log_queue.qsize() >= self.max_items:
                try:
                    self.log_queue.get_nowait()
                except queue.Empty:
                    pass
            self.log_queue.put_nowait((record.levelname, msg))
        except Exception:
            self.handleError(record)


def setup_logger(
    config: dict,
    base_dir: Optional[Path] = None,
    log_queue: Optional["queue.Queue[tuple[str, str]]"] = None,
) -> logging.Logger:
    log_cfg = config.get("logging", {})
    level_name = str(log_cfg.get("level", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)

    save_dir_cfg = Path(log_cfg.get("save_dir", "logs"))
    if base_dir is not None and not save_dir_cfg.is_absolute():
        save_dir = base_dir / save_dir_cfg
    else:
        save_dir = save_dir_cfg
    save_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("ww_auto_quest")
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    log_file = save_dir / f"run_{time.strftime('%Y%m%d_%H%M%S')}.log"
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    if log_queue is not None:
        qh = QueueLogHandler(log_queue)
        qh.setLevel(level)
        qh.setFormatter(fmt)
        logger.addHandler(qh)

    return logger


def save_debug_screenshot(
    frame: Optional[np.ndarray],
    save_dir: str,
    prefix: str = "error",
) -> Optional[str]:
    if frame is None:
        return None
    out_dir = Path(save_dir) / "screenshots"
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = out_dir / f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}.png"
    ok, buf = cv2.imencode(".png", frame)
    if ok:
        try:
            filename.write_bytes(buf.tobytes())
        except Exception:
            return None
    return str(filename)


def notify_takeover(reason: str, logger: Optional[logging.Logger] = None) -> None:
    msg = f"[需要人工接管] {reason} —— 处理完成后请按 F7 恢复。"
    if logger:
        logger.warning(msg)
    else:
        print(msg)

    if winsound is not None:
        try:
            for freq in (880, 1175, 880):
                winsound.Beep(freq, 180)
                time.sleep(0.06)
        except RuntimeError:
            pass
