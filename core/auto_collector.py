import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from core.capture import ScreenCapture
from core.vision import VisionEngine


DEFAULT_ROIS_1080P: Dict[str, List[int]] = {
    "quest_panel": [35, 175, 470, 390],
    "skip_button": [30, 20, 290, 120],
    "confirm_dialog": [600, 340, 1320, 780],
    "dialogue_options": [1180, 440, 1850, 890],
    "interact_prompt": [1060, 380, 1580, 720],
    "minimap": [40, 20, 265, 245],
    "skill_bar": [1380, 840, 1910, 1070],
    "team_portraits": [1650, 220, 1910, 640],
    "combat_center": [320, 70, 1600, 880],
    "gadget_prompt": [820, 620, 1100, 850],
    "map_canvas": [110, 80, 1810, 1000],
    "map_right_panel": [1300, 740, 1890, 1050],
    "quest_menu_list": [100, 140, 660, 970],
    "quest_menu_action": [1340, 860, 1880, 1050],
    "deliver_panel": [580, 240, 1340, 880],
}


class AutoTemplateCollector:
    """
    后台自动 UI 模板收集器：
    当 OCR 或颜色/轮廓兜底识别成功命中某个 UI 元素时，自动裁剪并保存到 assets/，
    同时刷新 VisionEngine 缓存，让新手无需手动截图，越跑越快越准。
    """

    def __init__(
        self,
        assets_dir: Path,
        vision: VisionEngine,
        logger: Optional[logging.Logger] = None,
        enabled: bool = True,
    ) -> None:
        self.assets_dir = Path(assets_dir)
        self.assets_dir.mkdir(parents=True, exist_ok=True)
        self.vision = vision
        self.logger = logger
        self.enabled = enabled
        self._last_saved_ts: Dict[str, float] = {}

    def collect_bbox(
        self,
        frame: np.ndarray,
        template_name: str,
        bbox: Tuple[int, int, int, int],
        pad: int = 6,
        overwrite: bool = False,
    ) -> bool:
        if not self.enabled or frame is None:
            return False

        out_path = self.assets_dir / template_name
        if out_path.exists() and not overwrite:
            return False

        now = time.monotonic()
        if (now - self._last_saved_ts.get(template_name, 0.0)) < 10.0:
            return False

        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)

        cw, ch = x2 - x1, y2 - y1
        if cw < 16 or ch < 12 or cw > 480 or ch > 220:
            return False

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return False

        ok, buf = cv2.imencode(".png", crop)
        if not ok:
            return False
        try:
            out_path.write_bytes(buf.tobytes())
        except Exception:
            return False
        self._last_saved_ts[template_name] = now
        # 刷新 VisionEngine 缓存，使下次直接命中模板
        self.vision._template_cache[template_name] = crop
        if self.logger:
            self.logger.info("已自动收集 UI 模板: %s (%dx%d)", template_name, cw, ch)
        return True


class AutoCalibrator:
    """
    一键自动校准调试框（ROI）与自动初筛采集器：
    无需用户手动调坐标，自动绑定游戏窗口、按 1920x1080 标准生成自适应 ROI，
    并对当前画面执行一次全扫描，自动微调任务栏框并顺手收集当前可见模板。
    """

    def __init__(
        self,
        capture: ScreenCapture,
        vision: VisionEngine,
        collector: AutoTemplateCollector,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.capture = capture
        self.vision = vision
        self.collector = collector
        self.logger = logger

    def calibrate(self, config: dict) -> Tuple[bool, str, Dict[str, List[int]]]:
        rect = self.capture.find_window()
        if rect is None:
            return False, "未检测到鸣潮游戏窗口，请先启动游戏（建议 16:9 窗口化，如 1920x1080）。", {}

        frame = self.capture.grab(normalize=True)
        if frame is None:
            return False, "已找到游戏窗口但截屏失败，请确保游戏未最小化。", {}

        rois: Dict[str, List[int]] = {k: list(v) for k, v in DEFAULT_ROIS_1080P.items()}
        collected: List[str] = []

        # 1. 自动扫描左侧区域定位任务栏实际位置，动态微调 quest_panel
        broad_quest_roi = [20, 140, 560, 480]
        q_lines = self.vision.ocr_roi(frame, broad_quest_roi, cache_key="calib_quest", force=True)
        if q_lines:
            xs1 = [ln.bbox[0] for ln in q_lines]
            ys1 = [ln.bbox[1] for ln in q_lines]
            xs2 = [ln.bbox[2] for ln in q_lines]
            ys2 = [ln.bbox[3] for ln in q_lines]
            rois["quest_panel"] = [
                max(10, min(xs1) - 25),
                max(120, min(ys1) - 25),
                min(640, max(xs2) + 60),
                min(540, max(ys2) + 45),
            ]

        # 2. 顺手扫描当前画面可见元素并自动采集模板
        skip_lines = self.vision.ocr_roi(frame, rois["skip_button"], cache_key="calib_skip", force=True)
        for ln in skip_lines:
            if "跳过" in ln.text:
                if self.collector.collect_bbox(frame, "btn_skip.png", ln.bbox, pad=10):
                    collected.append("btn_skip.png")
                break

        f_lines = self.vision.ocr_roi(frame, rois["interact_prompt"], cache_key="calib_f", force=True)
        for ln in f_lines:
            if ln.text.upper() == "F" or "[F]" in ln.text.upper():
                if self.collector.collect_bbox(frame, "icon_f.png", ln.bbox, pad=8):
                    collected.append("icon_f.png")
                break

        marker = self.vision.detect_yellow_quest_marker(frame, roi=rois["combat_center"])
        if marker is not None:
            mx, my = marker
            if self.collector.collect_bbox(frame, "quest_marker.png", (mx - 18, my - 18, mx + 18, my + 18), pad=2):
                collected.append("quest_marker.png")

        config["rois"] = rois
        summary = (
            f"校准成功！已绑定 [{rect.title}] ({rect.width}x{rect.height})，"
            f"已自动生成 {len(rois)} 组调试框"
            + (f"，并自动收集模板：{', '.join(collected)}" if collected else "（其余模板将在运行中自动收集）。")
        )
        if self.logger:
            self.logger.info(summary)
        return True, summary, rois
