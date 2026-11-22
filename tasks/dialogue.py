import time
from typing import Optional, Tuple

import numpy as np

from core.auto_collector import AutoTemplateCollector
from core.capture import ScreenCapture
from core.input_ctrl import InputController
from core.vision import MatchResult, VisionEngine


class DialogueHandler:
    """
    零模板直驱剧情与对话处理器：
    优先使用已收集的模板，若模板不存在则自动使用 OCR 识别“跳过 / 确认 / 自动播放 / 对话选项”，
    命中后自动将区域保存为 UI 模板。
    """

    def __init__(
        self,
        vision: VisionEngine,
        capture: ScreenCapture,
        inp: InputController,
        config: dict,
        collector: Optional[AutoTemplateCollector] = None,
    ) -> None:
        self.vision = vision
        self.capture = capture
        self.inp = inp
        self.rois = config.get("rois", {})
        self.collector = collector

    def detect_skip_button(self, frame: np.ndarray) -> Optional[MatchResult]:
        roi = self.rois.get("skip_button", [30, 20, 290, 120])
        hit = self.vision.match_template(frame, "btn_skip.png", roi=roi)
        if hit is not None:
            return hit

        lines = self.vision.ocr_roi(frame, roi, cache_key="skip_btn_ocr")
        for ln in lines:
            if "跳过" in ln.text or "skip" in ln.text.lower():
                x1, y1, x2, y2 = ln.bbox
                if self.collector:
                    self.collector.collect_bbox(frame, "btn_skip.png", ln.bbox, pad=10)
                return MatchResult("btn_skip.png", (x1 + x2) // 2, (y1 + y2) // 2, ln.score, ln.bbox)
        return None

    def detect_confirm_button(self, frame: np.ndarray) -> Optional[MatchResult]:
        roi = self.rois.get("confirm_dialog", [600, 340, 1320, 780])
        hit = self.vision.match_template(frame, "btn_confirm.png", roi=roi)
        if hit is not None:
            return hit

        lines = self.vision.ocr_roi(frame, roi, cache_key="confirm_btn_ocr")
        for ln in lines:
            if any(kw in ln.text for kw in ("确认", "确定", "跳过剧情")):
                x1, y1, x2, y2 = ln.bbox
                if self.collector:
                    self.collector.collect_bbox(frame, "btn_confirm.png", ln.bbox, pad=10)
                return MatchResult("btn_confirm.png", (x1 + x2) // 2, (y1 + y2) // 2, ln.score, ln.bbox)
        return None

    def detect_dialogue_option(self, frame: np.ndarray) -> Optional[MatchResult]:
        roi = self.rois.get("dialogue_options", [1180, 440, 1850, 890])
        hit = self.vision.match_template(frame, "dialogue_option.png", roi=roi)
        if hit is not None:
            return hit
        return None

    def _detect_auto_play_ocr(self, frame: np.ndarray) -> bool:
        roi = self.rois.get("skip_button", [30, 20, 290, 120])
        if self.vision.match_template(frame, "icon_auto_play.png", roi=roi) is not None:
            return True
        lines = self.vision.ocr_roi(frame, roi, cache_key="skip_btn_ocr")
        for ln in lines:
            if any(kw in ln.text for kw in ("自动", "播放", "隐藏")):
                if self.collector:
                    self.collector.collect_bbox(frame, "icon_auto_play.png", ln.bbox, pad=8)
                return True
        return False

    def is_in_dialogue(self, frame: np.ndarray) -> bool:
        if self.detect_skip_button(frame) is not None:
            return True
        if self.detect_dialogue_option(frame) is not None:
            return True
        if self._detect_auto_play_ocr(frame):
            return True
        return False

    def step(self, frame: np.ndarray) -> bool:
        confirm = self.detect_confirm_button(frame)
        if confirm is not None:
            sx, sy = self.capture.to_screen_coords(confirm.cx, confirm.cy)
            self.inp.click_at(sx, sy)
            time.sleep(0.25)
            return True

        skip = self.detect_skip_button(frame)
        if skip is not None:
            sx, sy = self.capture.to_screen_coords(skip.cx, skip.cy)
            self.inp.click_at(sx, sy)
            time.sleep(0.25)
            return True

        opt = self.detect_dialogue_option(frame)
        if opt is not None:
            sx, sy = self.capture.to_screen_coords(opt.cx, opt.cy)
            self.inp.click_at(sx, sy)
            time.sleep(0.2)
            return True

        if self._detect_auto_play_ocr(frame):
            # 不可跳过剧情：按 F 并顺手点一下右侧选项区常见位置
            self.inp.press_key("f")
            time.sleep(0.12)
            return True

        return False
