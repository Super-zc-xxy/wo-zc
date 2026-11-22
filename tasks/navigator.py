import random
import time
from typing import Optional

import numpy as np

from core.auto_collector import AutoTemplateCollector
from core.input_ctrl import InputController
from core.vision import VisionEngine


class Navigator:
    def __init__(
        self,
        vision: VisionEngine,
        inp: InputController,
        config: dict,
        collector: Optional[AutoTemplateCollector] = None,
    ) -> None:
        self.vision = vision
        self.inp = inp
        self.rois = config.get("rois", {})
        self.collector = collector
        nav_cfg = config.get("navigation", {})
        win_cfg = config.get("window", {})

        self.base_width = int(win_cfg.get("base_width", 1920))
        self.sprint_hold_sec = float(nav_cfg.get("sprint_hold_sec", 0.45))
        self.turn_sensitivity = float(nav_cfg.get("turn_sensitivity", 0.42))
        self.center_deadzone_px = int(nav_cfg.get("center_deadzone_px", 70))
        self.far_distance_m = int(nav_cfg.get("far_distance_m", 300))

    def detect_interact(self, frame: np.ndarray) -> bool:
        roi = self.rois.get("interact_prompt", [1060, 380, 1580, 720])
        if self.vision.match_template(frame, "icon_f.png", roi=roi) is not None:
            return True

        # 零模板 OCR 兜底检测 [F] 交互提示，并自动收集模板
        lines = self.vision.ocr_roi(frame, roi, cache_key="interact_f_ocr")
        for ln in lines:
            txt = ln.text.strip().upper()
            if txt == "F" or "[F]" in txt or any(kw in ln.text for kw in ("对话", "调查", "开启", "吸收", "激活", "交互", "拾取", "进入")):
                if self.collector and txt in ("F", "[F]"):
                    self.collector.collect_bbox(frame, "icon_f.png", ln.bbox, pad=8)
                return True
        return False

    def detect_gadget(self, frame: np.ndarray) -> bool:
        roi = self.rois.get("gadget_prompt", [820, 620, 1100, 850])
        if self.vision.match_template(frame, "icon_t.png", roi=roi) is not None:
            return True

        lines = self.vision.ocr_roi(frame, roi, cache_key="gadget_t_ocr")
        for ln in lines:
            txt = ln.text.strip().upper()
            if txt == "T" or "[T]" in txt:
                if self.collector:
                    self.collector.collect_bbox(frame, "icon_t.png", ln.bbox, pad=8)
                return True
        return False

    def interact(self) -> None:
        self.inp.press_key("f")
        time.sleep(0.25)

    def trigger_gadget(self) -> None:
        self.inp.press_key("t")
        time.sleep(0.3)

    def step_towards_marker(self, frame: np.ndarray) -> Optional[str]:
        center_roi = self.rois.get("combat_center", [320, 70, 1600, 880])
        marker = self.vision.detect_yellow_quest_marker(frame, roi=center_roi)
        if marker is None:
            self.inp.smooth_turn(180, steps=6)
            return "scan_for_marker"

        mx, my = marker
        if self.collector:
            self.collector.collect_bbox(frame, "quest_marker.png", (mx - 18, my - 18, mx + 18, my + 18), pad=2)

        dx = mx - (self.base_width // 2)
        if abs(dx) > self.center_deadzone_px:
            turn_px = int(dx * self.turn_sensitivity)
            turn_px = max(-360, min(360, turn_px))
            self.inp.smooth_turn(turn_px, steps=5)

        self.inp.key_down("w")
        if abs(dx) <= self.center_deadzone_px * 2:
            self.inp.key_down("shift")
            time.sleep(self.sprint_hold_sec)
            self.inp.key_up("shift")
        else:
            time.sleep(self.sprint_hold_sec * 0.7)
        self.inp.key_up("w")
        return "move_to_marker"

    def unstuck(self, attempt: int) -> None:
        self.inp.release_all()
        side_key = "a" if attempt % 2 == 1 else "d"
        turn_dir = -220 if attempt % 2 == 1 else 220

        self.inp.key_down("s")
        time.sleep(0.35)
        self.inp.key_up("s")

        self.inp.smooth_turn(turn_dir + random.randint(-40, 40), steps=6)
        self.inp.key_down("w")
        self.inp.key_down(side_key)
        time.sleep(0.2)
        self.inp.press_key("space")
        time.sleep(0.55)
        self.inp.key_up(side_key)
        self.inp.key_up("w")
