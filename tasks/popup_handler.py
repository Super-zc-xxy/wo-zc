import time
from typing import Optional

import numpy as np

from core.auto_collector import AutoTemplateCollector
from core.capture import ScreenCapture
from core.input_ctrl import InputController
from core.vision import MatchResult, VisionEngine


class PopupHandler:
    """
    零模板直驱弹窗与物品交付处理器：
    优先模板匹配，无模板时通过 OCR 识别“复苏/复活/交付/提交/关闭”，
    命中后自动收集模板到 assets/。
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

    def detect_popup(self, frame: np.ndarray) -> Optional[MatchResult]:
        for tpl in ("btn_revive.png", "btn_deliver.png", "btn_close.png"):
            hit = self.vision.match_template(frame, tpl)
            if hit is not None:
                return hit

        # 零模板 OCR 兜底：检测中央弹窗区的复苏/交付按钮
        dialog_roi = self.rois.get("confirm_dialog", [600, 340, 1320, 780])
        lines = self.vision.ocr_roi(frame, dialog_roi, cache_key="popup_dialog_ocr")
        for ln in lines:
            x1, y1, x2, y2 = ln.bbox
            if any(kw in ln.text for kw in ("复苏", "复活", "重新挑战")):
                if self.collector:
                    self.collector.collect_bbox(frame, "btn_revive.png", ln.bbox, pad=10)
                return MatchResult("btn_revive.png", (x1 + x2) // 2, (y1 + y2) // 2, ln.score, ln.bbox)
            if any(kw in ln.text for kw in ("交付", "提交")):
                if self.collector:
                    self.collector.collect_bbox(frame, "btn_deliver.png", ln.bbox, pad=10)
                return MatchResult("btn_deliver.png", (x1 + x2) // 2, (y1 + y2) // 2, ln.score, ln.bbox)
        return None

    def handle(self, frame: np.ndarray) -> Optional[str]:
        hit = self.detect_popup(frame)
        if hit is None:
            return None

        if hit.name == "btn_deliver.png":
            deliver_roi = self.rois.get("deliver_panel", [580, 240, 1340, 880])
            slot_hit = self.vision.match_template(frame, "deliver_item_slot.png", roi=deliver_roi)
            if slot_hit is not None:
                ix, iy = self.capture.to_screen_coords(slot_hit.cx, slot_hit.cy)
                self.inp.click_at(ix, iy)
                time.sleep(0.25)

        sx, sy = self.capture.to_screen_coords(hit.cx, hit.cy)
        self.inp.click_at(sx, sy)
        time.sleep(0.35)
        return hit.name
