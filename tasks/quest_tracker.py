from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from core.vision import VisionEngine


COMBAT_KEYWORDS = ("击败", "消灭", "清理", "战胜", "讨伐", "击倒", "清除", "守护")
TIME_KEYWORDS = ("调整时间", "次日", "等待至", "调至")
TAKEOVER_KEYWORDS = ("拍照", "潜行", "跟踪", "拼图", "连线", "解谜")


@dataclass
class QuestInfo:
    raw_text: str
    lines: List[str]
    distance_m: Optional[int]
    need_combat: bool
    need_time_adjust: bool
    need_takeover: bool


class QuestTracker:
    def __init__(self, vision: VisionEngine, config: dict) -> None:
        self.vision = vision
        self.rois = config.get("rois", {})
        self.last_info: Optional[QuestInfo] = None

    def read(self, frame: np.ndarray, force: bool = False) -> QuestInfo:
        roi = self.rois.get("quest_panel", [40, 190, 440, 360])
        ocr_lines = self.vision.ocr_roi(frame, roi, cache_key="quest_panel", force=force)
        texts = [line.text for line in ocr_lines if line.text]
        raw_text = " ".join(texts)

        distance_m = self.vision.parse_distance_m(raw_text)
        need_combat = any(kw in raw_text for kw in COMBAT_KEYWORDS)
        need_time_adjust = any(kw in raw_text for kw in TIME_KEYWORDS)
        need_takeover = any(kw in raw_text for kw in TAKEOVER_KEYWORDS)

        info = QuestInfo(
            raw_text=raw_text,
            lines=texts,
            distance_m=distance_m,
            need_combat=need_combat,
            need_time_adjust=need_time_adjust,
            need_takeover=need_takeover,
        )
        self.last_info = info
        return info
