import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    from rapidocr import RapidOCR
except ImportError:
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError:
        RapidOCR = None


@dataclass
class MatchResult:
    name: str
    cx: int
    cy: int
    score: float
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2) in 1920x1080 coords


@dataclass
class OCRLine:
    text: str
    score: float
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)


class VisionEngine:
    def __init__(
        self,
        assets_dir: str = "assets",
        default_threshold: float = 0.80,
        ocr_interval_sec: float = 1.0,
    ) -> None:
        self.assets_dir = Path(assets_dir)
        self.default_threshold = default_threshold
        self.ocr_interval_sec = ocr_interval_sec
        self._template_cache: Dict[str, Optional[np.ndarray]] = {}
        self._ocr = RapidOCR() if RapidOCR is not None else None
        self._last_ocr_ts: Dict[str, float] = {}
        self._last_ocr_res: Dict[str, List[OCRLine]] = {}

    def load_template(self, name: str) -> Optional[np.ndarray]:
        if name in self._template_cache:
            return self._template_cache[name]
        path = self.assets_dir / name
        if not path.exists():
            self._template_cache[name] = None
            return None
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        self._template_cache[name] = img
        return img

    @staticmethod
    def crop_roi(frame: np.ndarray, roi: Optional[List[int]]) -> Tuple[np.ndarray, int, int]:
        if not roi:
            return frame, 0, 0
        x1, y1, x2, y2 = roi
        h, w = frame.shape[:2]
        x1 = max(0, min(w, x1))
        y1 = max(0, min(h, y1))
        x2 = max(x1, min(w, x2))
        y2 = max(y1, min(h, y2))
        return frame[y1:y2, x1:x2], x1, y1

    def match_template(
        self,
        frame: np.ndarray,
        template_name: str,
        roi: Optional[List[int]] = None,
        threshold: Optional[float] = None,
    ) -> Optional[MatchResult]:
        tpl = self.load_template(template_name)
        if tpl is None:
            return None
        sub, ox, oy = self.crop_roi(frame, roi)
        th, tw = tpl.shape[:2]
        if sub.shape[0] < th or sub.shape[1] < tw:
            return None

        res = cv2.matchTemplate(sub, tpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        thr = threshold if threshold is not None else self.default_threshold
        if max_val < thr:
            return None

        x1 = ox + max_loc[0]
        y1 = oy + max_loc[1]
        x2 = x1 + tw
        y2 = y1 + th
        return MatchResult(
            name=template_name,
            cx=(x1 + x2) // 2,
            cy=(y1 + y2) // 2,
            score=float(max_val),
            bbox=(x1, y1, x2, y2),
        )

    def ocr_roi(
        self,
        frame: np.ndarray,
        roi: List[int],
        cache_key: str = "default",
        force: bool = False,
    ) -> List[OCRLine]:
        if self._ocr is None:
            return []

        now = time.monotonic()
        if (
            not force
            and cache_key in self._last_ocr_ts
            and (now - self._last_ocr_ts[cache_key]) < self.ocr_interval_sec
        ):
            return self._last_ocr_res.get(cache_key, [])

        sub, ox, oy = self.crop_roi(frame, roi)
        if sub.size == 0:
            return []

        raw_out = self._ocr(sub)
        lines: List[OCRLine] = []
        if hasattr(raw_out, "txts") and raw_out.txts is not None:
            boxes = raw_out.boxes if getattr(raw_out, "boxes", None) is not None else []
            txts = raw_out.txts
            scores = raw_out.scores if getattr(raw_out, "scores", None) is not None else [1.0] * len(txts)
            for box, text, score in zip(boxes, txts, scores):
                xs = [int(p[0]) for p in box]
                ys = [int(p[1]) for p in box]
                lines.append(
                    OCRLine(
                        text=str(text).strip(),
                        score=float(score),
                        bbox=(ox + min(xs), oy + min(ys), ox + max(xs), oy + max(ys)),
                    )
                )
        elif isinstance(raw_out, tuple) and raw_out[0]:
            for item in raw_out[0]:
                box, text, score = item
                xs = [int(p[0]) for p in box]
                ys = [int(p[1]) for p in box]
                lines.append(
                    OCRLine(
                        text=str(text).strip(),
                        score=float(score),
                        bbox=(ox + min(xs), oy + min(ys), ox + max(xs), oy + max(ys)),
                    )
                )

        self._last_ocr_ts[cache_key] = now
        self._last_ocr_res[cache_key] = lines
        return lines

    def detect_yellow_quest_marker(
        self, frame: np.ndarray, roi: Optional[List[int]] = None
    ) -> Optional[Tuple[int, int]]:
        """
        在未提供模板时，基于 HSV 颜色兜底定位任务追踪菱形黄标。
        优先建议仍使用 quest_marker.png 模板匹配。
        """
        tpl_hit = self.match_template(frame, "quest_marker.png", roi=roi, threshold=0.76)
        if tpl_hit is not None:
            return (tpl_hit.cx, tpl_hit.cy)

        sub, ox, oy = self.crop_roi(frame, roi)
        hsv = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)
        # 鸣潮任务黄标近似 HSV 区间
        mask = cv2.inRange(hsv, np.array([18, 150, 180]), np.array([34, 255, 255]))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        best_area = 0.0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 60 <= area <= 1800:
                x, y, w, h = cv2.boundingRect(cnt)
                ratio = w / max(1, h)
                if 0.6 <= ratio <= 1.5 and area > best_area:
                    best_area = area
                    best = (ox + x + w // 2, oy + y + h // 2)
        return best

    @staticmethod
    def parse_distance_m(text: str) -> Optional[int]:
        m = re.search(r"(\d+)\s*[mM米]", text)
        return int(m.group(1)) if m else None
