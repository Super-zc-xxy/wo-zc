import math
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
        self._last_roi_thumbs: Dict[str, np.ndarray] = {}
        self._last_global_ocr_ts: float = 0.0
        self._min_global_ocr_interval: float = 0.12  # 全局分时节流：单 tick 最多执行 1 次昂贵 OCR

    def load_template(self, name: str) -> Optional[np.ndarray]:
        if name in self._template_cache:
            return self._template_cache[name]
        path = self.assets_dir / name
        if not path.exists():
            self._template_cache[name] = None
            return None
        try:
            raw = np.frombuffer(path.read_bytes(), dtype=np.uint8)
            img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        except Exception:
            img = None
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
        # 1. 单项时间节流：未达到指定间隔直接复用
        if (
            not force
            and cache_key in self._last_ocr_ts
            and (now - self._last_ocr_ts[cache_key]) < self.ocr_interval_sec
        ):
            return self._last_ocr_res.get(cache_key, [])

        # 2. 全局分时节流：若本 tick 已有其他模块执行过 OCR，暂用缓存平滑分摊到后续 tick
        if not force and (now - self._last_global_ocr_ts) < self._min_global_ocr_interval:
            return self._last_ocr_res.get(cache_key, [])

        sub, ox, oy = self.crop_roi(frame, roi)
        if sub.size == 0:
            return []

        # 3. 视觉预筛（脏标记检测）：若该区域画面无明显变化（diff < 1.8），跳过推理
        thumb = cv2.resize(cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY), (32, 16))
        if not force and cache_key in self._last_roi_thumbs:
            diff = float(np.mean(cv2.absdiff(thumb, self._last_roi_thumbs[cache_key])))
            if diff < 1.8 and cache_key in self._last_ocr_res:
                self._last_ocr_ts[cache_key] = now
                return self._last_ocr_res[cache_key]

        raw_out = self._ocr(sub)
        self._last_global_ocr_ts = time.monotonic()
        self._last_roi_thumbs[cache_key] = thumb
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

    def detect_minimap_quest_azimuth(
        self,
        frame: np.ndarray,
        roi: Optional[List[int]] = None,
    ) -> Optional[float]:
        """
        从小地图（雷达圆盘）外圈提取任务指针，计算目标相对于当前视角的水平相对偏角（-180度 ~ +180度）。
        0度表示目标在正前方，90度表示在正右方，-90度表示在正左方，+/-180度表示在正后方。
        """
        roi = roi or [20, 20, 270, 270]
        sub, ox, oy = self.crop_roi(frame, roi)
        if sub.size == 0:
            return None

        h, w = sub.shape[:2]
        center_x, center_y = w // 2, h // 2
        min_r = int(w * 0.30)
        max_r = int(w * 0.49)

        hsv = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([16, 140, 160]), np.array([36, 255, 255]))

        ring_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(ring_mask, (center_x, center_y), max_r, 255, -1)
        cv2.circle(ring_mask, (center_x, center_y), min_r, 0, -1)
        candidate_mask = cv2.bitwise_and(mask, ring_mask)

        contours, _ = cv2.findContours(candidate_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_pt = None
        best_area = 0.0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 15 <= area <= 600:
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    px = int(M["m10"] / M["m00"])
                    py = int(M["m01"] / M["m00"])
                    dist = math.hypot(px - center_x, py - center_y)
                    if min_r <= dist <= max_r and area > best_area:
                        best_area = area
                        best_pt = (px, py)

        if best_pt is None:
            return None

        dx = best_pt[0] - center_x
        dy = best_pt[1] - center_y
        angle_rad = math.atan2(dx, -dy)
        return float(math.degrees(angle_rad))

    def detect_yellow_quest_marker_with_elevation(
        self, frame: np.ndarray, roi: Optional[List[int]] = None
    ) -> Optional[Tuple[int, int, int]]:
        """
        定位任务追踪菱形黄标并检测高度落差指示。
        返回 (cx, cy, elevation)：
        elevation:
          +1: 目标在高处 (黄标上方带 ▲ 三角)
          -1: 目标在低处 (黄标下方带 ▼ 三角)
           0: 目标在当前平视高度
        """
        tpl_hit = self.match_template(frame, "quest_marker.png", roi=roi, threshold=0.76)
        sub, ox, oy = self.crop_roi(frame, roi)
        if sub.size == 0:
            return None

        hsv = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([18, 150, 180]), np.array([34, 255, 255]))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        if tpl_hit is not None:
            cx_local = tpl_hit.cx - ox
            cy_local = tpl_hit.cy - oy
            elevation = self._check_marker_elevation(mask, cx_local, cy_local, half_w=16, half_h=16)
            return (tpl_hit.cx, tpl_hit.cy, elevation)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        best_score = -1.0
        center_x = sub.shape[1] // 2

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (50 <= area <= 2500):
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            ratio = w / max(1, h)
            if not (0.6 <= ratio <= 1.6):
                continue

            rect_area = float(w * h)
            fill_ratio = area / max(1.0, rect_area)
            if not (0.34 <= fill_ratio <= 0.70):
                continue

            cx = x + w // 2
            cy = y + h // 2
            shape_score = 1.0 - abs(fill_ratio - 0.50) * 2.0
            dist_factor = 1.0 - (abs(cx - center_x) / max(1, center_x)) * 0.25
            total_score = shape_score * dist_factor

            if total_score > best_score:
                best_score = total_score
                elev = self._check_marker_elevation(mask, cx, cy, half_w=w // 2, half_h=h // 2)
                best = (ox + cx, oy + cy, elev)

        return best

    @staticmethod
    def _check_marker_elevation(mask: np.ndarray, cx: int, cy: int, half_w: int, half_h: int) -> int:
        h, w = mask.shape[:2]
        y_up_start = max(0, cy - half_h - 14)
        y_up_end = max(0, cy - half_h)
        x_left = max(0, cx - half_w)
        x_right = min(w, cx + half_w)
        up_pixels = cv2.countNonZero(mask[y_up_start:y_up_end, x_left:x_right]) if y_up_end > y_up_start else 0

        y_down_start = min(h, cy + half_h)
        y_down_end = min(h, cy + half_h + 14)
        down_pixels = cv2.countNonZero(mask[y_down_start:y_down_end, x_left:x_right]) if y_down_end > y_down_start else 0

        if up_pixels > 12 and up_pixels > down_pixels * 1.5:
            return 1
        if down_pixels > 12 and down_pixels > up_pixels * 1.5:
            return -1
        return 0

    def detect_yellow_quest_marker(
        self, frame: np.ndarray, roi: Optional[List[int]] = None
    ) -> Optional[Tuple[int, int]]:
        """
        定位任务追踪菱形黄标。
        1. 优先模板匹配 quest_marker.png
        2. 无模板时采用 HSV + 菱形几何形态学特征过滤抗干扰识别（有效抗沙地、麦田、黄昏强光背景）
        """
        res = self.detect_yellow_quest_marker_with_elevation(frame, roi=roi)
        if res is not None:
            return (res[0], res[1])
        return None

    @staticmethod
    def parse_distance_m(text: str) -> Optional[int]:
        m = re.search(r"(\d+)\s*[mM米]", text)
        return int(m.group(1)) if m else None

    def get_player_hp_ratio(
        self,
        frame: np.ndarray,
        roi: Optional[List[int]] = None,
    ) -> Optional[float]:
        """
        提取当前主控角色的生命值百分比 (0.0 ~ 1.0)。
        默认底部血条 ROI: [780, 990, 1140, 1025]
        """
        roi = roi or [780, 990, 1140, 1025]
        sub, _, _ = self.crop_roi(frame, roi)
        if sub.size == 0:
            return None

        h, w = sub.shape[:2]
        hsv = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)
        mask_green = cv2.inRange(hsv, np.array([32, 70, 90]), np.array([90, 255, 255]))
        mask_white = cv2.inRange(hsv, np.array([0, 0, 185]), np.array([180, 50, 255]))
        mask = cv2.bitwise_or(mask_green, mask_white)

        col_counts = np.sum(mask > 0, axis=0)
        valid_cols = np.where(col_counts >= max(2, int(h * 0.15)))[0]
        if len(valid_cols) == 0:
            return None

        hp_width = float(len(valid_cols))
        ratio = hp_width / float(w)
        return float(max(0.0, min(1.0, ratio)))
