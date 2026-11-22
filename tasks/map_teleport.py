import logging
import math
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

from core.auto_collector import AutoTemplateCollector
from core.capture import ScreenCapture
from core.input_ctrl import InputController
from core.vision import VisionEngine


class MapTeleporter:
    """
    V2 大地图自动传送模块（支持零模板 OCR 直驱 + 自动收集模板）：
    1. 按 M 打开大地图
    2. 定位任务黄标与距离任务黄标最近的信标（模板匹配 + 青蓝色 HSV 兜底）
    3. 点击信标 -> 识别并点击右下角“快速旅行”按钮 -> 等待加载
    """

    def __init__(
        self,
        vision: VisionEngine,
        capture: ScreenCapture,
        inp: InputController,
        config: dict,
        logger: Optional[logging.Logger] = None,
        collector: Optional[AutoTemplateCollector] = None,
    ) -> None:
        self.vision = vision
        self.capture = capture
        self.inp = inp
        self.rois = config.get("rois", {})
        self.logger = logger
        self.collector = collector

    def _find_beacons(self, frame: np.ndarray, roi: List[int]) -> List[Tuple[int, int]]:
        beacons: List[Tuple[int, int]] = []
        tpl_hit = self.vision.match_template(frame, "icon_beacon.png", roi=roi, threshold=0.75)
        if tpl_hit is not None:
            beacons.append((tpl_hit.cx, tpl_hit.cy))

        sub, ox, oy = self.vision.crop_roi(frame, roi)
        hsv = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([88, 120, 170]), np.array([112, 255, 255]))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 35 <= area <= 1400:
                x, y, w, h = cv2.boundingRect(cnt)
                ratio = w / max(1, h)
                if 0.5 <= ratio <= 1.8:
                    cx, cy = ox + x + w // 2, oy + y + h // 2
                    beacons.append((cx, cy))
        return beacons

    def _find_teleport_button(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        panel_roi = self.rois.get("map_right_panel", [1300, 740, 1890, 1050])
        hit = self.vision.match_template(frame, "btn_teleport.png", roi=panel_roi, threshold=0.78)
        if hit is not None:
            return (hit.cx, hit.cy)

        lines = self.vision.ocr_roi(frame, panel_roi, cache_key="map_teleport_btn", force=True)
        for line in lines:
            if any(kw in line.text for kw in ("快速旅行", "传送", "前往")):
                x1, y1, x2, y2 = line.bbox
                if self.collector:
                    self.collector.collect_bbox(frame, "btn_teleport.png", line.bbox, pad=10)
                return ((x1 + x2) // 2, (y1 + y2) // 2)
        return None

    def teleport_to_quest(self) -> bool:
        if self.logger:
            self.logger.info("尝试打开大地图并传送到任务附近信标...")
        self.inp.release_all()
        self.inp.press_key("m")
        time.sleep(1.0)

        frame = self.capture.grab(normalize=True)
        if frame is None:
            self.inp.press_key("esc")
            return False

        map_roi = self.rois.get("map_canvas", [110, 80, 1810, 1000])
        quest_pos = self.vision.detect_yellow_quest_marker(frame, roi=map_roi)
        if quest_pos is None:
            if self.logger:
                self.logger.warning("大地图中未定位到任务黄标，关闭地图。")
            self.inp.press_key("esc")
            time.sleep(0.4)
            return False

        beacons = self._find_beacons(frame, map_roi)
        if not beacons:
            if self.logger:
                self.logger.warning("大地图中未找到可用信标，关闭地图。")
            self.inp.press_key("esc")
            time.sleep(0.4)
            return False

        nearest = min(
            beacons,
            key=lambda p: math.hypot(p[0] - quest_pos[0], p[1] - quest_pos[1]),
        )
        if self.collector:
            nx, ny = nearest
            self.collector.collect_bbox(frame, "icon_beacon.png", (nx - 16, ny - 16, nx + 16, ny + 16), pad=2)

        sx, sy = self.capture.to_screen_coords(nearest[0], nearest[1])
        self.inp.click_at(sx, sy)
        time.sleep(0.6)

        frame2 = self.capture.grab(normalize=True)
        if frame2 is None:
            self.inp.press_key("esc")
            return False

        tp_btn = self._find_teleport_button(frame2)
        if tp_btn is None:
            if self.logger:
                self.logger.warning("点击信标后未检测到“快速旅行”按钮，关闭地图。")
            self.inp.press_key("esc")
            time.sleep(0.4)
            return False

        bx, by = self.capture.to_screen_coords(tp_btn[0], tp_btn[1])
        self.inp.click_at(bx, by)
        if self.logger:
            self.logger.info("已触发快速旅行，等待加载完成...")
        time.sleep(1.2)
        return True
