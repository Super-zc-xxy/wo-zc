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

    def _find_beacons(self, frame: np.ndarray, roi: List[int]) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """
        在大地图视口中同时提取普通信标与大型界碑神像（Nexus）。
        返回 (普通信标列表, 界碑神像列表)。
        """
        normal_beacons: List[Tuple[int, int]] = []
        nexus_beacons: List[Tuple[int, int]] = []

        # 1. 模板匹配普通信标与神像
        tpl_beacon = self.vision.match_template(frame, "icon_beacon.png", roi=roi, threshold=0.75)
        if tpl_beacon is not None:
            normal_beacons.append((tpl_beacon.cx, tpl_beacon.cy))

        tpl_nexus = self.vision.match_template(frame, "icon_nexus.png", roi=roi, threshold=0.75)
        if tpl_nexus is not None:
            nexus_beacons.append((tpl_nexus.cx, tpl_nexus.cy))

        # 2. 图像青蓝高亮特征提取
        sub, ox, oy = self.vision.crop_roi(frame, roi)
        if sub.size == 0:
            return normal_beacons, nexus_beacons

        hsv = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([88, 120, 170]), np.array([112, 255, 255]))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 35 <= area <= 2200:
                x, y, w, h = cv2.boundingRect(cnt)
                ratio = w / max(1, h)
                if 0.5 <= ratio <= 1.8:
                    cx, cy = ox + x + w // 2, oy + y + h // 2
                    # 面积较大（>= 360）的青蓝图标多为大型界碑/神像
                    if area >= 360:
                        nexus_beacons.append((cx, cy))
                    else:
                        normal_beacons.append((cx, cy))

        return normal_beacons, nexus_beacons

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
            self.logger.info("尝试打开大地图并传送到任务附近信标/神像...")
        self.inp.release_all()
        self.inp.press_key("m")
        time.sleep(1.0)

        frame = self.capture.grab(normalize=True)
        if frame is None:
            self.inp.press_key("esc")
            return False

        map_roi = self.rois.get("map_canvas", [110, 80, 1810, 1000])
        quest_pos = self.vision.detect_yellow_quest_marker(frame, roi=map_roi)

        # 智能全景视口自适应缩放巡航：当前视口无黄标时向下滚轮缩小地图
        if quest_pos is None:
            for zoom_attempt in range(1, 3):
                if self.logger:
                    self.logger.info(f"[全景寻标] 当前视口未定位到任务黄标，尝试第 {zoom_attempt} 档缩小地图全景视口...")
                self.inp.scroll(-160)
                time.sleep(0.45)
                frame_zoomed = self.capture.grab(normalize=True)
                if frame_zoomed is not None:
                    frame = frame_zoomed
                    quest_pos = self.vision.detect_yellow_quest_marker(frame, roi=map_roi)
                    if quest_pos is not None:
                        if self.logger:
                            self.logger.info("[全景寻标] 缩放后已成功捕获跨区域任务黄标！")
                        break

        if quest_pos is None:
            if self.logger:
                self.logger.warning("大地图全景缩放后仍未定位到任务黄标，关闭地图。")
            self.inp.press_key("esc")
            time.sleep(0.4)
            return False

        normal_beacons, nexus_beacons = self._find_beacons(frame, map_roi)
        all_candidates = normal_beacons + nexus_beacons
        if not all_candidates:
            if self.logger:
                self.logger.warning("大地图中未找到可用信标或神像，关闭地图。")
            self.inp.press_key("esc")
            time.sleep(0.4)
            return False

        # 目标信标智能决策：优先考虑神像界碑（全队回满血+复活）
        selected_target = None
        if nexus_beacons and normal_beacons:
            nearest_nexus = min(nexus_beacons, key=lambda p: math.hypot(p[0] - quest_pos[0], p[1] - quest_pos[1]))
            nearest_normal = min(normal_beacons, key=lambda p: math.hypot(p[0] - quest_pos[0], p[1] - quest_pos[1]))
            dist_nexus = math.hypot(nearest_nexus[0] - quest_pos[0], nearest_nexus[1] - quest_pos[1])
            dist_normal = math.hypot(nearest_normal[0] - quest_pos[0], nearest_normal[1] - quest_pos[1])

            # 若神像到任务的距离并不比普通信标远太多（<= 180px），优先选神像
            if dist_nexus <= dist_normal + 180:
                selected_target = nearest_nexus
                if self.logger:
                    self.logger.info("[神像优先] 任务附近检测到大型界碑/枢纽神像，优先传送以免费复活并回满全队生命值！")
            else:
                selected_target = nearest_normal
        elif nexus_beacons:
            selected_target = min(nexus_beacons, key=lambda p: math.hypot(p[0] - quest_pos[0], p[1] - quest_pos[1]))
        else:
            selected_target = min(normal_beacons, key=lambda p: math.hypot(p[0] - quest_pos[0], p[1] - quest_pos[1]))

        if self.collector and selected_target is not None:
            nx, ny = selected_target
            self.collector.collect_bbox(frame, "icon_beacon.png", (nx - 16, ny - 16, nx + 16, ny + 16), pad=2)

        sx, sy = self.capture.to_screen_coords(selected_target[0], selected_target[1])
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
            self.logger.info("已触发快速旅行传送，等待区域加载完成...")
        time.sleep(1.2)
        return True
