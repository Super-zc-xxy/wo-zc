import logging
import random
import time
from typing import Optional

import cv2
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
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.vision = vision
        self.inp = inp
        self.rois = config.get("rois", {})
        self.collector = collector
        self.logger = logger
        nav_cfg = config.get("navigation", {})
        win_cfg = config.get("window", {})

        self.base_width = int(win_cfg.get("base_width", 1920))
        self.sprint_hold_sec = float(nav_cfg.get("sprint_hold_sec", 0.45))
        self.turn_sensitivity = float(nav_cfg.get("turn_sensitivity", 0.42))
        self.center_deadzone_px = int(nav_cfg.get("center_deadzone_px", 70))
        self.far_distance_m = int(nav_cfg.get("far_distance_m", 300))

        # 三维立体避障与耐力状态监控
        self._sprint_cumulative_sec: float = 0.0
        self._last_stamina_rest_ts: float = 0.0
        self._prev_frame_gray: Optional[np.ndarray] = None
        self._fall_streak: int = 0
        self._last_jump_ts: float = 0.0

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

    def patrol_and_absorb_echo(self, frame: np.ndarray) -> bool:
        """
        战后声骸与战利品定向寻径吸附：
        1. 若已出现 [F] 交互提示，立即按 F 吸收
        2. 若未出现，扫描地面金/紫光柱或掉落物，微步向前并按 F 扩大拾取判定区
        """
        if self.detect_interact(frame):
            self.interact()
            return True

        ground_roi = self.rois.get("loot_ground", [480, 500, 1440, 920])
        sub, ox, oy = self.vision.crop_roi(frame, ground_roi)
        if sub.size > 0:
            hsv = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)
            # 金色声骸与紫色高亮光晕
            mask_gold = cv2.inRange(hsv, np.array([18, 160, 180]), np.array([32, 255, 255]))
            mask_purple = cv2.inRange(hsv, np.array([125, 140, 160]), np.array([155, 255, 255]))
            mask = cv2.bitwise_or(mask_gold, mask_purple)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            best_drop = None
            max_area = 0.0
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if 40 <= area <= 3000 and area > max_area:
                    max_area = area
                    M = cv2.moments(cnt)
                    if M["m00"] > 0:
                        best_drop = (ox + int(M["m10"] / M["m00"]), oy + int(M["m01"] / M["m00"]))

            if best_drop is not None:
                dx = best_drop[0] - (self.base_width // 2)
                if abs(dx) > 60:
                    self.inp.smooth_turn(int(dx * 0.35), steps=4)
                self.inp.key_down("w")
                time.sleep(0.28)
                self.inp.key_up("w")
                self.interact()
                return True

        # 盲探微步前进与微幅环视吸收
        self.inp.key_down("w")
        time.sleep(0.20)
        self.inp.key_up("w")
        self.interact()
        self.inp.smooth_turn(35, steps=3)
        return False

    def trigger_gadget(self) -> None:
        self.inp.press_key("t")
        time.sleep(0.3)

    def detect_falling(self, frame: np.ndarray) -> bool:
        """
        基于轻量下采样中心光流/像素位移检测角色是否处于高空极速下落状态。
        当角色从高处跌落时，画面景物会呈现强烈的垂直向上反向位移。
        """
        try:
            h, w = frame.shape[:2]
            # 截取中央 1/2 观察区域并缩小至 120x68
            center_crop = frame[int(h * 0.25) : int(h * 0.75), int(w * 0.25) : int(w * 0.75)]
            gray = cv2.cvtColor(center_crop, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(gray, (120, 68), interpolation=cv2.INTER_AREA)

            if self._prev_frame_gray is not None:
                # 快速计算相邻帧的光流均值（calcOpticalFlowFarneback）
                flow = cv2.calcOpticalFlowFarneback(
                    self._prev_frame_gray,
                    small,
                    None,
                    0.5,
                    2,
                    12,
                    2,
                    5,
                    1.1,
                    0,
                )
                dy = np.median(flow[..., 1])
                # 景物显著向上移动（dy < -3.2）代表角色正在快速向下坠落
                if dy < -3.2:
                    self._fall_streak += 1
                else:
                    self._fall_streak = max(0, self._fall_streak - 1)
            self._prev_frame_gray = small
            return self._fall_streak >= 2
        except Exception:
            return False

    def try_deploy_glider(self) -> bool:
        """在空中触发空格键展开滑翔伞，平稳滑降避免摔死。"""
        now = time.monotonic()
        if (now - self._last_jump_ts) > 0.8:
            if self.logger:
                self.logger.info("[滑翔保护] 检测到空中极速下落，自动触发空格键展开滑翔伞！")
            self.inp.press_key("space")
            self._last_jump_ts = now
            self._fall_streak = 0
            time.sleep(0.3)
            return True
        return False

    def step_towards_marker(self, frame: np.ndarray) -> Optional[str]:
        # 1. 跌落与滞空安全防护
        if self.detect_falling(frame):
            self.try_deploy_glider()
            return "gliding_protection"

        center_roi = self.rois.get("combat_center", [320, 70, 1600, 880])
        marker_data = None
        if hasattr(self.vision, "detect_yellow_quest_marker_with_elevation"):
            try:
                res = self.vision.detect_yellow_quest_marker_with_elevation(frame, roi=center_roi)
                if isinstance(res, tuple):
                    if len(res) == 3:
                        marker_data = res
                    elif len(res) == 2:
                        marker_data = (res[0], res[1], 0)
            except Exception:
                pass

        if marker_data is None:
            try:
                res2 = self.vision.detect_yellow_quest_marker(frame, roi=center_roi)
                if isinstance(res2, tuple) and len(res2) >= 2:
                    marker_data = (res2[0], res2[1], 0)
            except Exception:
                pass

        if marker_data is None:
            # 2. 主视野丢标：从小地图雷达边缘圆盘提取指针相对偏角
            azimuth = None
            if hasattr(self.vision, "detect_minimap_quest_azimuth"):
                try:
                    azimuth = self.vision.detect_minimap_quest_azimuth(frame)
                except Exception:
                    pass
            if azimuth is not None and isinstance(azimuth, (int, float)):
                turn_px = int(azimuth * 3.6)
                turn_px = max(-480, min(480, turn_px))
                if abs(turn_px) > 15:
                    if self.logger:
                        self.logger.info(f"[雷达导向] 主视野丢失黄标，小地图检测到目标偏角 {azimuth:.1f}°，精准转向 {turn_px}px 对准目标")
                    self.inp.smooth_turn(turn_px, steps=5)
                    return "radar_guided_turn"

            # 雷达亦未捕捉到黄标：原地常规 180 度大转角环视
            self.inp.smooth_turn(180, steps=6)
            return "scan_for_marker"

        mx, my, elevation = marker_data
        if self.collector:
            self.collector.collect_bbox(frame, "quest_marker.png", (mx - 18, my - 18, mx + 18, my + 18), pad=2)

        dx = mx - (self.base_width // 2)
        if abs(dx) > self.center_deadzone_px:
            turn_px = int(dx * self.turn_sensitivity)
            turn_px = max(-360, min(360, turn_px))
            self.inp.smooth_turn(turn_px, steps=5)

        # 3. 耐力过载保护与拟人化冲刺
        now = time.monotonic()
        self.inp.key_down("w")

        # 高度落差感知：若目标在高处(▲)，移动中在适当时机尝试轻触空格触发翻越跑酷
        if elevation == 1 and (now - self._last_jump_ts) > 3.0:
            self.inp.press_key("space")
            self._last_jump_ts = now

        # 若对准目标且耐力充足，开启冲刺；若连续冲刺超 3.2 秒，间歇慢跑回体
        can_sprint = abs(dx) <= self.center_deadzone_px * 2
        if can_sprint and self._sprint_cumulative_sec < 3.2:
            self.inp.key_down("shift")
            time.sleep(self.sprint_hold_sec)
            self.inp.key_up("shift")
            self._sprint_cumulative_sec += self.sprint_hold_sec
        else:
            time.sleep(self.sprint_hold_sec * 0.7)
            if self._sprint_cumulative_sec >= 3.2:
                self._sprint_cumulative_sec = 0.0

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
