import logging
import math
import random
import time
from typing import Optional, Tuple

import numpy as np

from core.auto_collector import AutoTemplateCollector
from core.capture import ScreenCapture
from core.input_ctrl import InputController
from core.vision import VisionEngine


class TimeAdjuster:
    """
    游戏内时间自动调节器（全自动闭环）：
    当检测到任务需要“调整时间”、“等待至次日”时，
    1. 按 Esc 打开主菜单 / 终端界面
    2. 定位并点击“时间”图标（支持模板匹配 + 底部底栏 OCR 兜底）
    3. 在 24 小时表盘界面顺时针旋转拨动指针或点击次日
    4. 点击右下角“确认”按钮，等待游戏内时间加速过渡动画
    5. 按 Esc 退出菜单返回游戏世界
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
        self.config = config
        self.logger = logger
        self.collector = collector
        self.rois = config.get("rois", {})

    def _find_clock_icon(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """在菜单中寻找时间设置图标（左下或底栏）。"""
        bottom_roi = self.rois.get("menu_bottom_bar", [30, 840, 1890, 1060])
        hit = self.vision.match_template(frame, "icon_clock.png", roi=bottom_roi, threshold=0.72)
        if hit is not None:
            return (hit.cx, hit.cy)

        # OCR 兜底寻找“时间”文字
        lines = self.vision.ocr_roi(frame, bottom_roi, cache_key="menu_clock_ocr", force=True)
        for line in lines:
            if "时间" in line.text or "时钟" in line.text:
                x1, y1, x2, y2 = line.bbox
                if self.collector:
                    self.collector.collect_bbox(frame, "icon_clock.png", line.bbox, pad=6)
                return ((x1 + x2) // 2, (y1 + y2) // 2)

        # 鸣潮 1080p 终端菜单左下角/底栏默认时钟相对坐标保底 (约 x=135, y=985)
        return (135, 985)

    def _find_confirm_button(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """在时间调节表盘中寻找确认按钮。"""
        action_roi = self.rois.get("time_dialog_action", [1200, 800, 1880, 1050])
        hit = self.vision.match_template(frame, "btn_confirm.png", roi=action_roi, threshold=0.75)
        if hit is not None:
            return (hit.cx, hit.cy)

        lines = self.vision.ocr_roi(frame, action_roi, cache_key="time_confirm_ocr", force=True)
        for line in lines:
            if any(kw in line.text for kw in ("确认", "确定", "调整", "保存")):
                x1, y1, x2, y2 = line.bbox
                if self.collector:
                    self.collector.collect_bbox(frame, "btn_confirm.png", line.bbox, pad=8)
                return ((x1 + x2) // 2, (y1 + y2) // 2)

        # 1080p 右下角确定按钮默认保底坐标
        return (1520, 940)

    def adjust_time(self, forward_hours: int = 12) -> bool:
        """
        全自动执行调整时间闭环流程。
        forward_hours: 快进小时数（默认快进半天/12小时，通常足以触发次日或指定时间段）。
        """
        if self.logger:
            self.logger.info("开始执行游戏内时间自动调节闭环（预计快进 %d 小时）...", forward_hours)

        self.inp.release_all()
        self.inp.press_key("esc")
        time.sleep(0.9)

        # 1. 抓取菜单画面并寻找时间图标
        frame = self.capture.grab(normalize=True)
        if frame is None:
            self.inp.press_key("esc")
            return False

        clock_pos = self._find_clock_icon(frame)
        if clock_pos is None:
            if self.logger:
                self.logger.warning("菜单中未找到时间设置入口，关闭菜单。")
            self.inp.press_key("esc")
            return False

        sx, sy = self.capture.to_screen_coords(clock_pos[0], clock_pos[1])
        if self.logger:
            self.logger.info("点击时间设置入口: (%d, %d)", clock_pos[0], clock_pos[1])
        self.inp.click_at(sx, sy)
        time.sleep(0.9)

        # 2. 拨动表盘指针（顺时针旋转表盘）
        # 1080p 标准表盘圆心约 (960, 540)，半径约 220 像素
        frame_dial = self.capture.grab(normalize=True)
        if frame_dial is None:
            self.inp.press_key("esc")
            return False

        # 检查是否直接有“次日”按钮
        dial_roi = [600, 300, 1320, 780]
        lines = self.vision.ocr_roi(frame_dial, dial_roi, cache_key="time_dial_ocr", force=True)
        clicked_next_day = False
        for line in lines:
            if "次日" in line.text or "+1天" in line.text or "后一天" in line.text:
                x1, y1, x2, y2 = line.bbox
                dx, dy = self.capture.to_screen_coords((x1 + x2) // 2, (y1 + y2) // 2)
                self.inp.click_at(dx, dy)
                clicked_next_day = True
                if self.logger:
                    self.logger.info("已点击快速切换次日按钮")
                time.sleep(0.4)
                break

        if not clicked_next_day:
            # 顺时针圆弧平滑拖拽表盘指针：从 12 点方向顺时针拖动至 6 点方向
            # 12点: (960, 330) -> 3点: (1170, 540) -> 6点: (960, 750)
            cx, cy = 960, 540
            r = 210
            # 构造顺时针多段拖拽轨迹
            angles = [-math.pi / 2, -math.pi / 4, 0, math.pi / 4, math.pi / 2]
            coords = [
                self.capture.to_screen_coords(int(cx + r * math.cos(a)), int(cy + r * math.sin(a)))
                for a in angles
            ]
            if len(coords) >= 2:
                start_x, start_y = coords[0]
                end_x, end_y = coords[-1]
                self.inp.drag(start_x, start_y, end_x, end_y, steps=15, duration=0.45)
                time.sleep(0.3)

        # 3. 寻找并点击“确认”按钮
        frame_confirm = self.capture.grab(normalize=True)
        btn_pos = self._find_confirm_button(frame_confirm if frame_confirm is not None else frame_dial)
        if btn_pos:
            bx, by = self.capture.to_screen_coords(btn_pos[0], btn_pos[1])
            if self.logger:
                self.logger.info("点击时间调节确认按钮: (%d, %d)", btn_pos[0], btn_pos[1])
            self.inp.click_at(bx, by)
        else:
            # 兜底直接敲击 Enter 键或点击默认位置
            self.inp.press_key("enter")

        # 4. 等待游戏内昼夜交替时间加速动画（通常约 2.5 秒）
        if self.logger:
            self.logger.info("等待时间加速流逝过渡动画...")
        time.sleep(2.5)

        # 5. 安全按 Esc 确保完全退出菜单返回大世界
        self.inp.press_key("esc")
        time.sleep(0.6)

        if self.logger:
            self.logger.info("游戏时间自动调节已顺利完成！")
        return True
