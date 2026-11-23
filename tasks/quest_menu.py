import logging
import re
import time
from typing import List, Optional, Tuple

import numpy as np

from core.auto_collector import AutoTemplateCollector
from core.capture import ScreenCapture
from core.input_ctrl import InputController
from core.vision import VisionEngine


class QuestMenuHandler:
    """
    V2 任务面板自动切换模块（支持零模板 OCR 直驱 + 自动收集模板）：
    当左上角任务追踪栏长时间为空时，按 J 打开任务列表，
    按预设优先级挑选未完成任务并点击“追踪”。
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

    def _find_track_button(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        action_roi = self.rois.get("quest_menu_action", [1340, 860, 1880, 1050])
        hit = self.vision.match_template(frame, "btn_track.png", roi=action_roi, threshold=0.78)
        if hit is not None:
            return (hit.cx, hit.cy)

        lines = self.vision.ocr_roi(frame, action_roi, cache_key="quest_track_btn", force=True)
        for line in lines:
            if "取消追踪" in line.text:
                return None
            if "追踪" in line.text:
                x1, y1, x2, y2 = line.bbox
                if self.collector:
                    self.collector.collect_bbox(frame, "btn_track.png", line.bbox, pad=10)
                return ((x1 + x2) // 2, (y1 + y2) // 2)
        return None

    def switch_to_next_quest(
        self,
        priority_keywords: Optional[List[str]] = None,
        exclude_texts: Optional[List[str]] = None,
    ) -> bool:
        if self.logger:
            exc_hint = f"（已排除受阻任务: {', '.join(exclude_texts[:2])}）" if exclude_texts else ""
            self.logger.info("尝试按 J 打开任务列表切换下一个未完成任务%s...", exc_hint)
        self.inp.release_all()
        self.inp.press_key("j")
        time.sleep(0.9)

        frame = self.capture.grab(normalize=True)
        if frame is None:
            self.inp.press_key("esc")
            return False

        list_roi = self.rois.get("quest_menu_list", [100, 140, 660, 970])
        lines = self.vision.ocr_roi(frame, list_roi, cache_key="quest_menu_list", force=True)
        if not lines:
            if self.logger:
                self.logger.warning("任务列表未识别到可追踪条目，关闭任务面板。")
            self.inp.press_key("esc")
            time.sleep(0.4)
            return False

        category_headers = ("潮汐任务", "伴星任务", "危行任务", "纪闻任务", "每日任务", "日常", "任务列表", "追踪")

        def _is_excluded(t: str) -> bool:
            clean = re.sub(r"[\s：:，,。.！!（）()]", "", t).strip()
            if not clean or clean in category_headers:
                return True
            if exclude_texts:
                for exc in exclude_texts:
                    clean_exc = re.sub(r"[\s：:，,。.！!（）()]", "", exc).strip()
                    if len(clean_exc) >= 2 and (clean_exc in clean or clean in clean_exc):
                        return True
            return False

        # 过滤候选条目
        candidates = [ln for ln in lines if len(ln.text.strip()) >= 2 and not _is_excluded(ln.text)]
        if not candidates:
            # 宽容兜底：如果全部被排除，至少尝试非分类标题的条目
            candidates = [ln for ln in lines if len(ln.text.strip()) >= 2 and ln.text.strip() not in category_headers]

        if not candidates:
            if self.logger:
                self.logger.warning("任务列表中未找到可用的候选任务。")
            self.inp.press_key("esc")
            time.sleep(0.4)
            return False

        # 按优先级排序候选条目
        keywords = priority_keywords or ["潮汐任务", "伴星任务", "危行任务", "纪闻任务", "每日"]

        def _candidate_rank(ln) -> int:
            for idx, kw in enumerate(keywords):
                if kw in ln.text:
                    return idx
            return len(keywords)

        candidates.sort(key=_candidate_rank)

        # 逐个候选尝试点击并确认是否能追踪（最多尝试前 4 个候选）
        for cand in candidates[:4]:
            x1, y1, x2, y2 = cand.bbox
            sx, sy = self.capture.to_screen_coords((x1 + x2) // 2, (y1 + y2) // 2)
            self.inp.click_at(sx, sy)
            time.sleep(0.45)

            frame2 = self.capture.grab(normalize=True)
            if frame2 is None:
                continue

            track_btn = self._find_track_button(frame2)
            if track_btn is not None:
                tx, ty = self.capture.to_screen_coords(track_btn[0], track_btn[1])
                self.inp.click_at(tx, ty)
                time.sleep(0.4)
                if self.logger:
                    self.logger.info(f"已成功切换并追踪新任务：[{cand.text}]")
                self.inp.press_key("esc")
                time.sleep(0.4)
                return True
            else:
                if self.logger:
                    self.logger.info(f"条目 [{cand.text}] 当前不可追踪或已在追踪中，尝试下一个候选...")

        self.inp.press_key("esc")
        time.sleep(0.4)
        return False
