import logging
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

from core.input_ctrl import InputController
from core.vision import VisionEngine


class CombatHandler:
    """
    V2 自动战斗模块：
    - 进战/脱战缓冲（防多波次刷怪间隙误判）
    - 锁敌、普攻/重击/闪避/技能/大招/切人轮切
    - 支持协奏满头像高亮检测与大招就绪检测
    - 支持声骸 Q 技主动释放与红光突刺极限闪避反击
    - 支持通过预设动态热更新战斗参数与连招表
    """

    def __init__(
        self,
        vision: VisionEngine,
        inp: InputController,
        config: dict,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.vision = vision
        self.inp = inp
        self.config = config
        self.logger = logger
        self.rois = config.get("rois", {})
        combat_cfg = config.get("combat", {})
        self.exit_grace_sec = float(combat_cfg.get("exit_grace_sec", 2.5))
        self.lock_on_key = str(combat_cfg.get("lock_on_key", "middle"))
        self.use_concerto_switch = bool(combat_cfg.get("use_concerto_switch", True))
        self.use_ult_ready = bool(combat_cfg.get("use_ult_ready", True))
        self.auto_echo_skill = bool(combat_cfg.get("auto_echo_skill", True))
        self.echo_cooldown_sec = float(combat_cfg.get("echo_cooldown_sec", 15.0))
        self.auto_dodge_counter = bool(combat_cfg.get("auto_dodge_counter", True))
        self.rotation: List[dict] = list(combat_cfg.get("rotation", []))

        self._last_combat_ts = 0.0
        self._last_echo_ts = 0.0
        self._last_dodge_counter_ts = 0.0
        self._last_parry_ts = 0.0
        self._rotation_idx = 0
        self._current_char = 1
        self._locked_in_encounter = False

    def apply_preset(self, combat_cfg: dict) -> None:
        if not combat_cfg:
            return
        self.exit_grace_sec = float(combat_cfg.get("exit_grace_sec", self.exit_grace_sec))
        self.lock_on_key = str(combat_cfg.get("lock_on_key", self.lock_on_key))
        self.use_concerto_switch = bool(combat_cfg.get("use_concerto_switch", self.use_concerto_switch))
        self.use_ult_ready = bool(combat_cfg.get("use_ult_ready", self.use_ult_ready))
        self.auto_echo_skill = bool(combat_cfg.get("auto_echo_skill", self.auto_echo_skill))
        self.echo_cooldown_sec = float(combat_cfg.get("echo_cooldown_sec", self.echo_cooldown_sec))
        self.auto_dodge_counter = bool(combat_cfg.get("auto_dodge_counter", self.auto_dodge_counter))
        if "rotation" in combat_cfg and isinstance(combat_cfg["rotation"], list):
            self.rotation = list(combat_cfg["rotation"])
            self._rotation_idx = 0

    def _crop(self, frame: np.ndarray, roi: List[int]) -> Tuple[np.ndarray, int, int]:
        try:
            res = self.vision.crop_roi(frame, roi)
            if isinstance(res, tuple) and len(res) == 3:
                return res
        except Exception:
            pass
        return VisionEngine.crop_roi(frame, roi)

    def _has_danger_flash(self, frame: np.ndarray) -> bool:
        """检测中央视野是否出现敌人攻击前摇的高饱和红橙色光效警示。"""
        roi = self.rois.get("combat_center", [360, 80, 1560, 860])
        sub, _, _ = self._crop(frame, roi)
        if sub.size == 0:
            return False
        hsv = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, np.array([0, 170, 200]), np.array([10, 255, 255]))
        mask2 = cv2.inRange(hsv, np.array([170, 170, 200]), np.array([180, 255, 255]))
        mask = cv2.bitwise_or(mask1, mask2)
        return cv2.countNonZero(mask) >= 320

    def _has_parry_flash(self, frame: np.ndarray) -> bool:
        """检测敌人招式前摇中出现的金色拼刀光环（Parry Circle）。"""
        roi = self.rois.get("combat_center", [360, 80, 1560, 860])
        sub, _, _ = self._crop(frame, roi)
        if sub.size == 0:
            return False
        hsv = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([18, 140, 210]), np.array([36, 255, 255]))
        return cv2.countNonZero(mask) >= 280

    def _has_enemy_hp_bar(self, frame: np.ndarray) -> bool:
        roi = self.rois.get("combat_center", [360, 80, 1560, 860])
        sub, _, _ = self._crop(frame, roi)
        hsv = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, np.array([0, 160, 140]), np.array([8, 255, 255]))
        mask2 = cv2.inRange(hsv, np.array([170, 160, 140]), np.array([180, 255, 255]))
        mask = cv2.bitwise_or(mask1, mask2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w >= 45 and 3 <= h <= 16 and (w / max(1, h)) >= 4.5:
                return True
        return False

    def _detect_concerto_ready_char(self, frame: np.ndarray) -> Optional[int]:
        """检测右侧队伍头像栏是否有协奏/变奏就绪高亮，返回建议切换的角色位 (1/2/3)。"""
        if not self.use_concerto_switch:
            return None
        roi = self.rois.get("team_portraits", [1680, 240, 1900, 620])
        hit = self.vision.match_template(frame, "icon_concerto_ready.png", roi=roi, threshold=0.78)
        if hit is None:
            return None
        y1, y2 = roi[1], roi[3]
        slot_h = max(1, (y2 - y1) / 3.0)
        slot_idx = int((hit.cy - y1) // slot_h) + 1
        slot_idx = max(1, min(3, slot_idx))
        if slot_idx != self._current_char:
            return slot_idx
        return None

    def _is_ult_ready(self, frame: np.ndarray) -> bool:
        if not self.use_ult_ready:
            return False
        skill_roi = self.rois.get("skill_bar", [1420, 860, 1900, 1060])
        return self.vision.match_template(frame, "icon_ult_ready.png", roi=skill_roi, threshold=0.78) is not None

    def detect_combat(self, frame: np.ndarray, quest_need_combat: bool = False) -> bool:
        now = time.monotonic()
        center_roi = self.rois.get("combat_center", [360, 80, 1560, 860])
        skill_roi = self.rois.get("skill_bar", [1420, 860, 1900, 1060])

        hit = (
            self.vision.match_template(frame, "in_combat.png", roi=skill_roi, threshold=0.76) is not None
            or self.vision.match_template(frame, "enemy_lock.png", roi=center_roi, threshold=0.76) is not None
            or self._has_enemy_hp_bar(frame)
        )

        if hit:
            self._last_combat_ts = now
            return True

        if self._last_combat_ts > 0 and (now - self._last_combat_ts) < self.exit_grace_sec:
            return True

        if quest_need_combat and self._last_combat_ts > 0 and (now - self._last_combat_ts) < (self.exit_grace_sec * 1.5):
            return True

        self._locked_in_encounter = False
        return False

    def reset_encounter(self) -> None:
        self._locked_in_encounter = False
        self._rotation_idx = 0

    def step(self, frame: Optional[np.ndarray] = None) -> None:
        now = time.monotonic()
        if not self._locked_in_encounter:
            if self.lock_on_key in ("left", "right", "middle"):
                self.inp.click(button=self.lock_on_key)
            else:
                self.inp.press_key(self.lock_on_key)
            self._locked_in_encounter = True
            time.sleep(0.08)

        # 0. L5 电竞级毫秒拼刀弹反（Parry）
        if frame is not None and (now - self._last_parry_ts) > 1.4:
            if self._has_parry_flash(frame):
                if self.logger:
                    self.logger.info("⚔️ [L5毫秒弹刀] 捕捉到敌人金色拼刀光环，瞬间打出弹刀截击反制！")
                self.inp.click("left")
                self._last_parry_ts = now
                time.sleep(0.08)
                self.inp.press_key("e")
                return

        # 1. 突发红光高危攻击 -> 立即触发极限闪避反击（右键闪避 + 左键反击）
        if frame is not None and self.auto_dodge_counter:
            if self._has_danger_flash(frame) and (now - self._last_dodge_counter_ts) > 1.2:
                if self.logger:
                    self.logger.info("[极限闪避] 检测到敌人强力攻击前摇，触发极限闪避与反击连招！")
                self.inp.click("right")
                time.sleep(0.12)
                self.inp.click("left")
                self._last_dodge_counter_ts = now
                time.sleep(0.10)
                return

        # 2. 优先响应协奏变奏切人与大招就绪
        if frame is not None:
            ready_char = self._detect_concerto_ready_char(frame)
            if ready_char is not None:
                self._current_char = ready_char
                self.inp.press_key(str(ready_char))
                time.sleep(0.16)
                return

            if self._is_ult_ready(frame):
                self.inp.press_key("r")
                time.sleep(0.12)
                # L5 极速取消后摇：大招释放后立即切下一位角色打出协同复合输出
                next_c = (self._current_char % 3) + 1
                self.inp.press_key(str(next_c))
                self._current_char = next_c
                time.sleep(0.06)
                return

            # 3. 声骸主动技能定时释放 (Q 键)
            if self.auto_echo_skill and (now - self._last_echo_ts) > self.echo_cooldown_sec:
                if self.logger:
                    self.logger.info("[声骸爆发] 触发主声骸主动技能 (Q)")
                self.inp.press_key("q")
                self._last_echo_ts = now
                time.sleep(0.18)
                return

        if not self.rotation:
            self.inp.click("left")
            self.inp.press_key("e")
            return

        step_cfg = self.rotation[self._rotation_idx % len(self.rotation)]
        self._rotation_idx += 1

        action = step_cfg.get("action", "click")
        if action == "click":
            count = int(step_cfg.get("count", 1))
            interval = float(step_cfg.get("interval", 0.16))
            for _ in range(count):
                self.inp.click("left")
                time.sleep(interval)
        elif action == "hold_click":
            duration = float(step_cfg.get("duration", 0.6))
            self.inp.click("left", hold_sec=duration)
        elif action == "dodge":
            self.inp.click("right")
            if self.auto_dodge_counter:
                time.sleep(0.12)
                self.inp.click("left")
            time.sleep(0.10)
        elif action in ("echo", "q"):
            self.inp.press_key("q")
            self._last_echo_ts = now
            time.sleep(0.18)
        elif action == "key":
            key = str(step_cfg.get("key", "e"))
            self.inp.press_key(key)
        elif action == "switch_next":
            self._current_char = (self._current_char % 3) + 1
            self.inp.press_key(str(self._current_char))
            time.sleep(0.15)
        elif action in ("switch_1", "switch_2", "switch_3"):
            idx = int(action.split("_")[1])
            self._current_char = idx
            self.inp.press_key(str(idx))
            time.sleep(0.15)
