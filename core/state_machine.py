import logging
import time
from enum import Enum
from typing import Callable, Dict, List, Optional

import numpy as np

from core.auto_collector import AutoTemplateCollector
from core.capture import ScreenCapture
from core.input_ctrl import InputController
from core.logger import notify_takeover, save_debug_screenshot
from core.vision import VisionEngine
from tasks.combat import CombatHandler
from tasks.dialogue import DialogueHandler
from tasks.map_teleport import MapTeleporter
from tasks.navigator import Navigator
from tasks.popup_handler import PopupHandler
from tasks.quest_menu import QuestMenuHandler
from tasks.quest_tracker import QuestInfo, QuestTracker


class BotState(str, Enum):
    IDLE = "IDLE"
    LOADING = "LOADING"
    POPUP = "POPUP"
    DIALOGUE = "DIALOGUE"
    COMBAT = "COMBAT"
    ABSORB_ECHO = "ABSORB_ECHO"
    INTERACT = "INTERACT"
    GADGET = "GADGET"
    TELEPORT = "TELEPORT"
    SWITCH_QUEST = "SWITCH_QUEST"
    NAVIGATE = "NAVIGATE"
    TAKEOVER = "TAKEOVER"


class QuestStateMachine:
    def __init__(
        self,
        config: dict,
        capture: ScreenCapture,
        inp: InputController,
        vision: VisionEngine,
        logger: logging.Logger,
        preset: Optional[dict] = None,
        on_state_change: Optional[Callable[[BotState, str], None]] = None,
        collector: Optional[AutoTemplateCollector] = None,
    ) -> None:
        self.config = config
        self.capture = capture
        self.inp = inp
        self.vision = vision
        self.logger = logger
        self.on_state_change = on_state_change
        self.collector = collector

        loop_cfg = config.get("loop", {})
        log_cfg = config.get("logging", {})
        self.stuck_timeout_sec = float(loop_cfg.get("stuck_timeout_sec", 4.0))
        self.max_unstuck_retries = int(loop_cfg.get("max_unstuck_retries", 3))
        self.state_timeout_sec = float(loop_cfg.get("state_timeout_sec", 45.0))
        self.loading_dark_threshold = float(loop_cfg.get("loading_dark_threshold", 18))
        self.no_quest_switch_sec = float(loop_cfg.get("no_quest_switch_sec", 8.0))
        self.log_save_dir = str(log_cfg.get("save_dir", "logs"))
        self.save_error_screenshot = bool(log_cfg.get("save_error_screenshot", True))

        self.tracker = QuestTracker(vision, config)
        self.dialogue = DialogueHandler(vision, capture, inp, config, collector=collector)
        self.combat = CombatHandler(vision, inp, config)
        self.navigator = Navigator(vision, inp, config, collector=collector)
        self.teleporter = MapTeleporter(vision, capture, inp, config, logger, collector=collector)
        self.quest_menu = QuestMenuHandler(vision, capture, inp, config, logger, collector=collector)
        self.popup_handler = PopupHandler(vision, capture, inp, config, collector=collector)

        self.features = {
            "auto_dialogue": True,
            "auto_combat": True,
            "auto_navigate": True,
            "auto_teleport": True,
            "auto_switch_quest": True,
            "auto_absorb_echo": True,
            "auto_popup_close": True,
            "auto_gadget": True,
        }
        self.quest_priority_keywords = ["潮汐任务", "伴星任务", "危行任务", "纪闻任务", "每日"]

        self.state: BotState = BotState.IDLE
        self.state_enter_ts: float = time.monotonic()
        self.last_frame: Optional[np.ndarray] = None
        self.last_motion_ts: float = time.monotonic()
        self.no_quest_since: Optional[float] = None
        self.unstuck_count: int = 0
        self.takeover_reason: str = ""
        self.last_quest_info: Optional[QuestInfo] = None
        self._post_combat_echo_until: float = 0.0
        self._last_teleport_ts: float = 0.0

        if preset:
            self.apply_preset(preset)

    def update_rois(self, rois: Dict[str, List[int]]) -> None:
        self.config["rois"] = rois
        for module in (
            self.tracker,
            self.dialogue,
            self.combat,
            self.navigator,
            self.teleporter,
            self.quest_menu,
            self.popup_handler,
        ):
            module.rois = rois

    def apply_preset(self, preset: dict) -> None:
        if not preset:
            return
        feats = preset.get("features", {})
        for k in list(self.features.keys()):
            if k in feats:
                self.features[k] = bool(feats[k])

        self.combat.apply_preset(preset.get("combat", {}))

        nav_cfg = preset.get("navigation", {})
        if "sprint_hold_sec" in nav_cfg:
            self.navigator.sprint_hold_sec = float(nav_cfg["sprint_hold_sec"])
        if "turn_sensitivity" in nav_cfg:
            self.navigator.turn_sensitivity = float(nav_cfg["turn_sensitivity"])
        if "center_deadzone_px" in nav_cfg:
            self.navigator.center_deadzone_px = int(nav_cfg["center_deadzone_px"])
        if "far_distance_m" in nav_cfg:
            self.navigator.far_distance_m = int(nav_cfg["far_distance_m"])

        quest_cfg = preset.get("quest", {})
        if "priority_keywords" in quest_cfg and isinstance(quest_cfg["priority_keywords"], list):
            self.quest_priority_keywords = list(quest_cfg["priority_keywords"])

        self.logger.info("已应用操作预设: %s", preset.get("name", preset.get("id", "unknown")))

    def _set_state(self, new_state: BotState, detail: str = "") -> None:
        if new_state != self.state:
            self.logger.info(f"状态切换: {self.state.value} -> {new_state.value}")
            if self.state == BotState.COMBAT and new_state != BotState.COMBAT:
                self.combat.reset_encounter()
                if self.features.get("auto_absorb_echo", True):
                    self._post_combat_echo_until = time.monotonic() + 4.0
            self.state = new_state
            self.state_enter_ts = time.monotonic()
            if new_state != BotState.NAVIGATE:
                self.unstuck_count = 0
            if self.on_state_change is not None:
                try:
                    self.on_state_change(new_state, detail)
                except Exception:
                    pass

    def trigger_takeover(self, reason: str, frame: Optional[np.ndarray] = None) -> None:
        self.inp.release_all()
        self.takeover_reason = reason
        self._set_state(BotState.TAKEOVER, reason)
        if self.save_error_screenshot and frame is not None:
            saved = save_debug_screenshot(frame, self.log_save_dir, prefix="takeover")
            if saved:
                self.logger.info(f"已保存接管现场截图: {saved}")
        notify_takeover(reason, self.logger)

    def resume_from_takeover(self) -> None:
        self.takeover_reason = ""
        self.unstuck_count = 0
        self.last_motion_ts = time.monotonic()
        self._set_state(BotState.IDLE)
        self.logger.info("人工接管完成，恢复自动执行。")

    def _update_motion_watch(self, frame: np.ndarray) -> bool:
        now = time.monotonic()
        if self.last_frame is not None:
            diff = self.capture.frame_diff_score(self.last_frame, frame)
            if diff > 2.2:
                self.last_motion_ts = now
                self.unstuck_count = 0
        self.last_frame = frame
        return (now - self.last_motion_ts) >= self.stuck_timeout_sec

    def tick(self, frame: np.ndarray) -> BotState:
        if self.state == BotState.TAKEOVER:
            return self.state

        now = time.monotonic()

        # 1. 加载黑屏检测
        if self.capture.mean_brightness(frame) < self.loading_dark_threshold:
            self.inp.release_all()
            self._set_state(BotState.LOADING)
            self.last_motion_ts = now
            return self.state

        # 2. 非任务弹窗（复活/教学/交付/关闭按钮）
        if self.features.get("auto_popup_close", True):
            handled = self.popup_handler.handle(frame)
            if handled is not None:
                self._set_state(BotState.POPUP, handled)
                self.last_motion_ts = now
                return self.state

        # 3. 剧情与对话
        if self.features.get("auto_dialogue", True):
            if self.dialogue.is_in_dialogue(frame) or self.dialogue.detect_confirm_button(frame) is not None:
                self.inp.release_all()
                self._set_state(BotState.DIALOGUE)
                self.dialogue.step(frame)
                self.last_motion_ts = now
                return self.state

        # 4. 读取左上角任务栏（带节流）
        quest_info = self.tracker.read(frame)
        self.last_quest_info = quest_info

        if quest_info.raw_text:
            self.no_quest_since = None
        else:
            if self.no_quest_since is None:
                self.no_quest_since = now

        if quest_info.need_takeover:
            self.trigger_takeover(f"检测到特殊解谜/玩法任务：{quest_info.raw_text}", frame)
            return self.state

        if quest_info.need_time_adjust:
            self.trigger_takeover(f"任务需要调整游戏内时间：{quest_info.raw_text}", frame)
            return self.state

        # 5. 战斗状态
        if self.features.get("auto_combat", True):
            if self.combat.detect_combat(frame, quest_need_combat=quest_info.need_combat):
                self._set_state(BotState.COMBAT)
                self.combat.step(frame)
                self.last_motion_ts = now
                return self.state

        # 6. 战后吸收声骸
        if self.features.get("auto_absorb_echo", True) and now < self._post_combat_echo_until:
            if self.navigator.detect_interact(frame):
                self._set_state(BotState.ABSORB_ECHO)
                self.navigator.interact()
                self.last_motion_ts = now
                return self.state

        # 7. 可交互提示 [F]
        if self.navigator.detect_interact(frame):
            self.inp.release_all()
            self._set_state(BotState.INTERACT)
            self.navigator.interact()
            self.last_motion_ts = now
            return self.state

        # 8. 钩索/探索工具提示 [T]
        if self.features.get("auto_gadget", True) and self.navigator.detect_gadget(frame):
            self.inp.release_all()
            self._set_state(BotState.GADGET)
            self.navigator.trigger_gadget()
            self.last_motion_ts = now
            return self.state

        # 9. 左上角长时间无追踪任务 -> 自动打开任务面板切换下一个任务
        if (
            self.features.get("auto_switch_quest", True)
            and self.no_quest_since is not None
            and (now - self.no_quest_since) >= self.no_quest_switch_sec
        ):
            self._set_state(BotState.SWITCH_QUEST)
            self.quest_menu.switch_to_next_quest(self.quest_priority_keywords)
            self.no_quest_since = time.monotonic()
            self.last_motion_ts = time.monotonic()
            return self.state

        # 10. 远距离任务 -> 大地图自动传送
        if quest_info.distance_m is not None and quest_info.distance_m > self.navigator.far_distance_m:
            if self.features.get("auto_teleport", True) and (now - self._last_teleport_ts) > 15.0:
                self._set_state(BotState.TELEPORT)
                self._last_teleport_ts = now
                ok = self.teleporter.teleport_to_quest()
                self.last_motion_ts = time.monotonic()
                if ok:
                    return self.state
            self.trigger_takeover(
                f"任务目标距离较远（{quest_info.distance_m}m），请先传送到附近信标。",
                frame,
            )
            return self.state

        # 11. 默认寻路推进 + 卡死脱困
        if not self.features.get("auto_navigate", True):
            self._set_state(BotState.IDLE)
            return self.state

        self._set_state(BotState.NAVIGATE)
        is_stuck = self._update_motion_watch(frame)
        if is_stuck:
            self.unstuck_count += 1
            self.logger.warning(f"疑似卡死，尝试第 {self.unstuck_count}/{self.max_unstuck_retries} 次脱困...")
            if self.unstuck_count > self.max_unstuck_retries:
                if self.features.get("auto_teleport", True) and (now - self._last_teleport_ts) > 20.0:
                    self._last_teleport_ts = now
                    if self.teleporter.teleport_to_quest():
                        self.unstuck_count = 0
                        self.last_motion_ts = time.monotonic()
                        return self.state
                self.trigger_takeover("连续多次寻路脱困失败，疑似遇到悬崖/建筑阻挡或解谜。", frame)
                return self.state
            self.navigator.unstuck(self.unstuck_count)
            self.last_motion_ts = time.monotonic()
            return self.state

        if (now - self.state_enter_ts) > self.state_timeout_sec:
            self.trigger_takeover(
                f"状态 {self.state.value} 持续超过 {int(self.state_timeout_sec)} 秒无明显进展。",
                frame,
            )
            return self.state

        self.navigator.step_towards_marker(frame)
        return self.state
