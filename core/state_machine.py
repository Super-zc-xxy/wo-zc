import logging
import re
import time
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from core.auto_collector import AutoTemplateCollector
from core.capture import ScreenCapture
from core.experience_memory import ExperienceMemory
from core.input_ctrl import InputController
from core.l5_planner import GoalType, L5HierarchicalPlanner, SubGoal
from core.l5_world_model import L5WorldModel
from core.logger import notify_takeover, save_debug_screenshot
from core.vision import VisionEngine
from tasks.combat import CombatHandler
from tasks.dialogue import DialogueHandler
from tasks.gadget_manager import GadgetManager
from tasks.map_teleport import MapTeleporter
from tasks.navigator import Navigator
from tasks.popup_handler import PopupHandler
from tasks.quest_menu import QuestMenuHandler
from tasks.quest_tracker import QuestInfo, QuestTracker
from tasks.time_adjuster import TimeAdjuster


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
    TIME_SKIP = "TIME_SKIP"
    SWITCH_QUEST = "SWITCH_QUEST"
    NAVIGATE = "NAVIGATE"
    TAKEOVER = "TAKEOVER"
    AI_AUTOPILOT = "AI_AUTOPILOT"


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
        is_running: Optional[Callable[[], bool]] = None,
        ai_agent: Optional[Any] = None,
    ) -> None:
        self.config = config
        self.capture = capture
        self.inp = inp
        self.vision = vision
        self.logger = logger
        self.on_state_change = on_state_change
        self.collector = collector
        self.is_running = is_running
        self.ai_agent = ai_agent

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
        self.combat = CombatHandler(vision, inp, config, logger=logger)
        self.navigator = Navigator(vision, inp, config, collector=collector, logger=logger)
        self.teleporter = MapTeleporter(vision, capture, inp, config, logger, collector=collector)
        self.quest_menu = QuestMenuHandler(vision, capture, inp, config, logger, collector=collector)
        self.popup_handler = PopupHandler(vision, capture, inp, config, collector=collector)
        self.time_adjuster = TimeAdjuster(vision, capture, inp, config, logger, collector=collector)
        self.gadget_manager = GadgetManager(inp, capture, vision, logger=logger)
        self.experience_memory = ExperienceMemory(logger=logger)
        self.l5_planner = L5HierarchicalPlanner(logger=logger)
        self.world_model = L5WorldModel(logger=logger)

        self.features = {
            "auto_dialogue": True,
            "auto_combat": True,
            "auto_navigate": True,
            "auto_teleport": True,
            "auto_switch_quest": True,
            "auto_time_adjust": True,
            "auto_absorb_echo": True,
            "auto_popup_close": True,
            "auto_gadget": True,
            "auto_heal": True,
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
        self._last_heal_ts: float = 0.0
        self.blocked_quests: Dict[str, float] = {}

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
            self.time_adjuster,
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
        self.blocked_quests.clear()
        self._set_state(BotState.IDLE)
        self.logger.info("人工接管完成，已清空受阻黑名单并恢复自动执行。")

    def _switch_away_from_current_quest(self, reason: str, frame: Optional[np.ndarray] = None) -> bool:
        """
        当当前任务因解谜、无法调整时间、卡死、悬崖阻挡或距离过远无法推进时，
        将该任务加入冷却黑名单，自动按 J 打开任务面板切换至下一个可用任务。
        """
        now = time.monotonic()
        self.blocked_quests = {k: exp for k, exp in self.blocked_quests.items() if now < exp}

        curr_text = ""
        if self.last_quest_info and self.last_quest_info.raw_text:
            curr_text = self.last_quest_info.raw_text

        if curr_text:
            clean_name = re.sub(r"\d+\s*[mM米]", "", curr_text).strip()
            if clean_name:
                self.blocked_quests[clean_name] = now + 900.0  # 冷却 15 分钟

        if not self.features.get("auto_switch_quest", True):
            self.logger.warning(f"当前任务受阻（{reason}），但自动切换任务开关未开启。")
            return False

        self.logger.warning(f"当前任务受阻（{reason}），正在自动打开任务列表切换下一个任务...")
        self.inp.release_all()
        self._set_state(BotState.SWITCH_QUEST, reason)

        active_blocked = list(self.blocked_quests.keys())
        switched = self.quest_menu.switch_to_next_quest(
            priority_keywords=self.quest_priority_keywords,
            exclude_texts=active_blocked,
        )

        if switched:
            self.logger.info("已成功切换到下一个任务，重置寻路与脱困状态。")
            self.unstuck_count = 0
            self.last_motion_ts = time.monotonic()
            self.state_enter_ts = time.monotonic()
            self.no_quest_since = None
            self._set_state(BotState.IDLE)
            return True

        self.logger.warning("未能切换到其他可用任务。")
        return False

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
        if self.is_running is not None and not self.is_running():
            return self.state

        if self.state == BotState.TAKEOVER:
            return self.state

        now = time.monotonic()

        # 0. L5 世界模型时序微观物理观测录入
        try:
            dist = getattr(self.last_quest_info, "distance_m", None)
            self.world_model.update_observation(frame, distance_m=dist, action=self.state.value)
        except Exception:
            pass

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
                if "revive" in handled.lower() or "复苏" in handled:
                    self.logger.info("检测到团灭复苏已触发，重置脱困与寻路状态，准备信标重生续做...")
                    self.unstuck_count = 0
                    self.last_motion_ts = now
                return self.state

        # 2.2 濒血自动使用回复药剂 [C]
        if self.features.get("auto_heal", True) and (now - self._last_heal_ts) > 8.0:
            hp_ratio = None
            if hasattr(self.vision, "get_player_hp_ratio"):
                try:
                    hp_ratio = self.vision.get_player_hp_ratio(frame)
                except Exception:
                    pass
            if isinstance(hp_ratio, (int, float)) and hp_ratio < 0.30:
                self.inp.press_key("c")
                self._last_heal_ts = now
                self.logger.warning(f"[生存守护] 角色生命值过低 ({int(hp_ratio * 100)}%)，自动按 C 使用回复药剂！")
                if hp_ratio < 0.18:
                    self.l5_planner.trigger_interrupt(
                        title="紧急界碑神像满血补给",
                        reason="生命值跌破18%高危线，触发高优先级自保插队",
                        priority=0,
                        action_params={"action": "heal_at_nexus"},
                        context_snapshot={"last_quest": getattr(self.last_quest_info, "raw_text", "")},
                    )

        # 2.5 AI 全权接管托管模式（双速超低延迟流水线）
        if self.ai_agent and getattr(self.ai_agent, "enabled", False):
            quest_info = self.tracker.read(frame)
            self.last_quest_info = quest_info
            context = {
                "quest_text": quest_info.raw_text if quest_info else "",
                "distance_m": quest_info.distance_m if quest_info else None,
            }
            # 触发 AI 异步观察推理（后台网络请求，主线程 0ms 阻塞）
            self.ai_agent.trigger_ai_observation(frame, context)

            # 若 AI 产生了高层决策动作，即刻平滑执行
            act_data = self.ai_agent.execute_action(self.inp)
            if act_data:
                self.last_motion_ts = now
                act_name = act_data.get("act", "")
                reason = act_data.get("reason", "")
                self._set_state(BotState.AI_AUTOPILOT, f"{act_name}: {reason}")
                if act_name == "switch_quest":
                    self._switch_away_from_current_quest(f"AI决策跳过: {reason}", frame)
                return self.state

            # AI 思考间隙（网络往返等待期）：本地反射层高速惯性巡航/自保，杜绝原地呆滞
            self._set_state(BotState.AI_AUTOPILOT, getattr(self.ai_agent, "_latest_reason", "AI 巡航中"))
            if self.combat.detect_combat(frame):
                self.combat.step(frame)
            elif self.navigator.detect_interact(frame):
                self.navigator.interact()
            elif self.navigator.detect_gadget(frame):
                self.navigator.trigger_gadget()
            elif self.features.get("auto_navigate", True):
                self.navigator.step_towards_marker(frame)
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
            # 1. 优先查询本地离线经验记忆库
            exp = self.experience_memory.query_experience(quest_info.raw_text)
            if exp:
                act = exp.get("action", "")
                reason = exp.get("reason", "")
                self.logger.info(f"💡 [经验记忆回放] 命中历史策略: {reason} | 动作: {act}")
                if act == "switch_quest":
                    if self._switch_away_from_current_quest(f"经验策略跳过: {reason}", frame):
                        return self.state
                elif act.startswith("gadget:"):
                    target_tool = act.split(":")[1]
                    self.gadget_manager.switch_tool(target_tool)
                    self.gadget_manager.trigger_active_gadget()
                    self.last_motion_ts = now
                    return self.state

            # 2. 调用 L5 世界模型物理机制因果推演
            macro_actions = self.world_model.reason_mechanism_causality(quest_info.raw_text)
            if macro_actions:
                self.logger.info(f"🌐 [L5物理因果破局] 成功推演出因果宏动作序列 ({len(macro_actions)}步)，自主执行解谜...")
                self._set_state(BotState.GADGET, "正在执行L5物理因果宏动作解谜")
                ok = self.world_model.execute_macro_sequence(macro_actions, self.inp, self.gadget_manager)
                self.last_motion_ts = now
                if ok:
                    self.experience_memory.record_experience(
                        quest_info.raw_text,
                        {"action": "macro_executed", "reason": "世界模型物理因果推演破局", "status": "success"},
                    )
                    self.tracker.read(frame, force=True)
                    return self.state

            reason = f"检测到特殊解谜/玩法任务：{quest_info.raw_text}"
            if self._switch_away_from_current_quest(reason, frame):
                return self.state
            self.trigger_takeover(reason, frame)
            return self.state

        if quest_info.need_time_adjust:
            if self.features.get("auto_time_adjust", True):
                self._set_state(BotState.TIME_SKIP, "正在自动调节游戏内时间")
                ok = self.time_adjuster.adjust_time()
                self.last_motion_ts = now
                if ok:
                    # 调整成功后强制失效 OCR 缓存，以便下次循环识别最新任务文本
                    self.tracker.read(frame, force=True)
                    self._set_state(BotState.IDLE)
                    return self.state
            reason = f"任务需要调整游戏内时间：{quest_info.raw_text}"
            if self._switch_away_from_current_quest(reason, frame):
                return self.state
            self.trigger_takeover(reason, frame)
            return self.state

        # 5. 战斗状态
        if self.features.get("auto_combat", True):
            if self.combat.detect_combat(frame, quest_need_combat=quest_info.need_combat):
                self._set_state(BotState.COMBAT)
                self.combat.step(frame)
                self.last_motion_ts = now
                return self.state

        # 6. 战后吸收声骸与拾取
        if self.features.get("auto_absorb_echo", True) and now < self._post_combat_echo_until:
            self._set_state(BotState.ABSORB_ECHO, "战后声骸与战利品定向寻径吸附")
            absorbed = self.navigator.patrol_and_absorb_echo(frame)
            self.last_motion_ts = now
            if absorbed:
                # 触碰到掉落物后略微缩短后续轮询窗口
                self._post_combat_echo_until = min(self._post_combat_echo_until, now + 1.5)
            return self.state

        # 7. 可交互提示 [F]
        if self.navigator.detect_interact(frame):
            self.inp.release_all()
            self._set_state(BotState.INTERACT)
            self.navigator.interact()
            self.last_motion_ts = now
            return self.state

        # 8. 钩索/探索工具提示 [T] 与自适应切换
        if self.features.get("auto_gadget", True):
            if quest_info and quest_info.raw_text:
                rec_tool = self.gadget_manager.recommend_tool_for_quest(quest_info.raw_text)
                if rec_tool and rec_tool != self.gadget_manager.current_tool:
                    self.gadget_manager.switch_tool(rec_tool)

            if self.navigator.detect_gadget(frame):
                self.inp.release_all()
                self._set_state(BotState.GADGET)
                self.gadget_manager.trigger_active_gadget()
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
        if isinstance(quest_info.distance_m, (int, float)) and quest_info.distance_m > self.navigator.far_distance_m:
            if self.features.get("auto_teleport", True) and (now - self._last_teleport_ts) > 15.0:
                self._set_state(BotState.TELEPORT)
                self._last_teleport_ts = now
                ok = self.teleporter.teleport_to_quest()
                self.last_motion_ts = time.monotonic()
                if ok:
                    return self.state
            reason = f"任务目标距离较远（{quest_info.distance_m}m）且无法自动传送"
            if self._switch_away_from_current_quest(reason, frame):
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

            # 疑难卡死 AI 单帧会诊破局机制：当连续常规脱困 2 次仍未解除停滞，且接入了 AI 大模型
            if self.unstuck_count >= 2 and self.ai_agent and (getattr(self.ai_agent, "api_key", "") or getattr(self.ai_agent, "provider", "") == "ollama"):
                self.logger.info("常规脱困受阻，正在唤醒 AI 多模态大模型进行现场单帧疑难会诊...")
                try:
                    consult_ctx = {
                        "quest_text": quest_info.raw_text if quest_info else "",
                        "unstuck_count": self.unstuck_count,
                    }
                    ai_solution = self.ai_agent.consult_stuck_situation(frame, consult_ctx)
                    if ai_solution and isinstance(ai_solution, dict):
                        act = ai_solution.get("act", "")
                        reason = ai_solution.get("reason", "")
                        self.logger.info(f"💡 [AI 会诊方案] 现场诊断: {reason} | 建议破局动作: {act}")
                        if quest_info and quest_info.raw_text:
                            self.experience_memory.record_experience(quest_info.raw_text, ai_solution)
                        if act == "switch_quest":
                            if self._switch_away_from_current_quest(f"AI会诊建议跳过: {reason}", frame):
                                return self.state
                        elif act == "gadget":
                            self.navigator.trigger_gadget()
                            self.unstuck_count = 0
                            self.last_motion_ts = time.monotonic()
                            return self.state
                        elif act == "jump":
                            self.inp.key_down("w")
                            time.sleep(0.15)
                            self.inp.press_key("space")
                            time.sleep(0.4)
                            self.inp.key_up("w")
                            self.unstuck_count = 0
                            self.last_motion_ts = time.monotonic()
                            return self.state
                        elif act == "turn":
                            turn_deg = int(ai_solution.get("turn", 90))
                            self.inp.smooth_turn(int(turn_deg * 3.5), steps=6)
                            self.inp.key_down("w")
                            time.sleep(0.4)
                            self.inp.key_up("w")
                            self.unstuck_count = 0
                            self.last_motion_ts = time.monotonic()
                            return self.state
                except Exception as exc:
                    self.logger.warning(f"AI 疑难会诊处理异常: {exc}")

            if self.unstuck_count > self.max_unstuck_retries:
                if self.features.get("auto_teleport", True) and (now - self._last_teleport_ts) > 20.0:
                    self._last_teleport_ts = now
                    if self.teleporter.teleport_to_quest():
                        self.unstuck_count = 0
                        self.last_motion_ts = time.monotonic()
                        return self.state
                reason = "连续多次寻路脱困失败，疑似遇到悬崖/建筑阻挡或解谜"
                if self._switch_away_from_current_quest(reason, frame):
                    return self.state
                self.trigger_takeover(reason, frame)
                return self.state
            self.navigator.unstuck(self.unstuck_count)
            self.last_motion_ts = time.monotonic()
            return self.state

        if (now - self.state_enter_ts) > self.state_timeout_sec:
            reason = f"状态 {self.state.value} 持续超过 {int(self.state_timeout_sec)} 秒无明显进展"
            if self._switch_away_from_current_quest(reason, frame):
                return self.state
            self.trigger_takeover(
                reason,
                frame,
            )
            return self.state

        self.navigator.step_towards_marker(frame)
        return self.state

    def set_l5_goal(self, intent_text: str) -> None:
        """设置 L5 级全局大目标意图（如：清日常与体力、探索乘霄山、拉满今汐）。"""
        self.l5_planner.set_master_goal(intent_text)

    def get_l5_summary(self) -> Dict[str, Any]:
        """获取 L5 规划器与思考流概要。"""
        return self.l5_planner.get_plan_summary()
