import logging
import threading
import time
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import keyboard

from core.ai_agent import AIAgentController
from core.auto_collector import AutoCalibrator, AutoTemplateCollector
from core.capture import ScreenCapture
from core.input_ctrl import InputController
from core.presets import PresetManager
from core.state_machine import BotState, QuestStateMachine
from core.vision import VisionEngine


class BotRunner:
    """
    后台运行控制器，供桌面 GUI 软件与 CLI 共用。
    支持启动/暂停/停止、一键自动校准、后台自动收集 UI 模板、人工接管恢复、实时热切换预设。
    """

    def __init__(
        self,
        base_dir: Path,
        config: dict,
        preset_manager: PresetManager,
        logger: logging.Logger,
        on_status_update: Optional[Callable[[dict], None]] = None,
    ) -> None:
        self.base_dir = Path(base_dir)
        self.config = config
        self.preset_manager = preset_manager
        self.logger = logger
        self.on_status_update = on_status_update

        win_cfg = config.get("window", {})
        loop_cfg = config.get("loop", {})
        vis_cfg = config.get("vision", {})
        hk_cfg = config.get("hotkeys", {})

        self.toggle_key = str(hk_cfg.get("toggle", "f8"))
        self.takeover_key = str(hk_cfg.get("takeover_done", "f7"))
        self.tick_interval = float(loop_cfg.get("tick_interval", 0.15))

        self.capture = ScreenCapture(
            title_keywords=win_cfg.get("title_keywords", ["鸣潮", "Wuthering Waves"]),
            base_width=int(win_cfg.get("base_width", 1920)),
            base_height=int(win_cfg.get("base_height", 1080)),
        )
        self.inp = InputController(
            jitter_min=float(loop_cfg.get("jitter_min", 0.05)),
            jitter_max=float(loop_cfg.get("jitter_max", 0.13)),
        )
        self.vision = VisionEngine(
            assets_dir=str(self.base_dir / "assets"),
            default_threshold=float(vis_cfg.get("default_threshold", 0.80)),
            ocr_interval_sec=float(vis_cfg.get("ocr_interval_sec", 1.0)),
        )
        self.collector = AutoTemplateCollector(
            assets_dir=self.base_dir / "assets",
            vision=self.vision,
            logger=self.logger,
            enabled=bool(vis_cfg.get("auto_collect_templates", True)),
        )
        self.calibrator = AutoCalibrator(
            capture=self.capture,
            vision=self.vision,
            collector=self.collector,
            logger=self.logger,
        )

        active_id = str(config.get("active_preset", "default_full_quest"))
        self.active_preset = self.preset_manager.get_preset(active_id)

        self.ai_agent = AIAgentController(config=config, logger=self.logger)

        self.sm = QuestStateMachine(
            config=config,
            capture=self.capture,
            inp=self.inp,
            vision=self.vision,
            logger=self.logger,
            preset=self.active_preset,
            on_state_change=self._handle_state_change,
            collector=self.collector,
            is_running=lambda: self._running,
            ai_agent=self.ai_agent,
        )

        self._lock = threading.Lock()
        self._running = False
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._hotkeys_registered = False

    def auto_calibrate(self) -> Tuple[bool, str]:
        with self._lock:
            ok, msg, rois = self.calibrator.calibrate(self.config)
            if ok and rois:
                self.sm.update_rois(rois)
        self._emit_status()
        return ok, msg

    def set_auto_collect(self, enabled: bool) -> None:
        self.collector.enabled = bool(enabled)
        self.config.setdefault("vision", {})["auto_collect_templates"] = bool(enabled)
        self.logger.info("运行中自动收集 UI 模板：%s", "已开启" if enabled else "已关闭")

    def set_ai_mode(self, enabled: bool) -> None:
        if self.ai_agent:
            self.ai_agent.set_enabled(enabled)
        self._emit_status()

    def update_ai_config(self, ai_cfg: dict) -> None:
        if self.ai_agent:
            self.ai_agent.update_config(ai_cfg)
        self._emit_status()

    def test_ai_connection(self) -> Tuple[bool, float, str]:
        if not self.ai_agent:
            return False, 0.0, "AI 模块未初始化"
        frame = self.capture.grab(normalize=True)
        return self.ai_agent.test_connection(frame)

    def fetch_ai_models(self, provider: str, base_url: str, api_key: str) -> List[str]:
        if self.ai_agent:
            return self.ai_agent.fetch_models(provider, base_url, api_key)
        from core.ai_agent import AIAgentController
        return AIAgentController.fetch_models(provider, base_url, api_key)

    def _emit_status(self) -> None:
        if self.on_status_update is None:
            return
        quest_text = ""
        if self.sm.last_quest_info is not None:
            quest_text = self.sm.last_quest_info.raw_text
        win_title = (
            f"{self.capture.window.title} ({self.capture.window.width}x{self.capture.window.height})"
            if self.capture.window is not None
            else "未绑定（请先启动鸣潮）"
        )
        tpl_count = len(list((self.base_dir / "assets").glob("*.png")))
        ai_enabled, ai_reason = (
            self.ai_agent.get_status_info() if self.ai_agent else (False, "未启用")
        )
        payload = {
            "running": self._running,
            "state": self.sm.state.value,
            "takeover_reason": self.sm.takeover_reason,
            "quest_text": quest_text,
            "window_title": win_title,
            "preset_name": self.active_preset.get("name", self.active_preset.get("id", "")),
            "template_count": tpl_count,
            "ai_enabled": ai_enabled,
            "ai_reason": ai_reason,
            "l5_summary": self.sm.get_l5_summary() if hasattr(self.sm, "get_l5_summary") else {},
        }
        try:
            self.on_status_update(payload)
        except Exception:
            pass

    def _handle_state_change(self, new_state: BotState, detail: str) -> None:
        self._emit_status()

    def register_hotkeys(self) -> None:
        if self._hotkeys_registered:
            return
        try:
            keyboard.add_hotkey(self.toggle_key, self.toggle)
            keyboard.add_hotkey(self.takeover_key, self.resume_takeover)
            self._hotkeys_registered = True
            self.logger.info(
                "全局热键已绑定：[%s] 启动/暂停 | [%s] 接管完成恢复",
                self.toggle_key.upper(),
                self.takeover_key.upper(),
            )
        except Exception as exc:
            self.logger.warning("注册全局热键失败（可能需要管理员权限）: %s", exc)

    def unregister_hotkeys(self) -> None:
        if not self._hotkeys_registered:
            return
        try:
            keyboard.unhook_all_hotkeys()
        except Exception:
            pass
        self._hotkeys_registered = False

    def apply_preset(self, preset_id: str) -> None:
        preset = self.preset_manager.get_preset(preset_id)
        with self._lock:
            self.active_preset = preset
            self.sm.apply_preset(preset)
        self._emit_status()

    def start(self) -> None:
        with self._lock:
            if self._running:
                return
            self._running = True
            self._stop_event.clear()
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._loop, name="BotRunnerThread", daemon=True)
                self._thread.start()
        self.logger.info("脚本已启动（切换到鸣潮游戏窗口后自动执行，零模板也可直接运行）。")
        self._emit_status()

    def pause(self) -> None:
        with self._lock:
            if not self._running:
                return
            self._running = False
            self.inp.release_all()
        self.logger.info("脚本已暂停。")
        self._emit_status()

    def toggle(self) -> None:
        if self._running:
            self.pause()
        else:
            self.start()

    def resume_takeover(self) -> None:
        with self._lock:
            if self.sm.state == BotState.TAKEOVER:
                self.sm.resume_from_takeover()
        self._emit_status()

    def stop(self) -> None:
        self._running = False
        self._stop_event.set()
        if self.ai_agent:
            self.ai_agent.stop()
        self.inp.release_all()
        self.unregister_hotkeys()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self.capture.close()
        self.logger.info("后台引擎已安全停止。")
        self._emit_status()

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            if not self._running:
                time.sleep(0.1)
                continue

            try:
                if self.capture.window is None:
                    if self.capture.find_window() is None:
                        self._emit_status()
                        time.sleep(1.0)
                        continue
                    self.logger.info(
                        "已绑定游戏窗口: [%s] (%dx%d)",
                        self.capture.window.title,
                        self.capture.window.width,
                        self.capture.window.height,
                    )
                    self._emit_status()

                if not self.capture.is_foreground():
                    self.inp.release_all()
                    time.sleep(0.3)
                    continue

                frame = self.capture.grab(normalize=True)
                if frame is None:
                    time.sleep(0.2)
                    continue

                if not self._running:
                    time.sleep(0.1)
                    continue

                self.sm.tick(frame)
                self._emit_status()
                time.sleep(self.tick_interval)
            except Exception as exc:
                self.logger.exception("主循环异常: %s", exc)
                self.inp.release_all()
                time.sleep(1.0)
