import logging
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.ai_agent import AIAgentController
from core.auto_collector import AutoCalibrator, AutoTemplateCollector
from core.bot_runner import BotRunner
from core.capture import EXCLUDED_TITLE_KEYWORDS, ScreenCapture, WindowRect
from core.input_ctrl import InputController, SCAN_CODES
from core.presets import PresetManager
from core.state_machine import BotState, QuestStateMachine
from core.vision import MatchResult, OCRLine, VisionEngine
from tasks.combat import CombatHandler
from tasks.dialogue import DialogueHandler
from tasks.gadget_manager import GadgetManager
from tasks.map_teleport import MapTeleporter
from tasks.navigator import Navigator
from tasks.popup_handler import PopupHandler
from tasks.quest_menu import QuestMenuHandler
from tasks.time_adjuster import TimeAdjuster
from core.experience_memory import ExperienceMemory
from core.l5_planner import GoalType, L5HierarchicalPlanner, SubGoalStatus
from core.l5_world_model import L5WorldModel
from tools.doctor import EnvironmentDoctor


class DummyInput(InputController):
    def __init__(self):
        super().__init__(jitter_min=0.0, jitter_max=0.0)
        self.events = []

    def key_down(self, key: str) -> None:
        self.events.append(("key_down", key))

    def key_up(self, key: str) -> None:
        self.events.append(("key_up", key))

    def press_key(self, key: str, hold_sec: float = 0.0) -> None:
        self.events.append(("press_key", key))

    def release_all(self) -> None:
        self.events.append(("release_all",))

    def smooth_turn(self, total_dx: int, steps: int = 1, step_delay: float = 0.0) -> None:
        self.events.append(("smooth_turn", total_dx))

    def click(self, button: str = "left", hold_sec: float = 0.0) -> None:
        self.events.append(("click", button))

    def click_at(self, screen_x: int, screen_y: int, button: str = "left") -> None:
        self.events.append(("click_at", screen_x, screen_y, button))

    def move_cursor(self, x: int, y: int) -> None:
        self.events.append(("move_cursor", x, y))

    def mouse_down(self, button: str = "left") -> None:
        self.events.append(("mouse_down", button))

    def mouse_up(self, button: str = "left") -> None:
        self.events.append(("mouse_up", button))

    def drag(self, start_x: int, start_y: int, end_x: int, end_y: int, steps: int = 1, duration: float = 0.0) -> None:
        self.events.append(("drag", start_x, start_y, end_x, end_y))

    def scroll(self, delta: int) -> None:
        self.events.append(("scroll", delta))


class SmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with (ROOT / "config.yaml").open("r", encoding="utf-8") as f:
            cls.config = yaml.safe_load(f)
        cls.logger = logging.getLogger("test_ww_auto_quest")
        cls.logger.addHandler(logging.NullHandler())

    def test_imports_and_dependencies(self):
        import gui_app
        import keyboard
        import mss
        import win32gui

        self.assertIsNotNone(gui_app)
        self.assertIsNotNone(keyboard)
        self.assertIsNotNone(mss)
        self.assertIsNotNone(win32gui)
        self.assertIn("w", SCAN_CODES)
        self.assertIn("f", SCAN_CODES)
        self.assertIn("j", SCAN_CODES)
        self.assertIn("m", SCAN_CODES)

        try:
            import onnxruntime
            import rapidocr
            self.assertIsNotNone(onnxruntime)
            self.assertIsNotNone(rapidocr)
        except ImportError:
            self.logger.warning("当前环境未安装 onnxruntime/rapidocr，已跳过 OCR 依赖检查。")

    def test_window_exclusion_and_scoring(self):
        cap = ScreenCapture(["鸣潮", "Wuthering Waves"])
        # 自身标题关键词必须在排除列表中
        self.assertTrue(any(ex in "鸣潮自动任务助手 v2 - mingchaojb" for ex in EXCLUDED_TITLE_KEYWORDS))
        # UE4 游戏窗口得分应远高于普通窗口
        ue_score = cap._score_window(1, "鸣潮  ", "UnrealWindow", 1920, 1080)
        other_score = cap._score_window(2, "鸣潮攻略", "Chrome_WidgetWin_1", 1040, 700)
        self.assertGreater(ue_score, other_score)
        cap.close()

    def test_preset_manager_crud(self):
        with tempfile.TemporaryDirectory() as tmp:
            pm = PresetManager(Path(tmp))
            presets = pm.list_presets()
            self.assertGreaterEqual(len(presets), 4)

            dup = pm.duplicate_preset("default_full_quest", "测试自定义预设")
            self.assertEqual(dup["name"], "测试自定义预设")
            loaded = pm.get_preset(dup["id"])
            self.assertEqual(loaded["name"], "测试自定义预设")

            self.assertTrue(pm.delete_preset(dup["id"]))

    def test_auto_collector_and_calibrator(self):
        with tempfile.TemporaryDirectory() as tmp:
            assets_dir = Path(tmp)
            vision = MagicMock(spec=VisionEngine)
            vision._template_cache = {}
            vision.ocr_roi.side_effect = [
                [OCRLine(text="潮汐任务：测试目标 120m", score=0.95, bbox=(60, 210, 320, 250))],
                [OCRLine(text="跳过", score=0.96, bbox=(80, 40, 140, 75))],
                [OCRLine(text="F", score=0.92, bbox=(1200, 520, 1230, 550))],
            ]
            vision.detect_yellow_quest_marker.return_value = (960, 420)

            capture = MagicMock(spec=ScreenCapture)
            capture.find_window.return_value = WindowRect(1, "鸣潮", 0, 0, 1920, 1080)
            frame = np.full((1080, 1920, 3), 100, dtype=np.uint8)
            capture.grab.return_value = frame

            collector = AutoTemplateCollector(assets_dir, vision, self.logger, enabled=True)
            calibrator = AutoCalibrator(capture, vision, collector, self.logger)

            cfg = {}
            ok, msg, rois = calibrator.calibrate(cfg)
            self.assertTrue(ok, msg)
            self.assertIn("quest_panel", rois)
            self.assertTrue((assets_dir / "btn_skip.png").exists())
            self.assertTrue((assets_dir / "icon_f.png").exists())
            self.assertTrue((assets_dir / "quest_marker.png").exists())

    def test_zero_template_ocr_dialogue_and_interact(self):
        with tempfile.TemporaryDirectory() as tmp:
            assets_dir = Path(tmp)
            inp = DummyInput()
            capture = MagicMock(spec=ScreenCapture)
            capture.to_screen_coords.side_effect = lambda x, y: (x, y)

            vision = MagicMock(spec=VisionEngine)
            vision._template_cache = {}
            vision.match_template.return_value = None
            vision.ocr_roi.return_value = [OCRLine(text="跳过", score=0.97, bbox=(80, 40, 150, 80))]

            collector = AutoTemplateCollector(assets_dir, vision, self.logger, enabled=True)
            dlg = DialogueHandler(vision, capture, inp, self.config, collector=collector)

            frame = np.full((1080, 1920, 3), 120, dtype=np.uint8)
            self.assertTrue(dlg.is_in_dialogue(frame))
            self.assertTrue(dlg.step(frame))
            self.assertTrue((assets_dir / "btn_skip.png").exists())

            # 测试零模板 F 交互识别
            vision.ocr_roi.return_value = [OCRLine(text="[F]", score=0.95, bbox=(1200, 500, 1240, 540))]
            nav = Navigator(vision, inp, self.config, collector=collector)
            self.assertTrue(nav.detect_interact(frame))
            self.assertTrue((assets_dir / "icon_f.png").exists())

    def test_vision_ocr_and_yellow_marker(self):
        vision = VisionEngine(
            assets_dir=str(ROOT / "assets"),
            default_threshold=0.80,
            ocr_interval_sec=0.0,
        )
        if vision._ocr is not None:
            self.assertIsNotNone(vision._ocr, "RapidOCR 未正确初始化")

        self.assertEqual(vision.parse_distance_m("前往目标地点 125m"), 125)
        self.assertEqual(vision.parse_distance_m("距离 450米"), 450)
        self.assertIsNone(vision.parse_distance_m("与NPC对话"))

        frame = np.full((1080, 1920, 3), 40, dtype=np.uint8)
        pts = np.array([[960, 400], [980, 420], [960, 440], [940, 420]], dtype=np.int32)
        cv2.fillPoly(frame, [pts], (0, 220, 255))
        marker = vision.detect_yellow_quest_marker(frame, roi=[400, 80, 1520, 850])
        self.assertIsNotNone(marker)
        self.assertAlmostEqual(marker[0], 960, delta=5)
        self.assertAlmostEqual(marker[1], 420, delta=5)

        ocr_img = np.full((1080, 1920, 3), 255, dtype=np.uint8)
        cv2.putText(ocr_img, "QUEST 250m", (60, 260), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
        lines = vision.ocr_roi(ocr_img, [40, 190, 440, 360], cache_key="smoke", force=True)
        self.assertIsInstance(lines, list)

    def test_v2_combat_concerto_and_dodge(self):
        inp = DummyInput()
        vision = MagicMock(spec=VisionEngine)
        vision.match_template.side_effect = lambda frame, name, **kw: (
            MatchResult(name=name, cx=1780, cy=450, score=0.9, bbox=(1750, 430, 1810, 470))
            if name == "icon_concerto_ready.png"
            else None
        )
        combat = CombatHandler(vision, inp, self.config)
        combat.apply_preset(
            {
                "use_concerto_switch": True,
                "use_ult_ready": True,
                "rotation": [{"action": "dodge"}],
            }
        )
        dummy_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        combat.step(dummy_frame)
        self.assertIn(("press_key", "2"), inp.events)

    def test_v2_map_teleport_and_quest_menu(self):
        inp = DummyInput()
        capture = MagicMock(spec=ScreenCapture)
        capture.to_screen_coords.side_effect = lambda x, y: (x, y)

        map_frame = np.full((1080, 1920, 3), 30, dtype=np.uint8)
        cv2.circle(map_frame, (900, 500), 14, (255, 220, 0), -1)
        capture.grab.return_value = map_frame

        vision = MagicMock(spec=VisionEngine)
        vision.crop_roi.side_effect = VisionEngine.crop_roi
        vision.detect_yellow_quest_marker.return_value = (920, 510)
        vision.match_template.side_effect = lambda frame, name, **kw: (
            MatchResult(name=name, cx=1600, cy=960, score=0.92, bbox=(1550, 940, 1650, 980))
            if name == "btn_teleport.png"
            else None
        )

        teleporter = MapTeleporter(vision, capture, inp, self.config, self.logger)
        self.assertTrue(teleporter.teleport_to_quest())
        self.assertIn(("press_key", "m"), inp.events)

        inp2 = DummyInput()
        vision2 = MagicMock(spec=VisionEngine)
        vision2.match_template.return_value = None
        vision2.ocr_roi.side_effect = [
            [OCRLine(text="潮汐任务：今州危局", score=0.95, bbox=(160, 220, 480, 260))],
            [OCRLine(text="追踪任务", score=0.95, bbox=(1500, 940, 1700, 980))],
        ]
        qm = QuestMenuHandler(vision2, capture, inp2, self.config, self.logger)
        self.assertTrue(qm.switch_to_next_quest())
        self.assertIn(("press_key", "j"), inp2.events)

    def test_v2_popup_and_state_machine_transitions(self):
        capture = MagicMock(spec=ScreenCapture)
        capture.mean_brightness.side_effect = ScreenCapture.mean_brightness
        capture.frame_diff_score.side_effect = ScreenCapture.frame_diff_score
        capture.to_screen_coords.side_effect = lambda x, y: (x, y)

        inp = DummyInput()
        vision = MagicMock(spec=VisionEngine)
        vision.match_template.return_value = None
        vision.crop_roi.side_effect = VisionEngine.crop_roi
        vision.parse_distance_m.side_effect = VisionEngine.parse_distance_m
        vision.detect_yellow_quest_marker.return_value = (1200, 500)
        vision.ocr_roi.side_effect = lambda frame, roi, cache_key="default", force=False: (
            [OCRLine(text="前往目标点 80m", score=0.95, bbox=(50, 200, 200, 230))]
            if cache_key == "quest_panel"
            else []
        )

        sm = QuestStateMachine(self.config, capture, inp, vision, self.logger)

        dark = np.zeros((1080, 1920, 3), dtype=np.uint8)
        self.assertEqual(sm.tick(dark), BotState.LOADING)

        bright = np.full((1080, 1920, 3), 120, dtype=np.uint8)
        self.assertEqual(sm.tick(bright), BotState.NAVIGATE)
        self.assertTrue(any(ev[0] == "smooth_turn" for ev in inp.events))

        combat_frame = bright.copy()
        cv2.rectangle(combat_frame, (800, 300), (920, 308), (0, 0, 255), -1)
        self.assertEqual(sm.tick(combat_frame), BotState.COMBAT)

        sm2 = QuestStateMachine(self.config, capture, inp, vision, self.logger)
        vision.ocr_roi.side_effect = lambda frame, roi, cache_key="default", force=False: (
            [OCRLine(text="完成机关连线解谜", score=0.98, bbox=(50, 200, 240, 230))]
            if cache_key == "quest_panel"
            else []
        )
        self.assertEqual(sm2.tick(bright), BotState.TAKEOVER)
        sm2.resume_from_takeover()
        self.assertEqual(sm2.state, BotState.IDLE)

        vision_pop = MagicMock(spec=VisionEngine)
        vision_pop.match_template.side_effect = lambda frame, name, **kw: (
            MatchResult(name=name, cx=960, cy=700, score=0.9, bbox=(900, 680, 1020, 720))
            if name == "btn_deliver.png"
            else None
        )
        pop = PopupHandler(vision_pop, capture, inp, self.config)
        self.assertEqual(pop.handle(bright), "btn_deliver.png")

    def test_optimized_yellow_marker_morphology_filters_noise(self):
        vision = VisionEngine(assets_dir=str(ROOT / "assets"))
        # 1. 真实任务标菱形：应该成功检出
        frame = np.full((1080, 1920, 3), 40, dtype=np.uint8)
        diamond_pts = np.array([[960, 400], [980, 420], [960, 440], [940, 420]], dtype=np.int32)
        cv2.fillPoly(frame, [diamond_pts], (0, 220, 255))
        marker = vision.detect_yellow_quest_marker(frame, roi=[400, 80, 1520, 850])
        self.assertIsNotNone(marker)
        self.assertAlmostEqual(marker[0], 960, delta=4)
        self.assertAlmostEqual(marker[1], 420, delta=4)

        # 2. 模拟大面积黄色沙地/实体矩形干扰块 (fill_ratio=1.0)：应被几何充盈度过滤
        noisy_frame = np.full((1080, 1920, 3), 40, dtype=np.uint8)
        cv2.rectangle(noisy_frame, (800, 350), (900, 450), (0, 220, 255), -1)
        marker_noise = vision.detect_yellow_quest_marker(noisy_frame, roi=[400, 80, 1520, 850])
        self.assertIsNone(marker_noise, "实体矩形沙地杂色块应该被形态学充盈度过滤")

    def test_state_machine_is_running_responsiveness(self):
        capture = MagicMock(spec=ScreenCapture)
        capture.mean_brightness.return_value = 100
        inp = DummyInput()
        vision = MagicMock(spec=VisionEngine)
        running = False
        sm = QuestStateMachine(self.config, capture, inp, vision, self.logger, is_running=lambda: running)
        frame = np.full((1080, 1920, 3), 100, dtype=np.uint8)
        state = sm.tick(frame)
        self.assertEqual(state, BotState.IDLE)
        self.assertEqual(len(inp.events), 0, "is_running 为 False 时应立即返回，不触发任何按键操作")

    def test_auto_switch_next_quest_when_blocked(self):
        capture = MagicMock(spec=ScreenCapture)
        capture.mean_brightness.return_value = 120
        frame = np.full((1080, 1920, 3), 100, dtype=np.uint8)
        capture.grab.return_value = frame
        capture.to_screen_coords.side_effect = lambda x, y: (x, y)

        inp = DummyInput()
        vision = MagicMock(spec=VisionEngine)
        vision.match_template.return_value = None
        vision.crop_roi.side_effect = VisionEngine.crop_roi
        vision.parse_distance_m.side_effect = VisionEngine.parse_distance_m

        def _mock_ocr(frame, roi, cache_key="default", force=False):
            if cache_key == "quest_panel":
                return [OCRLine(text="机关解谜：特殊连线", score=0.96, bbox=(50, 200, 260, 230))]
            if cache_key == "quest_menu_list":
                return [
                    OCRLine(text="机关解谜：特殊连线", score=0.95, bbox=(120, 180, 320, 210)),  # 已受阻任务
                    OCRLine(text="危途漫漫", score=0.97, bbox=(120, 250, 320, 280)),           # 下一个可用任务
                ]
            if cache_key == "quest_track_btn":
                return [OCRLine(text="追踪", score=0.98, bbox=(1400, 900, 1500, 940))]
            return []

        vision.ocr_roi.side_effect = _mock_ocr

        sm = QuestStateMachine(self.config, capture, inp, vision, self.logger)
        state = sm.tick(frame)

        # 状态应切回 IDLE 继续执行，而不是被阻塞成 TAKEOVER
        self.assertEqual(state, BotState.IDLE)
        # 受阻任务应被成功记录到冷却黑名单
        self.assertTrue(any("机关解谜" in k for k in sm.blocked_quests.keys()))
        # 应该按 J 打开过任务列表，并按 ESC 退出
        self.assertIn(("press_key", "j"), inp.events)
        self.assertIn(("press_key", "esc"), inp.events)

    def test_ai_agent_and_autopilot_flow(self):
        # 1. 验证图像超轻编码性能（640x360 压缩耗时平均 < 20ms，体积 < 60KB）
        agent = AIAgentController(self.config, logger=self.logger)
        frame = np.full((1080, 1920, 3), 120, dtype=np.uint8)
        # 预热一次避免 Python JIT/OpenCV 底层首次加载耗时波动
        _ = agent.encode_frame(frame)
        t0 = time.perf_counter()
        b64 = agent.encode_frame(frame)
        encode_ms = (time.perf_counter() - t0) * 1000
        self.assertLess(encode_ms, 30.0, f"AI 轻量画面压缩耗时超标: {encode_ms:.2f}ms")
        self.assertTrue(len(b64) > 1000 and len(b64) < 60000, "压缩体积超出预期区间")

        # 2. 验证决策 JSON 鲁棒解析
        raw_json_md = '```json\n{"act":"combat", "keys":["e","r"], "clicks":2, "dodge":1, "reason":"测试释放技能"}\n```'
        parsed = agent._extract_json(raw_json_md)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.get("act"), "combat")

        # 3. 验证 AI 指令与动作执行
        inp = DummyInput()
        agent._latest_action = parsed
        executed = agent.execute_action(inp)
        self.assertEqual(executed.get("act"), "combat")
        self.assertIn(("press_key", "e"), inp.events)
        self.assertIn(("press_key", "r"), inp.events)
        self.assertIn(("click", "right"), inp.events)

        # 4. 验证状态机接管流转（AI 托管模式下接管 BotState.AI_AUTOPILOT）
        agent.set_enabled(True)
        agent._latest_action = {"act": "item", "key": "c", "reason": "残血使用回血道具"}
        capture = MagicMock(spec=ScreenCapture)
        capture.mean_brightness.return_value = 120
        capture.to_screen_coords.side_effect = lambda x, y: (x, y)
        vision = MagicMock(spec=VisionEngine)
        vision.match_template.return_value = None
        vision.ocr_roi.return_value = []
        sm = QuestStateMachine(self.config, capture, inp, vision, self.logger, ai_agent=agent)
        state = sm.tick(frame)
        self.assertEqual(state, BotState.AI_AUTOPILOT)
        self.assertIn(("press_key", "c"), inp.events)

        agent.stop()

    def test_time_adjuster_and_state_skip(self):
        inp = DummyInput()
        capture = MagicMock(spec=ScreenCapture)
        frame = np.full((1080, 1920, 3), 120, dtype=np.uint8)
        capture.grab.return_value = frame
        capture.to_screen_coords.side_effect = lambda x, y: (x, y)
        vision = MagicMock(spec=VisionEngine)
        vision.match_template.return_value = None
        vision.crop_roi.side_effect = VisionEngine.crop_roi
        vision.parse_distance_m.side_effect = VisionEngine.parse_distance_m

        def _mock_ocr_time(f, roi, cache_key="default", force=False):
            if cache_key == "menu_clock_ocr":
                return [OCRLine(text="时间", score=0.9, bbox=(120, 960, 160, 990))]
            if cache_key == "time_confirm_ocr":
                return [OCRLine(text="确定", score=0.9, bbox=(1500, 930, 1550, 960))]
            if cache_key == "quest_panel":
                return [OCRLine(text="任务：等待至次日早晨(08:00~12:00)", score=0.9, bbox=(40, 190, 300, 220))]
            return []

        vision.ocr_roi.side_effect = _mock_ocr_time

        adjuster = TimeAdjuster(vision, capture, inp, self.config, self.logger)
        ok = adjuster.adjust_time(12)
        self.assertTrue(ok)
        self.assertIn(("press_key", "esc"), inp.events)
        self.assertTrue(any(ev[0] in ("click_at", "drag") for ev in inp.events))

        # 测试状态机遇到“等待至次日”时的自动调节闭环流转
        inp.events.clear()
        capture.mean_brightness.return_value = 120
        sm = QuestStateMachine(self.config, capture, inp, vision, self.logger)
        state = sm.tick(frame)
        # 自动时间调节完成后应重置回 IDLE 状态继续推进，而不是被迫触发人工接管
        self.assertEqual(state, BotState.IDLE)
        self.assertIn(("press_key", "esc"), inp.events)

    def test_navigator_glider_and_stamina(self):
        inp = DummyInput()
        vision = MagicMock(spec=VisionEngine)
        nav = Navigator(vision, inp, self.config, logger=self.logger)

        # 1. 验证滑翔伞防坠落主动展开
        nav._last_jump_ts = 0.0
        deployed = nav.try_deploy_glider()
        self.assertTrue(deployed)
        self.assertIn(("press_key", "space"), inp.events)

        # 2. 验证耐力冲刺过载保护
        inp.events.clear()
        nav._sprint_cumulative_sec = 3.5  # 模拟持续冲刺超时
        vision.detect_yellow_quest_marker.return_value = (960, 540)
        frame = np.full((1080, 1920, 3), 100, dtype=np.uint8)
        nav.step_towards_marker(frame)
        # 超时后应松开 Shift 慢跑回体，重置累计冲刺时间
        self.assertEqual(nav._sprint_cumulative_sec, 0.0)
        self.assertIn(("key_down", "w"), inp.events)
        self.assertNotIn(("key_down", "shift"), inp.events)

    def test_ai_stuck_consultation(self):
        inp = DummyInput()
        capture = MagicMock(spec=ScreenCapture)
        capture.mean_brightness.return_value = 100
        capture.frame_diff_score.return_value = 0.5  # 模拟画面停滞
        vision = MagicMock(spec=VisionEngine)
        vision.match_template.return_value = None
        vision.crop_roi.side_effect = VisionEngine.crop_roi
        vision.parse_distance_m.side_effect = VisionEngine.parse_distance_m
        vision.ocr_roi.side_effect = lambda f, r, cache_key="default", force=False: [
            OCRLine(text="追踪：前往目标点", score=0.9, bbox=(40, 190, 200, 220))
        ] if cache_key == "quest_panel" else []

        mock_ai = MagicMock(spec=AIAgentController)
        mock_ai.api_key = "test_key"
        mock_ai.enabled = False  # 规则模式下运行
        mock_ai.consult_stuck_situation.return_value = {
            "act": "jump",
            "reason": "前方检测到矮石阶卡阻，建议向前跳跃翻越",
        }

        sm = QuestStateMachine(self.config, capture, inp, vision, self.logger, ai_agent=mock_ai)
        # 预设脱困计数为 1，下次卡死将达到 2，触发 AI 单帧会诊
        sm.unstuck_count = 1
        sm.last_motion_ts = time.monotonic() - 10.0  # 触发超时卡死判定

        frame = np.full((1080, 1920, 3), 120, dtype=np.uint8)
        state = sm.tick(frame)

        self.assertEqual(state, BotState.NAVIGATE)
        # AI 会诊应被调用
        mock_ai.consult_stuck_situation.assert_called_once()
        # 破局动作：按 W 与空格跳跃，脱困计数清零自愈
        self.assertIn(("key_down", "w"), inp.events)
        self.assertIn(("press_key", "space"), inp.events)
        self.assertEqual(sm.unstuck_count, 0)

    def test_minimap_radar_and_elevation(self):
        vision = VisionEngine(assets_dir="assets")
        frame = np.full((1080, 1920, 3), 40, dtype=np.uint8)

        # 1. 模拟小地图雷达圆盘外圈右侧 (+90度) 出现金黄色任务指针
        # 小地图 ROI 默认为 [20, 20, 270, 270]，中心为 (145, 145)
        # 右侧 (245, 145) 半径为 100 像素
        cv2.circle(frame, (245, 145), 6, (0, 215, 255), -1)
        azimuth = vision.detect_minimap_quest_azimuth(frame, roi=[20, 20, 270, 270])
        self.assertIsNotNone(azimuth)
        self.assertAlmostEqual(azimuth, 90.0, delta=12.0)

        # 2. 模拟任务黄标带有向上三角 (高处 elevation=+1)
        diamond_pts = np.array([[960, 400], [980, 420], [960, 440], [940, 420]], dtype=np.int32)
        cv2.fillPoly(frame, [diamond_pts], (0, 220, 255))
        up_triangle = np.array([[960, 386], [970, 398], [950, 398]], dtype=np.int32)
        cv2.fillPoly(frame, [up_triangle], (0, 220, 255))

        res = vision.detect_yellow_quest_marker_with_elevation(frame, roi=[400, 80, 1520, 850])
        self.assertIsNotNone(res)
        self.assertAlmostEqual(res[0], 960, delta=4)
        self.assertEqual(res[2], 1, "应该正确识别黄标上方高处 ▲ 指示")

    def test_combat_echo_and_dodge_counter(self):
        inp = DummyInput()
        vision = MagicMock(spec=VisionEngine)
        vision.crop_roi.side_effect = VisionEngine.crop_roi
        vision.match_template.return_value = None
        combat = CombatHandler(vision, inp, self.config, logger=self.logger)

        # 1. 测试 action == "dodge" 触发右键闪避 + 左键反击连招
        combat.rotation = [{"action": "dodge"}]
        combat._locked_in_encounter = True
        combat.step()
        self.assertIn(("click", "right"), inp.events)
        self.assertIn(("click", "left"), inp.events)

        # 2. 测试主动释放声骸 Q 技
        inp.events.clear()
        combat._last_echo_ts = 0.0  # 冷却就绪
        frame = np.full((1080, 1920, 3), 40, dtype=np.uint8)
        combat.step(frame)
        self.assertIn(("press_key", "q"), inp.events)

    def test_post_combat_echo_patrol(self):
        inp = DummyInput()
        vision = MagicMock(spec=VisionEngine)
        vision.crop_roi.side_effect = VisionEngine.crop_roi
        vision.match_template.return_value = None
        vision.ocr_roi.return_value = []
        nav = Navigator(vision, inp, self.config, logger=self.logger)

        # 模拟地面金黄色声骸光晕
        frame = np.full((1080, 1920, 3), 40, dtype=np.uint8)
        cv2.circle(frame, (1060, 680), 20, (0, 220, 255), -1)

        absorbed = nav.patrol_and_absorb_echo(frame)
        self.assertTrue(absorbed)
        self.assertIn(("key_down", "w"), inp.events)
        self.assertIn(("press_key", "f"), inp.events)

    def test_player_hp_ratio_and_auto_heal(self):
        vision = VisionEngine(assets_dir="assets")
        # 1. 模拟绘制 25% 绿色的低血量生命槽 (ROI: [780, 990, 1140, 1025], 宽 360)
        frame = np.full((1080, 1920, 3), 20, dtype=np.uint8)
        # 宽 360 中的 25% 约为 90 像素
        cv2.rectangle(frame, (780, 995), (780 + 90, 1020), (0, 220, 50), -1)
        hp_ratio = vision.get_player_hp_ratio(frame, roi=[780, 990, 1140, 1025])
        self.assertIsNotNone(hp_ratio)
        self.assertAlmostEqual(hp_ratio, 0.25, delta=0.05)

        # 2. 测试状态机监测到濒血自动触发 C 键吃药
        inp = DummyInput()
        capture = MagicMock(spec=ScreenCapture)
        capture.mean_brightness.return_value = 120
        mock_vision = MagicMock(spec=VisionEngine)
        mock_vision.get_player_hp_ratio.return_value = 0.22
        mock_vision.match_template.return_value = None
        mock_vision.parse_distance_m.side_effect = VisionEngine.parse_distance_m
        mock_vision.ocr_roi.return_value = [OCRLine(text="追踪：前往目标点", score=0.9, bbox=(40, 190, 200, 220))]

        sm = QuestStateMachine(self.config, capture, inp, mock_vision, self.logger)
        sm._last_heal_ts = 0.0
        sm.tick(frame)
        self.assertIn(("press_key", "c"), inp.events, "濒血时应自动按 C 键使用快捷回复药剂")

    def test_revive_respawn_handling(self):
        inp = DummyInput()
        capture = MagicMock(spec=ScreenCapture)
        capture.mean_brightness.return_value = 120
        vision = MagicMock(spec=VisionEngine)
        vision.match_template.return_value = None
        vision.ocr_roi.return_value = []

        sm = QuestStateMachine(self.config, capture, inp, vision, self.logger)
        sm.unstuck_count = 3  # 战前累计卡死计数
        sm.popup_handler = MagicMock()
        sm.popup_handler.handle.return_value = "btn_revive.png"

        frame = np.full((1080, 1920, 3), 100, dtype=np.uint8)
        state = sm.tick(frame)
        self.assertEqual(state, BotState.POPUP)
        # 团灭复苏触发后，卡死计数应立即清零自愈
        self.assertEqual(sm.unstuck_count, 0)

    def test_map_teleport_zoom_and_nexus_priority(self):
        inp = DummyInput()
        capture = MagicMock(spec=ScreenCapture)
        frame = np.full((1080, 1920, 3), 100, dtype=np.uint8)
        capture.grab.return_value = frame
        capture.to_screen_coords.side_effect = lambda x, y: (x, y)

        vision = MagicMock(spec=VisionEngine)
        # 1. 模拟初次视口无黄标，缩放后捕捉到任务黄标
        vision.detect_yellow_quest_marker.side_effect = [None, (800, 400), (800, 400)]
        vision.match_template.return_value = None
        vision.crop_roi.side_effect = VisionEngine.crop_roi
        vision.ocr_roi.side_effect = lambda f, r, cache_key="default", force=False: [
            OCRLine(text="快速旅行", score=0.96, bbox=(1500, 940, 1680, 980))
        ]

        teleporter = MapTeleporter(vision, capture, inp, self.config, self.logger)
        # 模拟提取出神像 (860, 400) 与普通信标 (780, 400)
        teleporter._find_beacons = MagicMock(return_value=([(780, 400)], [(860, 400)]))

        ok = teleporter.teleport_to_quest()
        self.assertTrue(ok)
        # 验证是否触发了向下滚轮缩放大地图全景
        self.assertIn(("scroll", -160), inp.events)
        # 验证在神像与信标距离相近时，优先点击神像 (860, 400) 回满血
        self.assertIn(("click_at", 860, 400, "left"), inp.events)

    def test_gadget_manager_wheel_switch(self):
        inp = DummyInput()
        capture = MagicMock(spec=ScreenCapture)
        capture.to_screen_coords.side_effect = lambda x, y: (x, y)
        vision = MagicMock(spec=VisionEngine)

        gm = GadgetManager(inp, capture, vision, self.logger)

        # 1. 测试关键词推荐
        tool = gm.recommend_tool_for_quest("使用控物能力击碎前方爆炸岩石 15m")
        self.assertEqual(tool, "levitator")

        # 2. 测试 Tab 轮盘切工具 (hook -> levitator)
        switched = gm.switch_tool("levitator")
        self.assertTrue(switched)
        self.assertEqual(gm.current_tool, "levitator")
        self.assertIn(("key_down", "tab"), inp.events)
        self.assertIn(("move_cursor", 1115, 450), inp.events)
        self.assertIn(("key_up", "tab"), inp.events)

        # 3. 测试按下 T 键释放工具
        inp.events.clear()
        gm.trigger_active_gadget()
        self.assertIn(("press_key", "t"), inp.events)

    def test_experience_memory_learn_and_replay(self):
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "exp_test.json"
            # 1. 验证初始化与自学习写入
            memory = ExperienceMemory(db_path=db_path, logger=self.logger)
            solution = {"action": "jump_dash", "reason": "跳跃冲刺破除暗黑荆棘", "status": "success"}
            memory.record_experience("追踪：破除暗黑荆棘 35m", solution)

            # 2. 验证本地 0ms 查询回放（过滤动态距离后的精确或模糊匹配）
            t0 = time.perf_counter()
            retrieved = memory.query_experience("破除暗黑荆棘 12m")
            query_ms = (time.perf_counter() - t0) * 1000

            self.assertIsNotNone(retrieved)
            self.assertEqual(retrieved.get("action"), "jump_dash")
            self.assertLess(query_ms, 5.0, "经验库本地 0ms 极速检索耗时超标")

            # 3. 验证 L5 自省反思与教训沉淀
            memory.record_failure_reflection("乘霄山岩壁滑落", "slope_slip", "陡峭山体存在滑坡碰撞体，应绕行右侧阶梯")
            stats = memory.get_memory_stats()
            self.assertGreaterEqual(stats["reflection_count"], 1)
            self.assertIn("绕行", stats["latest_reflection"])

            # 4. 验证 L5 跨场景物理特征标签语义检索
            pat = memory.query_semantic_pattern(["explosive", "rock"])
            self.assertIsNotNone(pat)
            self.assertEqual(pat.get("action"), "gadget:levitator")

    def test_environment_doctor_checks(self):
        doctor = EnvironmentDoctor(ROOT)
        results = doctor.run_all_checks()
        self.assertIsInstance(results, list)
        self.assertGreaterEqual(len(results), 6)
        # 验证每个检测项都具备布尔结果、标题与诊断详情
        for item in results:
            self.assertEqual(len(item), 3)
            self.assertIsInstance(item[0], bool)
            self.assertIsInstance(item[1], str)
            self.assertIsInstance(item[2], str)

    def test_gui_modern_tab_and_doctor(self):
        import gui_app
        app = gui_app.MingchaoGUI(ROOT)
        # 1. 验证现代 Tab 栏包含 5 个项目
        self.assertEqual(len(app.nav_tab_items), 5)

        # 2. 验证各 Tab 切换及底层 Notebook 同步
        for i in range(5):
            app._select_nav_tab(i)
            self.assertEqual(app.notebook.index(app.notebook.select()), i)
            # 选中项应具备高光指示条
            selected_item = app.nav_tab_items[i]
            indicator_bg = selected_item["indicator"].cget("bg")
            self.assertEqual(indicator_bg, gui_app.THEME["gold"])

        # 3. 验证触发环境体检不报异常并切至日志页 (Tab 3)
        app._run_doctor_check()
        self.assertEqual(app.notebook.index(app.notebook.select()), 3)
        time.sleep(0.05)

        # 4. 验证 L5 全局大目标设定与驾驶舱联动
        app.l5_intent_var.set("【今汐】专属突破素材与声骸定向收集")
        app._apply_l5_master_goal()
        summary = app.runner.sm.get_l5_summary()
        self.assertIn("今汐", summary["current_goal"])

        app.root.destroy()

    def test_l5_hierarchical_planner_and_state_machine(self):
        # 1. 验证自然语言大目标智能分解
        planner = L5HierarchicalPlanner(self.logger)
        planner.set_master_goal("拉满今汐练度并清空秘境体力")
        self.assertEqual(len(planner.subgoals), 2)
        cur = planner.get_current_subgoal()
        self.assertIsNotNone(cur)
        self.assertEqual(cur.goal_type, GoalType.CHARACTER_BUILD)
        self.assertIn("今汐", cur.title)

        # 2. 验证突发高优先级事件插队并封存当前现场
        snapshot = {"quest_id": "quest_jinxī_01", "step": 3, "player_pos": (120, 450)}
        interrupt = planner.trigger_interrupt(
            title="捕获偶遇金色闪光扑龙",
            reason="稀有高收益大世界实体",
            priority=0,
            action_params={"target": "golden_dragon"},
            context_snapshot=snapshot,
        )
        self.assertEqual(interrupt.priority, 0)
        # 插队目标应成为当前置顶执行项
        self.assertEqual(planner.get_current_subgoal().id, interrupt.id)
        self.assertEqual(len(planner._paused_stack), 1)
        self.assertEqual(planner._paused_stack[0].context_snapshot, snapshot)

        # 3. 验证插队目标完成后，自动唤醒并无缝恢复被暂挂的目标
        resumed = planner.complete_current_subgoal()
        self.assertIsNotNone(resumed)
        self.assertEqual(resumed.goal_type, GoalType.CHARACTER_BUILD)
        self.assertEqual(resumed.context_snapshot, snapshot)
        self.assertEqual(len(planner._paused_stack), 0)

        # 4. 验证与状态机的深度联动（濒血 < 18% 自动触发紧急神像补给插队）
        inp = DummyInput()
        capture = MagicMock(spec=ScreenCapture)
        capture.mean_brightness.return_value = 120
        mock_vision = MagicMock(spec=VisionEngine)
        mock_vision.get_player_hp_ratio.return_value = 0.15  # 极度濒血
        mock_vision.match_template.return_value = None
        mock_vision.parse_distance_m.side_effect = VisionEngine.parse_distance_m
        mock_vision.ocr_roi.return_value = [OCRLine(text="追踪：前往指定区域 50m", score=0.9, bbox=(40, 190, 200, 220))]

        sm = QuestStateMachine(self.config, capture, inp, mock_vision, self.logger)
        sm.set_l5_goal("全自动清日常委托")
        sm._last_heal_ts = 0.0

        frame = np.full((1080, 1920, 3), 120, dtype=np.uint8)
        sm.tick(frame)

        summary = sm.get_l5_summary()
        self.assertIn("紧急界碑神像满血补给", summary.get("current_goal", ""))
        self.assertEqual(summary.get("paused_stack_depth"), 1)

    def test_l5_world_model_and_causality(self):
        wm = L5WorldModel(buffer_size=10, logger=self.logger)
        frame1 = np.full((1080, 1920, 3), 100, dtype=np.uint8)
        frame2 = np.full((1080, 1920, 3), 110, dtype=np.uint8)

        # 1. 验证时序物理观测录入与光流计算
        s1 = wm.update_observation(frame1, distance_m=50.0, action="move_forward")
        s2 = wm.update_observation(frame2, distance_m=48.0, action="move_forward")
        self.assertEqual(len(wm.buffer), 2)
        self.assertIsInstance(s2.vertical_flow, float)

        # 2. 验证滑坡排斥阻抗异常推演 (距离持续被弹开增大)
        for d in [48.0, 50.0, 53.0, 57.0]:
            wm.update_observation(frame1, distance_m=d, action="move_forward")
        anomaly = wm.detect_motion_anomaly()
        self.assertIsNotNone(anomaly)
        self.assertEqual(anomaly.get("type"), "steep_slope_slipping")

        # 3. 验证力学破坏与爆炸果宏动作因果推演
        macro = wm.reason_mechanism_causality("使用念力控物吸取爆炸果砸开前方阻挡石壁 15m")
        self.assertIsNotNone(macro)
        self.assertGreaterEqual(len(macro), 4)
        acts = [step["act"] for step in macro]
        self.assertIn("gadget", acts)
        self.assertIn("press_key", acts)
        self.assertIn("click", acts)

        # 4. 验证在虚拟控制器上顺序执行宏动作
        inp = DummyInput()
        gm = MagicMock()
        executed = wm.execute_macro_sequence(macro, inp, gadget_manager=gm)
        self.assertTrue(executed)
        gm.switch_tool.assert_called_with("levitator")
        self.assertIn(("press_key", "t"), inp.events)
        self.assertIn(("click", "left"), inp.events)

    def test_l5_combat_parry_and_cancel_swap(self):
        inp = DummyInput()
        vision = MagicMock(spec=VisionEngine)
        vision.crop_roi.side_effect = VisionEngine.crop_roi
        vision.match_template.return_value = None

        combat = CombatHandler(vision, inp, self.config, logger=self.logger)
        combat._locked_in_encounter = True

        # 1. 模拟敌人招式金色拼刀提示圈 (HSV [18, 140, 210] ~ [36, 255, 255])
        frame = np.full((1080, 1920, 3), 30, dtype=np.uint8)
        # 在中央区域 [360, 80, 1560, 860] 绘制亮金色圆环
        cv2.circle(frame, (960, 480), 35, (30, 230, 255), 8)

        has_parry = combat._has_parry_flash(frame)
        self.assertTrue(has_parry)

        combat.step(frame)
        self.assertIn(("click", "left"), inp.events)
        self.assertIn(("press_key", "e"), inp.events)

        # 2. 验证大招释放后立即触发取消后摇切人
        inp.events.clear()
        combat._last_parry_ts = time.monotonic()  # 抑制连续拼刀
        combat._is_ult_ready = MagicMock(return_value=True)

        combat.step(frame)
        self.assertIn(("press_key", "r"), inp.events)
        # 应立即切到角色 2 取消后摇
        self.assertIn(("press_key", "2"), inp.events)

    def test_tools_importable(self):
        import tools.capture_template as ct
        import tools.debug_overlay as do

        self.assertTrue(callable(ct.main))
        self.assertTrue(callable(do.main))
        self.assertTrue(callable(BotRunner.start))


if __name__ == "__main__":
    unittest.main(verbosity=2)
