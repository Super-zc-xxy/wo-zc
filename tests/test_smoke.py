import logging
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.auto_collector import AutoCalibrator, AutoTemplateCollector
from core.bot_runner import BotRunner
from core.capture import EXCLUDED_TITLE_KEYWORDS, ScreenCapture, WindowRect
from core.input_ctrl import InputController, SCAN_CODES
from core.presets import PresetManager
from core.state_machine import BotState, QuestStateMachine
from core.vision import MatchResult, OCRLine, VisionEngine
from tasks.combat import CombatHandler
from tasks.dialogue import DialogueHandler
from tasks.map_teleport import MapTeleporter
from tasks.navigator import Navigator
from tasks.popup_handler import PopupHandler
from tasks.quest_menu import QuestMenuHandler


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
        import onnxruntime
        import rapidocr
        import win32gui

        self.assertIsNotNone(gui_app)
        self.assertIsNotNone(keyboard)
        self.assertIsNotNone(mss)
        self.assertIsNotNone(onnxruntime)
        self.assertIsNotNone(rapidocr)
        self.assertIsNotNone(win32gui)
        self.assertIn("w", SCAN_CODES)
        self.assertIn("f", SCAN_CODES)
        self.assertIn("j", SCAN_CODES)
        self.assertIn("m", SCAN_CODES)

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

    def test_tools_importable(self):
        import tools.capture_template as ct
        import tools.debug_overlay as do

        self.assertTrue(callable(ct.main))
        self.assertTrue(callable(do.main))
        self.assertTrue(callable(BotRunner.start))


if __name__ == "__main__":
    unittest.main(verbosity=2)
