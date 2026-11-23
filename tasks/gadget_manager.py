import logging
import time
from typing import Dict, Optional, Tuple

from core.capture import ScreenCapture
from core.input_ctrl import InputController
from core.vision import VisionEngine


class GadgetManager:
    """
    鸣潮多功能探索工具轮盘智能管理器：
    管理并自适应切换游戏中的四类核心探索小工具：
    1. hook（钩索）：高空飞跃、跨越裂谷、飞向锚点
    2. levitator（控物）：念力吸附岩石、爆炸果、投掷击碎石壁、重力踏板
    3. sensor（感知）：侦测共鸣痕迹、追踪虚影足迹与隐藏线索
    4. chronosorter（溯流）：机关时间轨迹回溯与齿轮拼合
    """

    # 1080p 轮盘对应工具的扇区方向坐标（以屏幕中心 960, 540 为基准）
    TOOL_OFFSETS: Dict[str, Tuple[int, int]] = {
        "hook": (960, 360),            # 正上方 12点：钩索
        "levitator": (1115, 450),       # 右上方 2点：控物
        "chronosorter": (1115, 630),    # 右下方 4点：溯流
        "sensor": (805, 540),           # 正左方 9点：感知
    }

    KEYWORD_MAPPING: Dict[str, Tuple[str, ...]] = {
        "levitator": ("控物", "吸取", "投掷", "爆炸", "碎石", "砸开", "重物"),
        "sensor": ("感知", "追踪", "足迹", "探寻", "痕迹", "残影", "虚影"),
        "chronosorter": ("溯流", "回溯", "倒流", "时间机关", "复原"),
        "hook": ("钩索", "飞跃", "锚点", "飞翔", "高处"),
    }

    def __init__(
        self,
        inp: InputController,
        capture: ScreenCapture,
        vision: VisionEngine,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.inp = inp
        self.capture = capture
        self.vision = vision
        self.logger = logger
        self.current_tool: str = "hook"
        self._last_switch_ts: float = 0.0

    def recommend_tool_for_quest(self, quest_text: str) -> Optional[str]:
        """根据当前任务文本与关键词推荐应当切换的探索工具。"""
        if not quest_text:
            return None
        for tool_name, keywords in self.KEYWORD_MAPPING.items():
            if any(kw in quest_text for kw in keywords):
                return tool_name
        return None

    def switch_tool(self, tool_name: str) -> bool:
        """
        长按 Tab 呼出探索工具轮盘，精准移动光标选择目标工具后松开。
        """
        tool_name = tool_name.lower().strip()
        if tool_name not in self.TOOL_OFFSETS:
            if self.logger:
                self.logger.warning(f"未知探索工具类型: {tool_name}")
            return False

        if tool_name == self.current_tool:
            return True

        now = time.monotonic()
        if (now - self._last_switch_ts) < 1.5:
            return False

        if self.logger:
            self.logger.info(f"[探索工具] 正在通过 Tab 轮盘切换工具: {self.current_tool} -> {tool_name}")

        self.inp.release_all()
        # 1. 长按 Tab 呼出轮盘
        self.inp.key_down("tab")
        time.sleep(0.28)

        # 2. 定位到目标工具扇区坐标
        tx, ty = self.TOOL_OFFSETS[tool_name]
        try:
            coords = self.capture.to_screen_coords(tx, ty)
            if isinstance(coords, (tuple, list)) and len(coords) == 2:
                sx, sy = int(coords[0]), int(coords[1])
            else:
                sx, sy = tx, ty
        except Exception:
            sx, sy = tx, ty
        self.inp.move_cursor(sx, sy)
        time.sleep(0.12)

        # 3. 松开 Tab 完成切换
        self.inp.key_up("tab")
        time.sleep(0.20)

        self.current_tool = tool_name
        self._last_switch_ts = now
        return True

    def trigger_active_gadget(self) -> None:
        """按下 T 键释放当前激活的探索工具。"""
        self.inp.press_key("t")
        time.sleep(0.25)
