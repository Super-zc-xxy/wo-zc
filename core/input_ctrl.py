import ctypes
import random
import time
from typing import Dict

# Win32 SendInput constants
INPUT_MOUSE = 0
INPUT_KEYBOARD = 1

KEYEVENTF_SCANCODE = 0x0008
KEYEVENTF_KEYUP = 0x0002

MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
MOUSEEVENTF_MIDDLEDOWN = 0x0020
MOUSEEVENTF_MIDDLEUP = 0x0040
MOUSEEVENTF_WHEEL = 0x0800

# DirectInput / Hardware Scan Codes (US Keyboard)
SCAN_CODES: Dict[str, int] = {
    "esc": 0x01,
    "1": 0x02,
    "2": 0x03,
    "3": 0x04,
    "4": 0x05,
    "5": 0x06,
    "tab": 0x0F,
    "q": 0x10,
    "w": 0x11,
    "e": 0x12,
    "r": 0x13,
    "t": 0x14,
    "a": 0x1E,
    "s": 0x1F,
    "d": 0x20,
    "f": 0x21,
    "g": 0x22,
    "j": 0x24,
    "m": 0x32,
    "v": 0x2F,
    "lshift": 0x2A,
    "shift": 0x2A,
    "lctrl": 0x1D,
    "ctrl": 0x1D,
    "lalt": 0x38,
    "alt": 0x38,
    "space": 0x39,
    "enter": 0x1C,
}

ULONG_PTR = ctypes.POINTER(ctypes.c_ulong)


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_long),
        ("dy", ctypes.c_long),
        ("mouseData", ctypes.c_ulong),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", ULONG_PTR),
    ]


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", ctypes.c_ushort),
        ("wScan", ctypes.c_ushort),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", ULONG_PTR),
    ]


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [
        ("uMsg", ctypes.c_ulong),
        ("wParamL", ctypes.c_ushort),
        ("wParamH", ctypes.c_ushort),
    ]


class _INPUT_UNION(ctypes.Union):
    _fields_ = [
        ("mi", MOUSEINPUT),
        ("ki", KEYBDINPUT),
        ("hi", HARDWAREINPUT),
    ]


class INPUT(ctypes.Structure):
    _anonymous_ = ("_union",)
    _fields_ = [
        ("type", ctypes.c_ulong),
        ("_union", _INPUT_UNION),
    ]


class InputController:
    def __init__(self, jitter_min: float = 0.06, jitter_max: float = 0.14) -> None:
        self.jitter_min = jitter_min
        self.jitter_max = jitter_max
        self._user32 = getattr(ctypes, "windll", None).user32 if hasattr(ctypes, "windll") else None
        self._held_keys = set()

    @staticmethod
    def _gaussian_duration(base: float, sigma: float = 0.012, min_val: float = 0.035, max_val: float = 0.11) -> float:
        """拟人化高斯正态分布耗时生成，模拟人类手部按键生理特征。"""
        val = random.gauss(base, sigma)
        return max(min_val, min(max_val, val))

    def _sleep_jitter(self, base: float = 0.0) -> None:
        jitter = self._gaussian_duration(
            (self.jitter_min + self.jitter_max) / 2.0,
            sigma=0.015,
            min_val=self.jitter_min * 0.8,
            max_val=self.jitter_max * 1.25,
        )
        time.sleep(base + jitter)

    def _send_input(self, inp: INPUT) -> None:
        if self._user32 is None:
            return
        self._user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))

    def key_down(self, key: str) -> None:
        code = SCAN_CODES.get(key.lower())
        if code is None:
            raise ValueError(f"不支持的按键扫描码: {key}")
        ki = KEYBDINPUT(0, code, KEYEVENTF_SCANCODE, 0, None)
        self._send_input(INPUT(INPUT_KEYBOARD, _INPUT_UNION(ki=ki)))
        self._held_keys.add(key.lower())

    def key_up(self, key: str) -> None:
        code = SCAN_CODES.get(key.lower())
        if code is None:
            raise ValueError(f"不支持的按键扫描码: {key}")
        ki = KEYBDINPUT(0, code, KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP, 0, None)
        self._send_input(INPUT(INPUT_KEYBOARD, _INPUT_UNION(ki=ki)))
        self._held_keys.discard(key.lower())

    def press_key(self, key: str, hold_sec: float = 0.05) -> None:
        self.key_down(key)
        time.sleep(self._gaussian_duration(hold_sec, sigma=0.015, min_val=0.035, max_val=0.12))
        self.key_up(key)
        self._sleep_jitter()

    def release_all(self) -> None:
        for k in list(self._held_keys):
            try:
                self.key_up(k)
            except Exception:
                pass
        self._held_keys.clear()

    def move_cursor(self, screen_x: int, screen_y: int) -> None:
        if self._user32 is None:
            return
        self._user32.SetCursorPos(int(screen_x), int(screen_y))

    def move_mouse_rel(self, dx: int, dy: int = 0) -> None:
        mi = MOUSEINPUT(int(dx), int(dy), 0, MOUSEEVENTF_MOVE, 0, None)
        self._send_input(INPUT(INPUT_MOUSE, _INPUT_UNION(mi=mi)))

    def smooth_turn(self, total_dx: int, steps: int = 6, step_delay: float = 0.012) -> None:
        """
        S 型三次贝塞尔/缓动非线性平滑转角（Ease-in-out Smoothstep: 3t^2 - 2t^3）。
        伴随微小的垂直轴人体生理震颤（Micro-Jitter），杜绝机械匀速特征。
        """
        if total_dx == 0:
            return
        steps = max(2, steps)
        prev_progress = 0.0
        for i in range(1, steps + 1):
            t = i / float(steps)
            # Smoothstep 缓动曲线
            cur_progress = (3.0 * t * t) - (2.0 * t * t * t)
            step_dx = int(round(total_dx * (cur_progress - prev_progress)))
            prev_progress = cur_progress

            # 注入极其微小的人体手部垂直微晃动（-1, 0, 1 像素，仅在步长较大时偶尔微晃）
            micro_dy = random.choice([-1, 0, 1]) if (abs(total_dx) > 30 and random.random() < 0.25) else 0

            if step_dx != 0 or micro_dy != 0:
                self.move_mouse_rel(step_dx, micro_dy)

            time.sleep(step_delay * random.uniform(0.85, 1.15))

    def mouse_down(self, button: str = "left") -> None:
        down_flag = {
            "left": MOUSEEVENTF_LEFTDOWN,
            "right": MOUSEEVENTF_RIGHTDOWN,
            "middle": MOUSEEVENTF_MIDDLEDOWN,
        }[button]
        self._send_input(INPUT(INPUT_MOUSE, _INPUT_UNION(mi=MOUSEINPUT(0, 0, 0, down_flag, 0, None))))

    def mouse_up(self, button: str = "left") -> None:
        up_flag = {
            "left": MOUSEEVENTF_LEFTUP,
            "right": MOUSEEVENTF_RIGHTUP,
            "middle": MOUSEEVENTF_MIDDLEUP,
        }[button]
        self._send_input(INPUT(INPUT_MOUSE, _INPUT_UNION(mi=MOUSEINPUT(0, 0, 0, up_flag, 0, None))))

    def click(self, button: str = "left", hold_sec: float = 0.04) -> None:
        self.mouse_down(button)
        time.sleep(self._gaussian_duration(hold_sec, sigma=0.012, min_val=0.03, max_val=0.09))
        self.mouse_up(button)
        self._sleep_jitter()

    def click_at(self, screen_x: int, screen_y: int, button: str = "left") -> None:
        self.move_cursor(screen_x, screen_y)
        time.sleep(random.uniform(0.03, 0.06))
        self.click(button=button)

    def drag(self, start_x: int, start_y: int, end_x: int, end_y: int, steps: int = 12, duration: float = 0.35) -> None:
        """
        拟人化平滑拖拽，使用 Smoothstep 曲线移动并在松开前后添加微小生理停顿。
        """
        self.move_cursor(start_x, start_y)
        time.sleep(random.uniform(0.05, 0.08))
        self.mouse_down("left")
        time.sleep(random.uniform(0.04, 0.07))
        steps = max(3, steps)
        step_delay = duration / float(steps)
        for i in range(1, steps + 1):
            t = i / float(steps)
            cur = (3.0 * t * t) - (2.0 * t * t * t)
            cx = int(round(start_x + (end_x - start_x) * cur))
            cy = int(round(start_y + (end_y - start_y) * cur))
            self.move_cursor(cx, cy)
            time.sleep(step_delay * random.uniform(0.9, 1.1))
        time.sleep(random.uniform(0.04, 0.08))
        self.mouse_up("left")
        self._sleep_jitter()

    def scroll(self, delta: int) -> None:
        # delta > 0 向上滚，delta < 0 向下滚（标准步长 120）
        mi = MOUSEINPUT(0, 0, ctypes.c_ulong(delta & 0xFFFFFFFF).value, MOUSEEVENTF_WHEEL, 0, None)
        self._send_input(INPUT(INPUT_MOUSE, _INPUT_UNION(mi=mi)))
        self._sleep_jitter()
