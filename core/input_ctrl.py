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

    def _sleep_jitter(self, base: float = 0.0) -> None:
        time.sleep(base + random.uniform(self.jitter_min, self.jitter_max))

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
        time.sleep(hold_sec + random.uniform(0.01, 0.04))
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
        if total_dx == 0:
            return
        steps = max(1, steps)
        per_step = total_dx / steps
        acc = 0.0
        for _ in range(steps):
            acc += per_step
            move_int = int(round(acc))
            if move_int != 0:
                self.move_mouse_rel(move_int, 0)
                acc -= move_int
            time.sleep(step_delay)

    def click(self, button: str = "left", hold_sec: float = 0.04) -> None:
        down_flag, up_flag = {
            "left": (MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP),
            "right": (MOUSEEVENTF_RIGHTDOWN, MOUSEEVENTF_RIGHTUP),
            "middle": (MOUSEEVENTF_MIDDLEDOWN, MOUSEEVENTF_MIDDLEUP),
        }[button]
        self._send_input(INPUT(INPUT_MOUSE, _INPUT_UNION(mi=MOUSEINPUT(0, 0, 0, down_flag, 0, None))))
        time.sleep(hold_sec + random.uniform(0.01, 0.03))
        self._send_input(INPUT(INPUT_MOUSE, _INPUT_UNION(mi=MOUSEINPUT(0, 0, 0, up_flag, 0, None))))
        self._sleep_jitter()

    def click_at(self, screen_x: int, screen_y: int, button: str = "left") -> None:
        self.move_cursor(screen_x, screen_y)
        time.sleep(random.uniform(0.03, 0.06))
        self.click(button=button)

    def scroll(self, delta: int) -> None:
        # delta > 0 向上滚，delta < 0 向下滚（标准步长 120）
        mi = MOUSEINPUT(0, 0, ctypes.c_ulong(delta & 0xFFFFFFFF).value, MOUSEEVENTF_WHEEL, 0, None)
        self._send_input(INPUT(INPUT_MOUSE, _INPUT_UNION(mi=mi)))
        self._sleep_jitter()
