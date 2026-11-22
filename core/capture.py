import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import mss
import numpy as np

try:
    import win32gui
    import win32process
except ImportError:
    win32gui = None
    win32process = None


EXCLUDED_TITLE_KEYWORDS = (
    "mingchaojb",
    "自动任务助手",
    "debug overlay",
    "capture template",
    "google chrome",
    "microsoft edge",
    "firefox",
    "visual studio code",
    "powershell",
    "cmd.exe",
    "deepseek",
)

PREFERRED_CLASS_NAMES = (
    "UnrealWindow",
)


@dataclass
class WindowRect:
    hwnd: int
    title: str
    left: int
    top: int
    width: int
    height: int

    def to_mss_monitor(self) -> dict:
        return {
            "left": self.left,
            "top": self.top,
            "width": self.width,
            "height": self.height,
        }


class ScreenCapture:
    def __init__(
        self,
        title_keywords: List[str],
        base_width: int = 1920,
        base_height: int = 1080,
    ) -> None:
        self.title_keywords = title_keywords
        self.base_width = base_width
        self.base_height = base_height
        self._sct = mss.mss()
        self.window: Optional[WindowRect] = None
        self._self_pid = os.getpid()

    def _score_window(self, hwnd: int, title: str, cls_name: str, width: int, height: int) -> int:
        score = 0
        title_lower = title.lower().strip()
        if cls_name in PREFERRED_CLASS_NAMES:
            score += 100
        if title_lower in ("鸣潮", "wuthering waves", "鸣潮  ", "wuthering waves  "):
            score += 80
        elif any(kw.lower() in title_lower for kw in self.title_keywords):
            score += 30
        if width >= 1280 and height >= 720:
            score += 20
        return score

    def find_window(self) -> Optional[WindowRect]:
        if win32gui is None:
            raise RuntimeError("未安装 pywin32，无法定位 Windows 游戏窗口。")

        candidates: List[Tuple[int, int, str, int, int, int, int]] = []

        def _enum_cb(hwnd, _):
            if not win32gui.IsWindowVisible(hwnd):
                return
            title = win32gui.GetWindowText(hwnd) or ""
            if not title.strip():
                return
            title_lower = title.lower()
            if any(ex in title_lower for ex in EXCLUDED_TITLE_KEYWORDS):
                return

            if win32process is not None:
                try:
                    _, pid = win32process.GetWindowThreadProcessId(hwnd)
                    if pid == self._self_pid:
                        return
                except Exception:
                    pass

            cls_name = ""
            try:
                cls_name = win32gui.GetClassName(hwnd) or ""
            except Exception:
                pass

            matched_title = any(kw.lower() in title_lower for kw in self.title_keywords)
            matched_class = cls_name in PREFERRED_CLASS_NAMES and ("鸣潮" in title or "wuthering" in title_lower or "client" in title_lower)
            if not (matched_title or matched_class):
                return

            try:
                left, top, right, bottom = win32gui.GetClientRect(hwnd)
                width = right - left
                height = bottom - top
                if width < 640 or height < 360:
                    return
                x, y = win32gui.ClientToScreen(hwnd, (left, top))
            except Exception:
                return

            score = self._score_window(hwnd, title, cls_name, width, height)
            candidates.append((score, hwnd, title, x, y, width, height))

        win32gui.EnumWindows(_enum_cb, None)
        if not candidates:
            self.window = None
            return None

        candidates.sort(key=lambda item: item[0], reverse=True)
        _, hwnd, title, x, y, width, height = candidates[0]
        self.window = WindowRect(
            hwnd=hwnd,
            title=title,
            left=x,
            top=y,
            width=width,
            height=height,
        )
        return self.window

    def refresh_window(self) -> Optional[WindowRect]:
        if self.window is None or win32gui is None:
            return self.find_window()
        if not win32gui.IsWindow(self.window.hwnd):
            return self.find_window()
        left, top, right, bottom = win32gui.GetClientRect(self.window.hwnd)
        width = right - left
        height = bottom - top
        if width <= 0 or height <= 0:
            return None
        x, y = win32gui.ClientToScreen(self.window.hwnd, (left, top))
        self.window = WindowRect(
            hwnd=self.window.hwnd,
            title=self.window.title,
            left=x,
            top=y,
            width=width,
            height=height,
        )
        return self.window

    def is_foreground(self) -> bool:
        if self.window is None or win32gui is None:
            return False
        return win32gui.GetForegroundWindow() == self.window.hwnd

    def grab(self, normalize: bool = True) -> Optional[np.ndarray]:
        rect = self.refresh_window()
        if rect is None:
            return None
        raw = np.array(self._sct.grab(rect.to_mss_monitor()))
        frame = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)
        if normalize and (rect.width != self.base_width or rect.height != self.base_height):
            frame = cv2.resize(
                frame,
                (self.base_width, self.base_height),
                interpolation=cv2.INTER_LINEAR,
            )
        return frame

    def to_screen_coords(self, x: int, y: int) -> Tuple[int, int]:
        if self.window is None:
            return x, y
        sx = int(self.window.left + x * (self.window.width / self.base_width))
        sy = int(self.window.top + y * (self.window.height / self.base_height))
        return sx, sy

    @staticmethod
    def mean_brightness(frame: np.ndarray) -> float:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return float(np.mean(gray))

    @staticmethod
    def frame_diff_score(frame_a: np.ndarray, frame_b: np.ndarray) -> float:
        small_a = cv2.resize(cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY), (160, 90))
        small_b = cv2.resize(cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY), (160, 90))
        diff = cv2.absdiff(small_a, small_b)
        return float(np.mean(diff))

    def close(self) -> None:
        try:
            self._sct.close()
        except Exception:
            pass
