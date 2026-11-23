import ctypes
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.capture import ScreenCapture
from core.input_ctrl import SCAN_CODES


class EnvironmentDoctor:
    """
    鸣潮任务自动化助手：一键环境健康诊断体检医生。
    全面体检系统环境、游戏窗口、DPI 缩放、图形性能、OCR 依赖、AI 网关及本地预设完整度。
    """

    def __init__(self, root_dir: Optional[Path] = None) -> None:
        self.root_dir = root_dir or ROOT
        self.config_path = self.root_dir / "config.yaml"
        self.config = {}
        if self.config_path.exists():
            with open(self.config_path, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f) or {}

    def run_all_checks(self) -> List[Tuple[bool, str, str]]:
        """
        运行全部系统与环境诊断。
        返回列表项：(是否通过, 诊断项名称, 详细信息或建议)。
        """
        results: List[Tuple[bool, str, str]] = []

        # 1. 操作系统与权限
        is_admin = False
        try:
            is_admin = bool(ctypes.windll.shell32.IsUserAnAdmin())
        except Exception:
            pass
        results.append((
            True,
            "操作系统与运行权限",
            f"Windows 环境 | 管理员特权: {'已获取 (推荐)' if is_admin else '普通权限 (若前台游戏有提权建议右键管理员运行)'}",
        ))

        # 2. 屏幕与 DPI 缩放检查
        user32 = getattr(ctypes, "windll", None).user32 if hasattr(ctypes, "windll") else None
        if user32:
            try:
                # 获取主屏幕真实物理分辨率
                user32.SetProcessDPIAware()
                sw = user32.GetSystemMetrics(0)
                sh = user32.GetSystemMetrics(1)
                dpi_ok = (sw >= 1920 and sh >= 1080)
                results.append((
                    dpi_ok,
                    "主显示器分辨率与 DPI",
                    f"当前物理分辨率: {sw}x{sh} | {'推荐基准 1080p 及以上' if dpi_ok else '分辨率低于 1080p 可能影响视觉匹配'}",
                ))
            except Exception as e:
                results.append((False, "屏幕分辨率检测", f"检测失败: {e}"))

        # 3. 游戏窗口检测
        cap = ScreenCapture(["鸣潮", "Wuthering Waves"])
        wnd = cap.find_window()
        if wnd:
            results.append((
                True,
                "鸣潮游戏窗口绑定",
                f"已检测到游戏窗口: [{wnd.title}] 尺寸: {wnd.width}x{wnd.height} 句柄: {wnd.hwnd}",
            ))
        else:
            results.append((
                False,
                "鸣潮游戏窗口绑定",
                "未检测到运行中的《鸣潮》客户端窗口 (启动游戏并进入大世界后将自动绑定)",
            ))

        # 4. 高速截屏与内存零拷贝视图性能
        t0 = time.perf_counter()
        dummy_img = cap.grab(normalize=True)
        grab_ms = (time.perf_counter() - t0) * 1000
        cap.close()
        perf_ok = grab_ms < 25.0
        results.append((
            perf_ok,
            "画面截取引擎 (零拷贝视图)",
            f"单帧采集耗时: {grab_ms:.2f}ms | {'超低延迟高效运作' if perf_ok else '截屏耗时稍高，建议关闭其他高负载应用'}",
        ))

        # 5. RapidOCR 与 ONNX Runtime 推理库
        ocr_ready = False
        try:
            from core.vision import VisionEngine
            ve = VisionEngine(ocr_interval_sec=0.0)
            if ve._ocr is not None:
                ocr_ready = True
        except Exception:
            pass
        results.append((
            ocr_ready,
            "本地深度学习文字识别 (RapidOCR)",
            "ONNX 推理引擎已装载，支持全自动零模板文字直驱识别" if ocr_ready else "未检测到 RapidOCR 引擎 (将回退至纯模板匹配模式)",
        ))

        # 6. Win32 DirectInput 扫描码完整度
        key_count = len(SCAN_CODES)
        results.append((
            key_count >= 20,
            "硬件级键鼠模拟驱动 (DirectInput)",
            f"已就绪 {key_count} 个按键扫描码 (支持 W/A/S/D/F/T/J/M/Q/E/R/Space/Tab/Shift 等全功能操控)",
        ))

        # 7. AI 托管与多模态服务配置
        ai_cfg = self.config.get("ai_autopilot", {})
        ai_prov = ai_cfg.get("provider", "gemini")
        ai_url = ai_cfg.get("base_url", "")
        ai_key = ai_cfg.get("api_key", "")
        has_key = bool(ai_key or ai_prov == "ollama")
        results.append((
            has_key,
            "多模态大模型服务配置",
            f"服务商: {ai_prov} | 接口地址: {ai_url or '默认地址'} | 密钥状态: {'已配置 (支持疑难会诊与全权接管)' if has_key else '未配置 (仅在纯规则模式运行)'}",
        ))

        # 8. 核心配队预设与经验记忆库检查
        presets_dir = self.root_dir / "presets"
        preset_count = len(list(presets_dir.glob("*.yaml"))) if presets_dir.exists() else 0
        exp_file = self.root_dir / "data" / "experience_db.json"
        results.append((
            preset_count >= 4,
            "配队预设与自学习知识库",
            f"已载入 {preset_count} 套配队预设 | 经验记忆库: {'已初始化自学习' if exp_file.exists() else '就绪待生成'}",
        ))

        return results

    def print_report(self) -> bool:
        """在控制台打印美观的系统体检诊断报告，返回全部是否达标。"""
        print("=" * 66)
        print("    ✦ MingchaoJB V2 - 鸣潮自动化助手环境体检健康诊断 ✦")
        print("=" * 66)
        checks = self.run_all_checks()
        all_passed = True
        for ok, name, detail in checks:
            mark = "✔ [正常]" if ok else "✖ [警告]"
            print(f"{mark} {name:<26} : {detail}")
            if not ok:
                all_passed = False
        print("-" * 66)
        print("体检诊断结论:", "【环境完全就绪，可流畅运行所有自动化功能】" if all_passed else "【存在部分建议项，请参考详情进行调优】")
        print("=" * 66)
        return all_passed


def main() -> None:
    doc = EnvironmentDoctor()
    doc.print_report()


if __name__ == "__main__":
    main()
