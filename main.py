import argparse
import sys
import time
from pathlib import Path

import keyboard
import yaml

from core.bot_runner import BotRunner
from core.logger import setup_logger
from core.presets import PresetManager


def get_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def run_cli(base_dir: Path) -> int:
    config_path = base_dir / "config.yaml"
    if not config_path.exists():
        print(f"找不到配置文件: {config_path}")
        return 1

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    logger = setup_logger(config, base_dir=base_dir)
    preset_manager = PresetManager(base_dir / "presets")
    runner = BotRunner(base_dir, config, preset_manager, logger)
    runner.register_hotkeys()

    stop_key = str(config.get("hotkeys", {}).get("stop", "f9"))
    should_exit = False

    def on_stop():
        nonlocal should_exit
        should_exit = True
        logger.info("收到退出热键，正在停止...")

    keyboard.add_hotkey(stop_key, on_stop)
    logger.info("CLI 模式已就绪，按 [%s] 退出。", stop_key.upper())

    try:
        while not should_exit:
            time.sleep(0.15)
    except KeyboardInterrupt:
        logger.info("收到 Ctrl+C 中断。")
    finally:
        runner.stop()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="鸣潮自动任务助手 V2")
    parser.add_argument("--cli", action="store_true", help="以纯命令行模式运行（不启动图形界面）")
    args = parser.parse_args()

    base_dir = get_base_dir()
    if args.cli:
        return run_cli(base_dir)

    from gui_app import launch_gui

    return launch_gui(base_dir)


if __name__ == "__main__":
    sys.exit(main())
