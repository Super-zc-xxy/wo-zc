"""
模板一键采集工具（1920x1080 归一化画面）。
运行方式：
    python -m tools.capture_template --name btn_skip.png
弹出当前游戏画面后，用鼠标框选目标图标，按 Enter / Space 确认保存，按 c 取消。
"""
import argparse
import sys
from pathlib import Path

import cv2
import yaml

from core.capture import ScreenCapture


def main() -> int:
    parser = argparse.ArgumentParser(description="鸣潮 UI 模板采集工具")
    parser.add_argument(
        "--name",
        required=True,
        help="保存的模板文件名，例如 btn_skip.png / icon_f.png / quest_marker.png",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="配置文件路径（默认 config.yaml）",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent.parent
    cfg_path = base_dir / args.config
    with cfg_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    win_cfg = config.get("window", {})
    capture = ScreenCapture(
        title_keywords=win_cfg.get("title_keywords", ["鸣潮", "Wuthering Waves"]),
        base_width=int(win_cfg.get("base_width", 1920)),
        base_height=int(win_cfg.get("base_height", 1080)),
    )

    rect = capture.find_window()
    if rect is None:
        print("未找到游戏窗口，请先以窗口化启动鸣潮。")
        return 1

    frame = capture.grab(normalize=True)
    capture.close()
    if frame is None:
        print("截屏失败。")
        return 1

    print("请在弹出的窗口中用鼠标框选模板区域，按 Enter/Space 确认，按 c 取消。")
    x, y, w, h = cv2.selectROI("Capture Template (1920x1080)", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()

    if w <= 0 or h <= 0:
        print("已取消采集。")
        return 0

    crop = frame[y : y + h, x : x + w]
    out_dir = base_dir / "assets"
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = args.name if args.name.endswith(".png") else f"{args.name}.png"
    out_path = out_dir / filename
    cv2.imwrite(str(out_path), crop)
    print(f"已保存模板: {out_path} (size={w}x{h}, pos={x},{y})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
