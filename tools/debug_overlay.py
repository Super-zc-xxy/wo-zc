"""
实时可视化调试工具：
在独立窗口中预览 1920x1080 归一化画面，绘制 config.yaml 中定义的 ROI 区域、
任务黄标定位点以及左上角任务栏 OCR 结果，按 q 退出。
运行方式：
    python -m tools.debug_overlay
"""
import sys
from pathlib import Path

import cv2
import yaml

from core.capture import ScreenCapture
from core.vision import VisionEngine


def main() -> int:
    base_dir = Path(__file__).resolve().parent.parent
    with (base_dir / "config.yaml").open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    win_cfg = config.get("window", {})
    vis_cfg = config.get("vision", {})
    rois = config.get("rois", {})

    capture = ScreenCapture(
        title_keywords=win_cfg.get("title_keywords", ["鸣潮", "Wuthering Waves"]),
        base_width=int(win_cfg.get("base_width", 1920)),
        base_height=int(win_cfg.get("base_height", 1080)),
    )
    vision = VisionEngine(
        assets_dir=str(base_dir / "assets"),
        default_threshold=float(vis_cfg.get("default_threshold", 0.80)),
        ocr_interval_sec=float(vis_cfg.get("ocr_interval_sec", 1.0)),
    )

    if capture.find_window() is None:
        print("未找到游戏窗口，请先启动鸣潮。")
        return 1

    print("调试预览已启动，按 q 退出。")
    try:
        while True:
            frame = capture.grab(normalize=True)
            if frame is None:
                continue

            canvas = frame.copy()
            # 1. 绘制所有 ROI
            for name, rect in rois.items():
                x1, y1, x2, y2 = rect
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 220, 255), 2)
                cv2.putText(
                    canvas,
                    name,
                    (x1 + 4, max(18, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 220, 255),
                    1,
                    cv2.LINE_AA,
                )

            # 2. 绘制任务黄标检测结果
            center_roi = rois.get("combat_center", [400, 80, 1520, 850])
            marker = vision.detect_yellow_quest_marker(frame, roi=center_roi)
            if marker is not None:
                cv2.circle(canvas, marker, 14, (0, 255, 0), 3)
                cv2.putText(
                    canvas,
                    f"marker {marker}",
                    (marker[0] + 16, marker[1]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

            # 3. 左上角任务栏 OCR
            quest_roi = rois.get("quest_panel", [40, 190, 440, 360])
            lines = vision.ocr_roi(frame, quest_roi, cache_key="debug_quest")
            text_preview = " | ".join(line.text for line in lines)
            if text_preview:
                print(f"\r[OCR] {text_preview[:80]:<80}", end="", flush=True)

            # 缩小显示以防遮挡游戏窗口
            preview = cv2.resize(canvas, (960, 540))
            cv2.imshow("ww_auto_quest Debug Overlay (press q to quit)", preview)
            if (cv2.waitKey(100) & 0xFF) == ord("q"):
                break
    finally:
        capture.close()
        cv2.destroyAllWindows()
        print("\n已关闭调试预览。")

    return 0


if __name__ == "__main__":
    sys.exit(main())
