import copy
import os
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml

from core.bot_runner import BotRunner
from core.logger import setup_logger
from core.presets import PresetManager

# ==============================================================================
# 鸣潮极简浅色科技主题调色盘（Clean Light & Resonator Amber Gold）
# 纯净白昼、层次丰富、高对比度与呼吸感交互
# ==============================================================================
THEME = {
    "bg_main": "#f1f5f9",         # 窗口最底层背景（清爽淡雅浅蓝灰 Slate 100）
    "bg_card": "#ffffff",         # 主卡片与容器纯白背景（White 50）
    "bg_elevated": "#f8fafc",     # 输入框、列表等次级微灰背景（Slate 50）
    "bg_hover": "#e2e8f0",        # 按钮悬停与交互浅灰底色（Slate 200）
    "border": "#e2e8f0",          # 极轻盈细腻边框线
    "border_light": "#cbd5e1",    # 悬停与聚焦边框（Slate 300）
    "gold": "#b45309",            # 鸣潮琥珀金（深金，浅色底极高清晰度与品牌感）
    "gold_hover": "#92400e",      # 悬停深琥珀金
    "gold_dark": "#78350f",       # 按钮深金底
    "gold_bg": "#fef3c7",         # 浅金胶囊底色（激活高亮）
    "gold_border": "#fde68a",     # 浅金胶囊边框
    "green": "#059669",           # 运行/就绪翡翠绿
    "green_bg": "#d1fae5",        # 翠绿胶囊底色
    "red": "#dc2626",             # 停止/告警鲜亮红
    "red_bg": "#fee2e2",          # 警示浅红底色
    "blue": "#0284c7",            # 科技信息天蓝
    "blue_bg": "#e0f2fe",         # 科技蓝浅胶囊底色
    "purple": "#7c3aed",          # 任务标识紫
    "purple_bg": "#ede9fe",       # 任务浅紫底色
    "text_primary": "#0f172a",    # 主标题与正文黑（Slate 900）
    "text_secondary": "#475569",  # 次级属性与说明字（Slate 600）
    "text_muted": "#94a3b8",      # 辅助提示与占位字（Slate 400）
}

FEATURE_LABELS = [
    ("auto_dialogue", "自动跳过剧情与对话选项"),
    ("auto_combat", "自动战斗与队伍轮切"),
    ("auto_navigate", "自动 3D 任务标寻路与脱困"),
    ("auto_teleport", "远距离大地图自动传送"),
    ("auto_switch_quest", "任务受阻时自动按 J 切换下一任务"),
    ("auto_time_adjust", "任务要求时自动调整游戏时间"),
    ("auto_absorb_echo", "战后自动寻径吸收声骸与拾取"),
    ("auto_heal", "濒血自动使用快捷回复药剂 [C]"),
    ("auto_popup_close", "自动清理弹窗/就近复苏/物品交付"),
    ("auto_gadget", "自动响应钩索/探索工具 [T]"),
]

COMMON_TEMPLATES = [
    ("btn_skip.png", "左上角剧情跳过按钮（支持自动收集）"),
    ("btn_confirm.png", "跳过二次确认按钮（支持自动收集）"),
    ("dialogue_option.png", "右侧对话选项气泡"),
    ("icon_auto_play.png", "左上角自动播放图标（支持自动收集）"),
    ("icon_f.png", "中右侧 [F] 交互提示（支持自动收集）"),
    ("icon_t.png", "屏幕下方 [T] 钩索/工具提示（支持自动收集）"),
    ("quest_marker.png", "任务追踪菱形黄标（支持自动收集）"),
    ("in_combat.png", "右下角战斗状态标识"),
    ("icon_concerto_ready.png", "右侧头像协奏变奏就绪高亮"),
    ("icon_ult_ready.png", "右下角大招就绪图标"),
    ("icon_beacon.png", "大地图信标图标（支持自动收集）"),
    ("btn_teleport.png", "大地图右下角快速旅行按钮（支持自动收集）"),
    ("btn_track.png", "任务列表右下角追踪按钮（支持自动收集）"),
    ("btn_close.png", "弹窗关闭按钮"),
    ("btn_revive.png", "团灭复苏按钮（支持自动收集）"),
]


def create_custom_scrolled_text(parent: tk.Widget, **kwargs) -> Tuple[tk.Frame, tk.Text]:
    """创建内置浅色优雅滚动条的现代代码/日志文本控件。"""
    frame = tk.Frame(parent, bg=THEME["bg_card"])
    txt = tk.Text(
        frame,
        bg=kwargs.get("bg", "#f8fafc"),
        fg=kwargs.get("fg", "#1e293b"),
        insertbackground=THEME["text_primary"],
        selectbackground="#fde68a",
        selectforeground="#78350f",
        relief="flat",
        highlightthickness=1,
        highlightbackground=THEME["border"],
        highlightcolor=THEME["gold"],
        wrap=kwargs.get("wrap", tk.WORD),
        font=kwargs.get("font", ("Consolas", 9)),
        height=kwargs.get("height", 10),
        state=kwargs.get("state", tk.NORMAL),
    )
    sb = ttk.Scrollbar(frame, orient="vertical", command=txt.yview, style="Vertical.TScrollbar")
    txt.configure(yscrollcommand=sb.set)
    txt.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    sb.pack(side=tk.RIGHT, fill=tk.Y)
    return frame, txt


# 保持函数兼容别名
create_dark_scrolled_text = create_custom_scrolled_text


class MingchaoGUI:
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = Path(base_dir)
        self.config_path = self.base_dir / "config.yaml"
        self.config = self._load_config()

        self.log_queue: "queue.Queue[tuple[str, str]]" = queue.Queue()
        self.logger = setup_logger(self.config, base_dir=self.base_dir, log_queue=self.log_queue)
        self.preset_manager = PresetManager(self.base_dir / "presets")

        self.root = tk.Tk()
        self.root.title("MingchaoJB V2 - 鸣潮桌面任务助手")
        self.root.geometry("1100x760")
        self.root.minsize(980, 660)
        self.root.configure(bg=THEME["bg_main"])

        # 状态变量
        self.status_vars = {
            "running": tk.StringVar(value="○ 已暂停"),
            "state": tk.StringVar(value="IDLE"),
            "window": tk.StringVar(value="未绑定（请启动鸣潮）"),
            "preset": tk.StringVar(value=""),
            "quest": tk.StringVar(value="暂无识别到的任务文本"),
            "templates": tk.StringVar(value="0 个已就绪（零模板可直接运行）"),
            "ai_mode": tk.StringVar(value="规则模式"),
            "ai_reason": tk.StringVar(value="就绪中"),
        }
        self.auto_collect_var = tk.BooleanVar(
            value=bool(self.config.get("vision", {}).get("auto_collect_templates", True))
        )
        self.ai_mode_var = tk.BooleanVar(
            value=bool(self.config.get("ai_autopilot", {}).get("enabled", False))
        )

        # AI 配置变量
        ai_cfg = self.config.get("ai_autopilot", {})
        self.ai_provider_var = tk.StringVar(value=str(ai_cfg.get("provider", "gemini")))
        self.ai_model_var = tk.StringVar(value=str(ai_cfg.get("model", "gemini-2.5-flash")))
        self.ai_key_var = tk.StringVar(value=str(ai_cfg.get("api_key", "")))
        self.ai_base_url_var = tk.StringVar(value=str(ai_cfg.get("base_url", "https://generativelanguage.googleapis.com/v1beta")))
        self.ai_resolution_var = tk.StringVar(value="640x360 (极速推荐, <35KB)")
        self.ai_quality_var = tk.StringVar(value=str(ai_cfg.get("jpeg_quality", 65)))
        self.ai_timeout_var = tk.StringVar(value=str(ai_cfg.get("timeout_sec", 2.2)))
        self.ai_test_status_var = tk.StringVar(value="点击“测试连接并测速”可即时检测大模型通信延迟。")
        self._show_key = False

        self.runner = BotRunner(
            base_dir=self.base_dir,
            config=self.config,
            preset_manager=self.preset_manager,
            logger=self.logger,
            on_status_update=self._on_runner_status,
        )
        self.runner.register_hotkeys()

        self._current_editing_preset_id: str = str(self.config.get("active_preset", "default_full_quest"))
        self._feature_vars: dict[str, tk.BooleanVar] = {}

        self._setup_styles()
        self._build_ui()
        self._refresh_preset_lists()
        self._load_preset_into_editor(self._current_editing_preset_id)
        self._refresh_template_status()
        self._poll_logs()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _load_config(self) -> dict:
        with self.config_path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _save_config(self) -> None:
        with self.config_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(self.config, f, allow_unicode=True, sort_keys=False)

    def _setup_styles(self) -> None:
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass

        style.configure(".", background=THEME["bg_main"], foreground=THEME["text_primary"], font=("Microsoft YaHei UI", 9))
        style.configure("TFrame", background=THEME["bg_main"])
        style.configure("Card.TFrame", background=THEME["bg_card"])
        style.configure("Elevated.TFrame", background=THEME["bg_elevated"])

        # LabelFrame 分组框
        style.configure(
            "TLabelframe",
            background=THEME["bg_card"],
            bordercolor=THEME["border"],
            darkcolor=THEME["border"],
            lightcolor=THEME["border"],
            borderwidth=1,
            relief="solid",
        )
        style.configure(
            "TLabelframe.Label",
            background=THEME["bg_card"],
            foreground=THEME["gold"],
            font=("Microsoft YaHei UI", 9, "bold"),
        )

        # Labels 标签
        style.configure("TLabel", background=THEME["bg_main"], foreground=THEME["text_primary"])
        style.configure("Card.TLabel", background=THEME["bg_card"], foreground=THEME["text_primary"])
        style.configure("CardMuted.TLabel", background=THEME["bg_card"], foreground=THEME["text_muted"], font=("Microsoft YaHei UI", 8))
        style.configure("CardValue.TLabel", background=THEME["bg_card"], foreground=THEME["text_primary"], font=("Microsoft YaHei UI", 10, "bold"))
        style.configure("BannerTitle.TLabel", background=THEME["bg_card"], foreground=THEME["text_primary"], font=("Microsoft YaHei UI", 13, "bold"))
        style.configure("BannerBrand.TLabel", background=THEME["bg_card"], foreground=THEME["gold"], font=("Segoe UI", 14, "bold"))
        style.configure("Badge.TLabel", background=THEME["blue_bg"], foreground=THEME["blue"], font=("Microsoft YaHei UI", 8, "bold"), padding=(6, 2))

        # 普通按钮
        style.configure(
            "TButton",
            background=THEME["bg_card"],
            foreground=THEME["text_primary"],
            bordercolor=THEME["border"],
            darkcolor=THEME["border"],
            lightcolor=THEME["border"],
            font=("Microsoft YaHei UI", 9),
            padding=(10, 5),
            relief="flat",
        )
        style.map(
            "TButton",
            background=[("active", THEME["bg_hover"]), ("pressed", THEME["border_light"])],
            foreground=[("active", THEME["text_primary"])],
            bordercolor=[("active", THEME["border_light"])],
        )

        # 核心主按钮（待机金色 - 浅色底高亮）
        style.configure(
            "Primary.TButton",
            background="#d97706",
            foreground="#ffffff",
            bordercolor="#b45309",
            darkcolor="#b45309",
            lightcolor="#b45309",
            font=("Microsoft YaHei UI", 10, "bold"),
            padding=(16, 6),
            relief="flat",
        )
        style.map(
            "Primary.TButton",
            background=[("active", "#b45309"), ("pressed", "#92400e")],
            foreground=[("active", "#ffffff")],
        )

        # 核心主按钮（运行中红色）
        style.configure(
            "Running.TButton",
            background=THEME["red"],
            foreground="#ffffff",
            bordercolor="#b91c1c",
            darkcolor="#b91c1c",
            lightcolor="#b91c1c",
            font=("Microsoft YaHei UI", 10, "bold"),
            padding=(16, 6),
            relief="flat",
        )
        style.map(
            "Running.TButton",
            background=[("active", "#dc2626"), ("pressed", "#991b1b")],
            foreground=[("active", "#ffffff")],
        )

        # AI 胶囊切换按钮样式
        style.configure(
            "AIOff.TButton",
            background=THEME["bg_elevated"],
            foreground=THEME["text_secondary"],
            bordercolor=THEME["border"],
            font=("Microsoft YaHei UI", 9),
            padding=(12, 4),
        )
        style.map("AIOff.TButton", background=[("active", THEME["bg_hover"])])

        style.configure(
            "AIOn.TButton",
            background=THEME["gold_bg"],
            foreground=THEME["gold"],
            bordercolor=THEME["gold_border"],
            font=("Microsoft YaHei UI", 9, "bold"),
            padding=(12, 4),
        )
        style.map("AIOn.TButton", background=[("active", "#fde68a")])

        # 模板自动收集胶囊按钮（取代丑陋的 Checkbutton 方块）
        style.configure(
            "ToggleOn.TButton",
            background=THEME["green_bg"],
            foreground=THEME["green"],
            bordercolor=THEME["green"],
            font=("Microsoft YaHei UI", 9, "bold"),
            padding=(10, 4),
        )
        style.map("ToggleOn.TButton", background=[("active", "#a7f3d0")])

        style.configure(
            "ToggleOff.TButton",
            background=THEME["bg_elevated"],
            foreground=THEME["text_muted"],
            bordercolor=THEME["border"],
            font=("Microsoft YaHei UI", 9),
            padding=(10, 4),
        )
        style.map("ToggleOff.TButton", background=[("active", THEME["bg_hover"])])

        # 隐藏系统原生丑陋灰方块 Tab 头部，使用顶层现代 Segmented 胶囊导航栏代替
        style.configure("TNotebook", background=THEME["bg_main"], borderwidth=0)
        style.layout("TNotebook.Tab", [])

        # 输入框与下拉框
        style.configure(
            "TEntry",
            fieldbackground=THEME["bg_card"],
            foreground=THEME["text_primary"],
            bordercolor=THEME["border"],
            darkcolor=THEME["border"],
            lightcolor=THEME["border"],
            insertcolor=THEME["text_primary"],
            padding=5,
        )
        style.configure(
            "TCombobox",
            fieldbackground=THEME["bg_card"],
            background=THEME["bg_card"],
            foreground=THEME["text_primary"],
            bordercolor=THEME["border"],
            arrowcolor=THEME["text_secondary"],
            darkcolor=THEME["border"],
            lightcolor=THEME["border"],
            padding=4,
        )
        style.map(
            "TCombobox",
            fieldbackground=[("readonly", THEME["bg_card"])],
            selectbackground=[("readonly", THEME["bg_hover"])],
            selectforeground=[("readonly", THEME["text_primary"])],
        )

        # 浅色优雅垂直滚动条
        style.configure(
            "Vertical.TScrollbar",
            background=THEME["border_light"],
            troughcolor=THEME["bg_elevated"],
            bordercolor=THEME["bg_elevated"],
            arrowcolor=THEME["text_secondary"],
            relief="flat",
        )
        style.map(
            "Vertical.TScrollbar",
            background=[("active", THEME["text_muted"])],
            arrowcolor=[("active", THEME["gold"])],
        )

        # Treeview 表格
        style.configure(
            "Treeview",
            background=THEME["bg_card"],
            fieldbackground=THEME["bg_card"],
            foreground=THEME["text_primary"],
            bordercolor=THEME["border"],
            rowheight=28,
            font=("Microsoft YaHei UI", 9),
        )
        style.configure(
            "Treeview.Heading",
            background=THEME["bg_elevated"],
            foreground=THEME["text_primary"],
            bordercolor=THEME["border"],
            font=("Microsoft YaHei UI", 9, "bold"),
            padding=(6, 4),
        )
        style.map(
            "Treeview",
            background=[("selected", THEME["gold_bg"])],
            foreground=[("selected", THEME["gold"])],
        )
        style.map(
            "Treeview",
            background=[("selected", THEME["bg_hover"])],
            foreground=[("selected", THEME["gold"])],
        )

    def _build_ui(self) -> None:
        # ================= 顶部品牌与控制栏 =================
        header = tk.Frame(self.root, bg=THEME["bg_card"], highlightthickness=1, highlightbackground=THEME["border"], padx=16, pady=12)
        header.pack(fill=tk.X, padx=12, pady=(10, 8))

        # 左侧：品牌 Logo 与标签
        brand_left = tk.Frame(header, bg=THEME["bg_card"])
        brand_left.pack(side=tk.LEFT)

        ttk.Label(brand_left, text="✦ MingchaoJB", style="BannerBrand.TLabel").pack(side=tk.LEFT)
        ttk.Label(brand_left, text=" 自动任务助手 V2", style="BannerTitle.TLabel").pack(side=tk.LEFT, padx=(4, 8))
        ttk.Label(brand_left, text="Windows 1080P 就绪", style="Badge.TLabel").pack(side=tk.LEFT)

        # 中部：快捷键指示与 AI 模式胶囊切换
        hotkey_box = tk.Frame(header, bg=THEME["bg_card"])
        hotkey_box.pack(side=tk.LEFT, padx=(24, 0))
        ttk.Label(hotkey_box, text="[F8] 启停  [F7] 恢复", style="CardMuted.TLabel").pack(side=tk.LEFT)

        init_ai_style = "AIOn.TButton" if self.ai_mode_var.get() else "AIOff.TButton"
        init_ai_text = "✦ AI 托管: 运行中" if self.ai_mode_var.get() else "✦ AI 托管: 已停用"
        self.btn_ai_toggle = ttk.Button(
            hotkey_box,
            text=init_ai_text,
            style=init_ai_style,
            command=self._toggle_ai_mode_btn,
        )
        self.btn_ai_toggle.pack(side=tk.LEFT, padx=(12, 0))

        # 右侧：主操作按钮组（呼吸感间距，文字不截断）
        btn_box = tk.Frame(header, bg=THEME["bg_card"])
        btn_box.pack(side=tk.RIGHT)

        self.btn_toggle = ttk.Button(btn_box, text="启动脚本 (F8)", command=self.runner.toggle, style="Primary.TButton", width=14)
        self.btn_toggle.pack(side=tk.RIGHT, padx=(8, 0))

        ttk.Button(btn_box, text="接管恢复 (F7)", command=self.runner.resume_takeover, width=13).pack(side=tk.RIGHT, padx=4)
        ttk.Button(btn_box, text="一键自动校准", command=self._run_auto_calibrate, width=13).pack(side=tk.RIGHT, padx=4)

        # ================= 现代高质感流线型导航栏（告别死板灰方块） =================
        tab_nav_bar = tk.Frame(self.root, bg=THEME["bg_main"])
        tab_nav_bar.pack(fill=tk.X, padx=12, pady=(0, 6))

        # 左侧：通透现代卡片 Tab 容器（无灰底槽、无粗暴硬黑框）
        self.nav_tabs_container = tk.Frame(tab_nav_bar, bg=THEME["bg_main"])
        self.nav_tabs_container.pack(side=tk.LEFT)

        self.nav_tabs = [
            ("📊 控制台看板", 0),
            ("✦ AI 托管配置", 1),
            ("🎛️ 预设管理", 2),
            ("📜 日志查看", 3),
            ("🛠️ 模板与调试工具", 4),
        ]
        self.nav_tab_items: List[dict] = []
        self.nav_buttons: List[tk.Widget] = []

        for title, idx in self.nav_tabs:
            # 单个 Tab 独立悬浮卡片容器
            card = tk.Frame(
                self.nav_tabs_container,
                bg=THEME["bg_main"],
                cursor="hand2",
                padx=0,
                pady=0,
            )
            card.pack(side=tk.LEFT, padx=(0, 6))

            # 内部内容 Label（高质感字体、舒适呼吸内边距）
            lbl = tk.Label(
                card,
                text=title,
                font=("Microsoft YaHei UI", 9),
                bg=THEME["bg_main"],
                fg=THEME["text_secondary"],
                padx=16,
                pady=7,
                cursor="hand2",
            )
            lbl.pack(fill=tk.X)

            # 底部 3px 专属品牌高光指示条（鸣潮琥珀金，未选中时与背景融合隐形）
            indicator = tk.Frame(card, height=3, bg=THEME["bg_main"], cursor="hand2")
            indicator.pack(fill=tk.X, side=tk.BOTTOM)

            # 全区域穿透事件绑定（点击任意位置均平滑切页，无事件死角）
            for widget in (card, lbl, indicator):
                widget.bind("<Button-1>", lambda e, i=idx: self._select_nav_tab(i))
                widget.bind("<Enter>", lambda e, i=idx: self._on_nav_item_hover(i, True))
                widget.bind("<Leave>", lambda e, i=idx: self._on_nav_item_hover(i, False))

            item_data = {
                "card": card,
                "label": lbl,
                "indicator": indicator,
                "index": idx,
                "title": title,
            }
            self.nav_tab_items.append(item_data)
            self.nav_buttons.append(lbl)

        # 右侧：快捷键提示与一键体检医生入口
        right_box = tk.Frame(tab_nav_bar, bg=THEME["bg_main"])
        right_box.pack(side=tk.RIGHT, pady=2)

        # 一键系统体检医生按钮（精致高亮小胶囊）
        self.btn_doctor = tk.Button(
            right_box,
            text="🩺 一键环境体检",
            command=self._run_doctor_check,
            font=("Microsoft YaHei UI", 8, "bold"),
            bg=THEME["bg_card"],
            fg=THEME["gold"],
            activebackground=THEME["gold_bg"],
            activeforeground=THEME["gold_dark"],
            relief="solid",
            bd=1,
            padx=10,
            pady=3,
            cursor="hand2",
        )
        self.btn_doctor.pack(side=tk.RIGHT, padx=(8, 0))

        tip_lbl = tk.Label(
            right_box,
            text="💡 [F8] 启停任务 · [F7] 接管恢复",
            bg=THEME["bg_main"],
            fg=THEME["text_muted"],
            font=("Microsoft YaHei UI", 8),
        )
        tip_lbl.pack(side=tk.RIGHT, pady=2)

        # ================= 中间多标签面板 =================
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 10))
        self.notebook.bind("<<NotebookTabChanged>>", self._on_notebook_tab_changed)

        self._build_dashboard_tab()
        self._build_ai_config_tab()   # 醒目的专属 AI 托管配置页！
        self._build_presets_tab()
        self._build_logs_tab()
        self._build_tools_tab()

        self._sync_nav_tab_highlight(0)

    def _select_nav_tab(self, index: int) -> None:
        """用户点击自定义导航 Tab，平滑切换页面并高亮选中态。"""
        self.notebook.select(index)
        self._sync_nav_tab_highlight(index)

    def _sync_nav_tab_highlight(self, selected_index: int) -> None:
        """
        优雅高亮选中态：
        - 选中项：纯白精致浮雕微卡片（#ffffff），深琥珀金文字加粗（#b45309），底部 3px 纯正琥珀金高光线！
        - 若是 AI 托管配置，辅以浅金星芒微光背景。
        - 未选中项：完全融入背景，无生硬灰边框，呈现低调克制的高级灰。
        """
        for item in self.nav_tab_items:
            idx = item["index"]
            card = item["card"]
            lbl = item["label"]
            indicator = item["indicator"]

            if idx == selected_index:
                is_ai = ("AI" in item["title"])
                active_bg = "#fffdf5" if is_ai else "#ffffff"
                active_fg = THEME["gold"]

                card.config(
                    bg=active_bg,
                    highlightthickness=1,
                    highlightbackground=THEME["border_light"],
                )
                lbl.config(
                    bg=active_bg,
                    fg=active_fg,
                    font=("Microsoft YaHei UI", 9, "bold"),
                )
                indicator.config(bg=THEME["gold"])
            else:
                card.config(
                    bg=THEME["bg_main"],
                    highlightthickness=0,
                )
                lbl.config(
                    bg=THEME["bg_main"],
                    fg=THEME["text_secondary"],
                    font=("Microsoft YaHei UI", 9),
                )
                indicator.config(bg=THEME["bg_main"])

    def _on_nav_item_hover(self, index: int, is_enter: bool) -> None:
        """悬停微呼吸反馈：未选中的 Tab 鼠标悬停时平滑变为浅灰并浮现淡金色预热下划线。"""
        try:
            curr_selected = self.notebook.index(self.notebook.select())
            if index == curr_selected:
                return  # 当前选中项保持纯白高亮
        except Exception:
            return

        if index < 0 or index >= len(self.nav_tab_items):
            return

        item = self.nav_tab_items[index]
        card = item["card"]
        lbl = item["label"]
        indicator = item["indicator"]

        if is_enter:
            card.config(bg="#f8fafc", highlightthickness=1, highlightbackground=THEME["border"])
            lbl.config(bg="#f8fafc", fg=THEME["text_primary"])
            indicator.config(bg="#fed7aa")  # 浅暖金悬停预热线
        else:
            card.config(bg=THEME["bg_main"], highlightthickness=0)
            lbl.config(bg=THEME["bg_main"], fg=THEME["text_secondary"])
            indicator.config(bg=THEME["bg_main"])

    def _on_nav_btn_hover(self, btn: tk.Widget, index: int, is_enter: bool) -> None:
        """兼容旧版悬停方法签名。"""
        self._on_nav_item_hover(index, is_enter)

    def _on_notebook_tab_changed(self, _event=None) -> None:
        """监听底层 Notebook 切页事件，自动同步外部卡片 Tab 高亮。"""
        try:
            curr_idx = self.notebook.index(self.notebook.select())
            self._sync_nav_tab_highlight(curr_idx)
        except Exception:
            pass

    def _run_doctor_check(self) -> None:
        """一键运行系统与环境健康体检诊断，并在日志窗口与对话框反馈体检报告。"""
        self.notebook.select(3)  # 自动切到“日志查看”页展示完整输出
        self.logger.info("=" * 55)
        self.logger.info("🩺 开始执行鸣潮自动化环境与运行健康体检诊断...")

        def _do_check():
            try:
                from tools.doctor import EnvironmentDoctor
                doc = EnvironmentDoctor(self.base_dir)
                results = doc.run_all_checks()
                passed_count = sum(1 for ok, _, _ in results if ok)
                total_count = len(results)

                self.logger.info("-" * 55)
                for ok, title, detail in results:
                    status_flag = "✅ [正常]" if ok else "⚠️ [需注意]"
                    self.logger.info(f"{status_flag} {title}: {detail}")
                self.logger.info("-" * 55)
                summary_msg = f"体检完成: 共 {total_count} 项检测，{passed_count} 项通过，{total_count - passed_count} 项需注意。"
                self.logger.info(f"🩺 {summary_msg}")
                self.logger.info("=" * 55)

                # 主线程弹窗简要提示
                def _show_msg():
                    if passed_count == total_count:
                        messagebox.showinfo("环境体检正常", f"🎉 全部 {total_count} 项环境检测均已就绪！\n可直接启动鸣潮并按 [F8] 运行。")
                    else:
                        issues = [f"• {t}: {d[:45]}" for ok, t, d in results if not ok]
                        issue_text = "\n".join(issues)
                        messagebox.showwarning(
                            "环境体检建议",
                            f"🩺 检测到 {total_count - passed_count} 项配置需注意（详情请见日志）：\n\n{issue_text}"
                        )
                try:
                    if self.root and self.root.winfo_exists():
                        self.root.after(0, _show_msg)
                except Exception:
                    pass
            except Exception as ex:
                self.logger.error(f"体检执行异常: {ex}")

        threading.Thread(target=_do_check, daemon=True).start()

    def _build_dashboard_tab(self) -> None:
        tab = ttk.Frame(self.notebook, padding=12)
        self.notebook.add(tab, text=" 控制台看板 ")

        # 1. 六宫格数据指标看板 (2行 x 3列)
        grid_frame = tk.Frame(tab, bg=THEME["bg_main"])
        grid_frame.pack(fill=tk.X, pady=(0, 10))
        grid_frame.columnconfigure((0, 1, 2), weight=1)

        cards = [
            ("运行状态", "running", THEME["green"]),
            ("当前状态机 / AI 意图", "state", THEME["gold"]),
            ("绑定游戏窗口", "window", THEME["blue"]),
            ("当前操作预设", "preset", THEME["text_primary"]),
            ("已收集 UI 模板", "templates", THEME["text_primary"]),
            ("任务栏 OCR 识别", "quest", THEME["purple"]),
        ]

        for idx, (title, key, highlight_color) in enumerate(cards):
            r, c = divmod(idx, 3)
            card = tk.Frame(
                grid_frame,
                bg=THEME["bg_card"],
                highlightthickness=1,
                highlightbackground=THEME["border"],
                padx=14,
                pady=10,
            )
            card.grid(row=r, column=c, padx=5, pady=5, sticky=tk.NSEW)

            title_row = tk.Frame(card, bg=THEME["bg_card"])
            title_row.pack(fill=tk.X)
            ttk.Label(title_row, text=title, style="CardMuted.TLabel").pack(side=tk.LEFT)

            # 若是绑定游戏窗口卡片，在标题右侧增加一个即时重新检测按钮
            if key == "window":
                ttk.Button(title_row, text="🔄 重新检测", command=self._detect_game_window, width=9).pack(side=tk.RIGHT)

            val_lbl = tk.Label(
                card,
                textvariable=self.status_vars[key],
                bg=THEME["bg_card"],
                fg=highlight_color,
                font=("Microsoft YaHei UI", 10, "bold"),
                anchor=tk.W,
                justify=tk.LEFT,
            )
            val_lbl.pack(anchor=tk.W, fill=tk.X, pady=(4, 0))

        # 2. 快捷操作栏
        quick_frame = ttk.LabelFrame(tab, text="快捷预设与一键配置", padding=12)
        quick_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(quick_frame, text="快速选择预设：", style="Card.TLabel").pack(side=tk.LEFT)
        self.quick_preset_combo = ttk.Combobox(quick_frame, state="readonly", width=28)
        self.quick_preset_combo.pack(side=tk.LEFT, padx=6)
        ttk.Button(quick_frame, text="应用所选预设", command=self._apply_quick_preset).pack(side=tk.LEFT, padx=4)
        ttk.Button(quick_frame, text="一键校准全部调试框", command=self._run_auto_calibrate).pack(side=tk.LEFT, padx=4)

        # 醒目的 AI 配置直达入口！
        ttk.Button(quick_frame, text="⚙ 配置 AI 大模型", command=lambda: self.notebook.select(1)).pack(side=tk.LEFT, padx=8)

        # 优雅的胶囊切换按钮（彻底取代系统 Checkbutton 乱码方块）
        init_ac_text = "✔ 自动收集模板: 已开启" if self.auto_collect_var.get() else "○ 自动收集模板: 已关闭"
        init_ac_style = "ToggleOn.TButton" if self.auto_collect_var.get() else "ToggleOff.TButton"
        self.btn_auto_collect = ttk.Button(
            quick_frame,
            text=init_ac_text,
            style=init_ac_style,
            command=self._on_toggle_auto_collect_btn,
        )
        self.btn_auto_collect.pack(side=tk.RIGHT)

        # 2.5 🧠 L5 完全自主意图与时序思考驾驶舱
        l5_frame = ttk.LabelFrame(tab, text="🧠 L5 完全自主大脑 · 长程意图与时序思考驾驶舱", padding=10)
        l5_frame.pack(fill=tk.X, pady=(0, 10))

        intent_row = tk.Frame(l5_frame, bg=THEME["bg_card"])
        intent_row.pack(fill=tk.X, pady=(0, 6))

        ttk.Label(intent_row, text="🎯 L5 全局大目标：", style="Card.TLabel", font=("Microsoft YaHei UI", 9, "bold")).pack(side=tk.LEFT)
        self.l5_intent_var = tk.StringVar(value="默认自主托管（日常与探索）")
        self.l5_intent_combo = ttk.Combobox(
            intent_row,
            textvariable=self.l5_intent_var,
            values=[
                "默认自主托管（日常与探索）",
                "全自动清空结晶溶剂体力与日常",
                "全大世界高价值金色声骸巡回猎杀",
                "【今汐】专属突破素材与声骸定向收集",
                "【忌炎】专属突破素材与声骸定向收集",
                "【椿】专属突破素材与声骸定向收集",
                "【长离】专属突破素材与声骸定向收集",
            ],
            width=36,
        )
        self.l5_intent_combo.pack(side=tk.LEFT, padx=6)
        ttk.Button(intent_row, text="下达大目标", command=self._apply_l5_master_goal).pack(side=tk.LEFT, padx=4)

        self.l5_status_lbl = tk.Label(
            intent_row,
            text="目标流转: 就绪 | 空间阻抗: 正常 | 经验库: 0ms回放",
            bg=THEME["bg_elevated"],
            fg=THEME["text_secondary"],
            font=("Microsoft YaHei UI", 8),
            padx=8,
            pady=3,
        )
        self.l5_status_lbl.pack(side=tk.RIGHT)

        thought_row = tk.Frame(l5_frame, bg=THEME["bg_elevated"], padx=8, pady=4)
        thought_row.pack(fill=tk.X)
        self.l5_thought_var = tk.StringVar(value="[就绪] 🧠 [L5思考] 已规划默认全局执行链路：日常委托 -> 体力清空 -> 大世界主支线 -> 声骸收益")
        tk.Label(
            thought_row,
            textvariable=self.l5_thought_var,
            bg=THEME["bg_elevated"],
            fg=THEME["gold"],
            font=("Consolas", 8, "bold"),
            anchor=tk.W,
            justify=tk.LEFT,
        ).pack(fill=tk.X)

        # 3. 控制台实时终端日志视窗（内置浅色定制优雅滚动条）
        log_frame = ttk.LabelFrame(tab, text="实时控制台日志（按 F8 启停 | 完整筛选请切换至“日志查看”页）", padding=8)
        log_frame.pack(fill=tk.BOTH, expand=True)

        sc_frame, self.mini_log_text = create_custom_scrolled_text(log_frame, height=12, state=tk.DISABLED)
        sc_frame.pack(fill=tk.BOTH, expand=True)
        self._setup_text_tags(self.mini_log_text)

    def _build_ai_config_tab(self) -> None:
        """独立的 AI 托管与大模型接入配置面板。"""
        tab = ttk.Frame(self.notebook, padding=12)
        self.notebook.add(tab, text=" ✦ AI 托管配置 ")

        # 1. 顶部说明横幅卡片
        desc_card = tk.Frame(tab, bg=THEME["bg_card"], highlightthickness=1, highlightbackground=THEME["border"], padx=14, pady=10)
        desc_card.pack(fill=tk.X, pady=(0, 10))

        top_row = tk.Frame(desc_card, bg=THEME["bg_card"])
        top_row.pack(fill=tk.X)
        ttk.Label(top_row, text="✦ AI 视觉多模态全权接管控制台", style="Card.TLabel", font=("Microsoft YaHei UI", 11, "bold")).pack(side=tk.LEFT)

        status_lbl = tk.Label(
            top_row,
            textvariable=self.status_vars["ai_mode"],
            bg=THEME["gold_bg"],
            fg=THEME["gold_hover"],
            font=("Microsoft YaHei UI", 9, "bold"),
            padx=8,
            pady=2,
        )
        status_lbl.pack(side=tk.RIGHT)

        ttk.Label(
            desc_card,
            text="开启后，大模型将以 60FPS 双速流水线实时观察游戏画面，结合左上角任务目标，自主做出技能连招、残血吃药、钩索寻路、跳过解谜等全流程操作决策。",
            style="CardMuted.TLabel",
        ).pack(anchor=tk.W, pady=(4, 0))

        # 2. API 与模型接入设置卡片
        api_frame = ttk.LabelFrame(tab, text="大模型 API 与服务接入设置", padding=12)
        api_frame.pack(fill=tk.X, pady=(0, 10))
        api_frame.columnconfigure(1, weight=1)

        # 提供商选择
        ttk.Label(api_frame, text="服务提供商：", style="Card.TLabel").grid(row=0, column=0, sticky=tk.W, pady=6)
        self.provider_combo = ttk.Combobox(
            api_frame,
            textvariable=self.ai_provider_var,
            values=["gemini", "openai", "ollama"],
            state="readonly",
            width=24,
        )
        self.provider_combo.grid(row=0, column=1, sticky=tk.W, pady=6)
        self.provider_combo.bind("<<ComboboxSelected>>", self._on_ai_provider_change)
        ttk.Label(api_frame, text="（推荐使用 gemini，多模态推理耗时仅 200~300ms 且极速敏捷）", style="CardMuted.TLabel").grid(row=0, column=2, sticky=tk.W, padx=6)

        # 模型选择
        ttk.Label(api_frame, text="模型名称 (Model)：", style="Card.TLabel").grid(row=1, column=0, sticky=tk.W, pady=6)
        model_box = tk.Frame(api_frame, bg=THEME["bg_card"])
        model_box.grid(row=1, column=1, sticky=tk.W, pady=6)
        self.model_combo = ttk.Combobox(
            model_box,
            textvariable=self.ai_model_var,
            values=["gemini-auto", "gemini-default", "gemini-2.5-flash", "gemini-1.5-flash", "deepseek-chat", "gpt-4o-mini", "qwen2.5-vl:7b"],
            width=28,
        )
        self.model_combo.pack(side=tk.LEFT)
        self.btn_fetch_models = ttk.Button(model_box, text="🔄 获取模型列表", command=self._fetch_remote_models, width=13)
        self.btn_fetch_models.pack(side=tk.LEFT, padx=(6, 0))
        ttk.Label(api_frame, text="（点击可一键拉取网关所有可用模型，也可手动输入）", style="CardMuted.TLabel").grid(row=1, column=2, sticky=tk.W, padx=6)

        # API Key
        ttk.Label(api_frame, text="API 密钥 (API Key)：", style="Card.TLabel").grid(row=2, column=0, sticky=tk.W, pady=6)
        key_box = tk.Frame(api_frame, bg=THEME["bg_card"])
        key_box.grid(row=2, column=1, sticky=tk.EW, pady=6)

        self.key_entry = ttk.Entry(key_box, textvariable=self.ai_key_var, show="*")
        self.key_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(key_box, text="显示/隐藏", command=self._toggle_show_key, width=9).pack(side=tk.LEFT, padx=(6, 0))

        ttk.Label(api_frame, text="（支持 sk- 格式网关密钥或 Google 官方 Key，保存在本地 config.yaml）", style="CardMuted.TLabel").grid(row=2, column=2, sticky=tk.W, padx=6)

        # 接口地址 Base URL
        ttk.Label(api_frame, text="接口地址 (Base URL)：", style="Card.TLabel").grid(row=3, column=0, sticky=tk.W, pady=6)
        ttk.Entry(api_frame, textvariable=self.ai_base_url_var, width=54).grid(row=3, column=1, sticky=tk.EW, pady=6)
        ttk.Label(api_frame, text="（留空即默认官方；本地聚合网关请填 http://127.0.0.1:8090；OpenAI填 http://127.0.0.1:8090/v1）", style="CardMuted.TLabel").grid(row=3, column=2, sticky=tk.W, padx=6)

        # 3. 极速流控与压缩调优
        speed_frame = ttk.LabelFrame(tab, text="超低延迟与视觉压缩调优（反应速度核心配置）", padding=12)
        speed_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(speed_frame, text="画面压缩分辨率：", style="Card.TLabel").grid(row=0, column=0, sticky=tk.W, padx=4, pady=4)
        res_combo = ttk.Combobox(
            speed_frame,
            textvariable=self.ai_resolution_var,
            values=[
                "640x360 (极速推荐, <35KB)",
                "480x270 (超轻量极速, <18KB)",
                "960x540 (平衡清晰度)",
                "1280x720 (高画质慢速, 需超时>15秒)",
            ],
            state="readonly",
            width=28,
        )
        res_combo.grid(row=0, column=1, sticky=tk.W, padx=4, pady=4)

        ttk.Label(speed_frame, text="JPEG 质量：", style="Card.TLabel").grid(row=0, column=2, sticky=tk.W, padx=6, pady=4)
        ttk.Entry(speed_frame, textvariable=self.ai_quality_var, width=6).grid(row=0, column=3, sticky=tk.W, padx=4, pady=4)

        ttk.Label(speed_frame, text="超时阈值(秒)：", style="Card.TLabel").grid(row=0, column=4, sticky=tk.W, padx=6, pady=4)
        ttk.Entry(speed_frame, textvariable=self.ai_timeout_var, width=6).grid(row=0, column=5, sticky=tk.W, padx=4, pady=4)
        ttk.Label(speed_frame, text="（推荐 10.0 秒，大模型云端视觉推理通常需 5~8 秒）", style="CardMuted.TLabel").grid(row=0, column=6, sticky=tk.W, padx=6, pady=4)

        # 4. 测试与保存控制条
        action_card = tk.Frame(tab, bg=THEME["bg_card"], highlightthickness=1, highlightbackground=THEME["border"], padx=14, pady=12)
        action_card.pack(fill=tk.X)

        test_row = tk.Frame(action_card, bg=THEME["bg_card"])
        test_row.pack(fill=tk.X)

        self.btn_test_ai = ttk.Button(test_row, text="⚡ 一键测试 AI 连接与响应延迟", command=self._run_ai_test)
        self.btn_test_ai.pack(side=tk.LEFT)

        ttk.Button(test_row, text="💾 保存 AI 配置并立即生效", command=self._save_ai_config, style="Primary.TButton").pack(side=tk.RIGHT)

        self.lbl_ai_test_status = tk.Label(
            action_card,
            textvariable=self.ai_test_status_var,
            bg=THEME["bg_card"],
            fg=THEME["text_secondary"],
            font=("Microsoft YaHei UI", 9),
            anchor=tk.W,
            justify=tk.LEFT,
        )
        self.lbl_ai_test_status.pack(anchor=tk.W, fill=tk.X, pady=(8, 0))

    def _build_presets_tab(self) -> None:
        tab = ttk.Frame(self.notebook, padding=12)
        self.notebook.add(tab, text=" 预设管理 ")

        left = tk.Frame(tab, bg=THEME["bg_card"], highlightthickness=1, highlightbackground=THEME["border"], width=260, padx=10, pady=10)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left.pack_propagate(False)

        ttk.Label(left, text="预设配置方案", style="Card.TLabel", font=("Microsoft YaHei UI", 10, "bold")).pack(anchor=tk.W, pady=(0, 8))
        self.preset_listbox = tk.Listbox(
            left,
            exportselection=False,
            font=("Microsoft YaHei UI", 9),
            bg="#ffffff",
            fg=THEME["text_primary"],
            selectbackground=THEME["gold_bg"],
            selectforeground=THEME["gold_hover"],
            relief="flat",
            highlightthickness=1,
            highlightbackground=THEME["border"],
        )
        self.preset_listbox.pack(fill=tk.BOTH, expand=True)
        self.preset_listbox.bind("<<ListboxSelect>>", self._on_preset_select)

        btn_row = tk.Frame(left, bg=THEME["bg_card"])
        btn_row.pack(fill=tk.X, pady=(8, 0))
        ttk.Button(btn_row, text="复制新建", command=self._duplicate_preset).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(btn_row, text="删除预设", command=self._delete_preset).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 0))

        right = tk.Frame(tab, bg=THEME["bg_main"])
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        meta = ttk.LabelFrame(right, text="方案基础信息", padding=10)
        meta.pack(fill=tk.X)
        ttk.Label(meta, text="预设名称：", style="Card.TLabel").grid(row=0, column=0, sticky=tk.W, pady=3)
        self.preset_name_var = tk.StringVar()
        ttk.Entry(meta, textvariable=self.preset_name_var, width=38).grid(row=0, column=1, sticky=tk.EW, pady=3)

        ttk.Label(meta, text="说明描述：", style="Card.TLabel").grid(row=1, column=0, sticky=tk.W, pady=3)
        self.preset_desc_var = tk.StringVar()
        ttk.Entry(meta, textvariable=self.preset_desc_var, width=50).grid(row=1, column=1, sticky=tk.EW, pady=3)
        meta.columnconfigure(1, weight=1)

        feat_frame = ttk.LabelFrame(right, text="功能开关矩阵", padding=10)
        feat_frame.pack(fill=tk.X, pady=(8, 0))
        for idx, (key, label) in enumerate(FEATURE_LABELS):
            var = tk.BooleanVar(value=True)
            self._feature_vars[key] = var
            ttk.Checkbutton(feat_frame, text=label, variable=var).grid(
                row=idx // 2, column=idx % 2, sticky=tk.W, padx=8, pady=3
            )

        param_frame = ttk.LabelFrame(right, text="战斗与寻路微调参数", padding=10)
        param_frame.pack(fill=tk.X, pady=(8, 0))
        self.far_dist_var = tk.StringVar(value="300")
        self.sprint_var = tk.StringVar(value="0.45")
        self.grace_var = tk.StringVar(value="2.5")
        self.concerto_var = tk.BooleanVar(value=True)

        ttk.Label(param_frame, text="传送距离阈值(米):", style="Card.TLabel").grid(row=0, column=0, sticky=tk.W, padx=4)
        ttk.Entry(param_frame, textvariable=self.far_dist_var, width=8).grid(row=0, column=1, sticky=tk.W, padx=4)
        ttk.Label(param_frame, text="冲刺持续时长(秒):", style="Card.TLabel").grid(row=0, column=2, sticky=tk.W, padx=4)
        ttk.Entry(param_frame, textvariable=self.sprint_var, width=8).grid(row=0, column=3, sticky=tk.W, padx=4)
        ttk.Label(param_frame, text="脱战缓冲等待(秒):", style="Card.TLabel").grid(row=0, column=4, sticky=tk.W, padx=4)
        ttk.Entry(param_frame, textvariable=self.grace_var, width=8).grid(row=0, column=5, sticky=tk.W, padx=4)
        ttk.Checkbutton(param_frame, text="协奏满自动切换角色", variable=self.concerto_var).grid(
            row=1, column=0, columnspan=3, sticky=tk.W, pady=(6, 0)
        )

        rot_frame = ttk.LabelFrame(right, text="战斗技能与连招序列（YAML 格式，支持 click/dodge/key/switch_next）", padding=8)
        rot_frame.pack(fill=tk.BOTH, expand=True, pady=(8, 0))

        sc_rot_frame, self.rotation_text = create_dark_scrolled_text(rot_frame, height=7)
        sc_rot_frame.pack(fill=tk.BOTH, expand=True)

        actions = tk.Frame(right, bg=THEME["bg_main"])
        actions.pack(fill=tk.X, pady=(8, 0))
        ttk.Button(actions, text="保存修改并立即生效", command=self._save_current_preset, style="Primary.TButton").pack(side=tk.RIGHT)

    def _build_logs_tab(self) -> None:
        tab = ttk.Frame(self.notebook, padding=12)
        self.notebook.add(tab, text=" 日志查看 ")

        toolbar = tk.Frame(tab, bg=THEME["bg_main"])
        toolbar.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(toolbar, text="日志级别筛选：").pack(side=tk.LEFT)
        self.log_level_filter = ttk.Combobox(
            toolbar,
            values=["ALL", "INFO", "WARNING", "ERROR"],
            state="readonly",
            width=10,
        )
        self.log_level_filter.set("ALL")
        self.log_level_filter.pack(side=tk.LEFT, padx=6)

        ttk.Button(toolbar, text="清空当前日志", command=self._clear_log_view).pack(side=tk.LEFT, padx=4)
        ttk.Button(toolbar, text="导出日志文件", command=self._export_logs).pack(side=tk.LEFT, padx=4)
        ttk.Button(toolbar, text="打开日志目录 (logs)", command=self._open_logs_dir).pack(side=tk.RIGHT)

        sc_frame, self.full_log_text = create_dark_scrolled_text(tab, height=20, state=tk.DISABLED)
        sc_frame.pack(fill=tk.BOTH, expand=True)
        self._setup_text_tags(self.full_log_text)

    def _build_tools_tab(self) -> None:
        tab = ttk.Frame(self.notebook, padding=12)
        self.notebook.add(tab, text=" 模板与调试工具 ")

        top = ttk.LabelFrame(tab, text="一键自动化工具箱", padding=10)
        top.pack(fill=tk.X)
        ttk.Button(top, text="一键自动校准所有调试框 (ROI)", command=self._run_auto_calibrate).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(top, text="打开实时调试预览窗口", command=self._launch_debug_overlay).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="打开模板文件夹 (assets)", command=self._open_assets_dir).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="刷新模板采集状态", command=self._refresh_template_status).pack(side=tk.RIGHT)

        tpl_frame = ttk.LabelFrame(tab, text="UI 模板资产状态（支持零模板直接运行，并在执行中自适应采集）", padding=10)
        tpl_frame.pack(fill=tk.BOTH, expand=True, pady=(12, 0))

        cols = ("name", "desc", "status")
        self.tpl_tree = ttk.Treeview(tpl_frame, columns=cols, show="headings", height=14)
        self.tpl_tree.heading("name", text="模板文件名")
        self.tpl_tree.heading("desc", text="用途与功能说明")
        self.tpl_tree.heading("status", text="状态")
        self.tpl_tree.column("name", width=220)
        self.tpl_tree.column("desc", width=480)
        self.tpl_tree.column("status", width=120, anchor=tk.CENTER)
        self.tpl_tree.pack(fill=tk.BOTH, expand=True)

        bottom = tk.Frame(tpl_frame, bg=THEME["bg_card"])
        bottom.pack(fill=tk.X, pady=(8, 0))
        ttk.Button(bottom, text="框选采集选中的模板", command=self._capture_selected_template).pack(side=tk.RIGHT)

    @staticmethod
    def _setup_text_tags(text_widget: tk.Text) -> None:
        text_widget.tag_config("WARNING", foreground="#b45309")   # 温暖深琥珀金
        text_widget.tag_config("ERROR", foreground="#dc2626")     # 警示绯红
        text_widget.tag_config("INFO", foreground="#0284c7")      # 明澈深天蓝

    def _detect_game_window(self, interactive: bool = True) -> None:
        """主动检测并绑定鸣潮游戏窗口。"""
        win = self.runner.capture.find_window()
        if win:
            title_str = f"{win.title} ({win.width}x{win.height})"
            self.status_vars["window"].set(title_str)
            self.logger.info("已成功绑定鸣潮游戏窗口: %s", title_str)
            if interactive:
                messagebox.showinfo("窗口绑定成功", f"已成功识别并绑定鸣潮游戏窗口：\n\n{title_str}\n\n分辨率与坐标完美匹配，可直接按 [F8] 启动！")
        else:
            self.status_vars["window"].set("未绑定（请先启动鸣潮）")
            if interactive:
                messagebox.showwarning("未找到游戏窗口", "未检测到鸣潮游戏窗口。\n\n请先打开鸣潮并进入游戏世界（建议设置为 1080P 无边框或窗口化），然后再点击此按钮重新检测。")

    def _on_toggle_auto_collect_btn(self) -> None:
        """现代化胶囊按钮切换模板自动收集，彻底杜绝 Win32 方块字符。"""
        new_val = not self.auto_collect_var.get()
        self.auto_collect_var.set(new_val)
        self.runner.set_auto_collect(new_val)
        self.config.setdefault("vision", {})["auto_collect_templates"] = new_val
        self._save_config()
        if new_val:
            self.btn_auto_collect.config(text="✔ 自动收集模板: 已开启", style="ToggleOn.TButton")
            self.logger.info("UI 模板自动收集已开启（脚本运行中将自适应识别并收集新模板）")
        else:
            self.btn_auto_collect.config(text="○ 自动收集模板: 已关闭", style="ToggleOff.TButton")
            self.logger.info("UI 模板自动收集已关闭")

    def _toggle_ai_mode_btn(self) -> None:
        new_state = not self.ai_mode_var.get()
        self.ai_mode_var.set(new_state)
        self.runner.set_ai_mode(new_state)
        self.config.setdefault("ai_autopilot", {})["enabled"] = new_state
        self._save_config()

        if new_state:
            self.btn_ai_toggle.config(text="✦ AI 托管: 运行中", style="AIOn.TButton")
            self.status_vars["ai_mode"].set("✦ AI 托管中")
        else:
            self.btn_ai_toggle.config(text="✦ AI 托管: 已停用", style="AIOff.TButton")
            self.status_vars["ai_mode"].set("规则模式")

    def _toggle_show_key(self) -> None:
        self._show_key = not self._show_key
        self.key_entry.config(show="" if self._show_key else "*")

    def _on_ai_provider_change(self, _event=None) -> None:
        prov = self.ai_provider_var.get().lower()
        if prov == "gemini":
            self.ai_model_var.set("gemini-auto")
            # 若已存在本地或其它自定义网关，保留原 URL，避免覆盖用户配置
            if not self.ai_base_url_var.get() or "deepseek" in self.ai_base_url_var.get() or "11434" in self.ai_base_url_var.get():
                self.ai_base_url_var.set("http://127.0.0.1:8090")
        elif prov == "openai":
            self.ai_model_var.set("gemini-auto")
            if not self.ai_base_url_var.get() or "googleapis" in self.ai_base_url_var.get() or "11434" in self.ai_base_url_var.get():
                self.ai_base_url_var.set("http://127.0.0.1:8090/v1")
        elif prov == "ollama":
            self.ai_model_var.set("qwen2.5-vl:7b")
            self.ai_base_url_var.set("http://localhost:11434/v1")

    def _fetch_remote_models(self) -> None:
        prov = self.ai_provider_var.get()
        base_url = self.ai_base_url_var.get().strip()
        api_key = self.ai_key_var.get().strip()
        self.btn_fetch_models.config(state=tk.DISABLED)
        self.ai_test_status_var.set("正在从目标接口或网关拉取可用模型列表...")

        def _bg():
            models = self.runner.fetch_ai_models(prov, base_url, api_key)

            def _update():
                self.btn_fetch_models.config(state=tk.NORMAL)
                if models:
                    self.model_combo["values"] = models
                    curr = self.ai_model_var.get()
                    if curr not in models:
                        preferred = next((m for m in models if "auto" in m or "flash" in m or "default" in m), models[0])
                        self.ai_model_var.set(preferred)
                    self.lbl_ai_test_status.config(fg=THEME["green"])
                    self.ai_test_status_var.set(f"✔ 成功拉取到 {len(models)} 个可用模型！已自动填入下拉列表。")
                else:
                    self.lbl_ai_test_status.config(fg=THEME["red"])
                    self.ai_test_status_var.set("✖ 未能拉取到模型列表（请确认 Base URL 端口已开启且已输入 API Key）")

            self.root.after(0, _update)

        threading.Thread(target=_bg, daemon=True).start()

    def _save_ai_config(self) -> None:
        res_str = self.ai_resolution_var.get().split(" ")[0]
        try:
            w, h = [int(v) for v in res_str.split("x")]
            quality = int(self.ai_quality_var.get())
            timeout = float(self.ai_timeout_var.get())
        except Exception as exc:
            messagebox.showerror("格式错误", f"数值参数格式错误: {exc}")
            return

        ai_cfg = self.config.setdefault("ai_autopilot", {})
        ai_cfg["provider"] = self.ai_provider_var.get()
        ai_cfg["model"] = self.ai_model_var.get()
        ai_cfg["api_key"] = self.ai_key_var.get().strip()
        ai_cfg["base_url"] = self.ai_base_url_var.get().strip()
        ai_cfg["fast_width"] = w
        ai_cfg["fast_height"] = h
        ai_cfg["jpeg_quality"] = quality
        ai_cfg["timeout_sec"] = timeout
        ai_cfg["enabled"] = self.ai_mode_var.get()

        self._save_config()
        self.runner.update_ai_config(ai_cfg)
        messagebox.showinfo("保存成功", "AI 托管配置已保存，并已热更新至运行引擎！")

    def _run_ai_test(self) -> None:
        self.btn_test_ai.config(state=tk.DISABLED)
        self.ai_test_status_var.set("正在向大模型发送画面测速（云端视觉多模态推理通常需 5~8 秒，请耐心稍候）...")

        def _bg_worker():
            # 临时生成最新配置进行测速
            res_str = self.ai_resolution_var.get().split(" ")[0]
            try:
                w, h = [int(v) for v in res_str.split("x")]
            except Exception:
                w, h = 640, 360
            temp_cfg = {
                "provider": self.ai_provider_var.get(),
                "model": self.ai_model_var.get(),
                "api_key": self.ai_key_var.get().strip(),
                "base_url": self.ai_base_url_var.get().strip(),
                "fast_width": w,
                "fast_height": h,
                "jpeg_quality": int(self.ai_quality_var.get() or 65),
                "timeout_sec": float(self.ai_timeout_var.get() or 3.0),
            }
            self.runner.update_ai_config(temp_cfg)
            ok, latency_ms, msg = self.runner.test_ai_connection()

            def _update():
                self.btn_test_ai.config(state=tk.NORMAL)
                if ok:
                    self.lbl_ai_test_status.config(fg="#065f46", bg="#d1fae5", padx=10, pady=4)
                    self.ai_test_status_var.set(f"✔ 通信成功！总延迟: {latency_ms:.0f}ms | {msg}")
                else:
                    self.lbl_ai_test_status.config(fg="#991b1b", bg="#fee2e2", padx=10, pady=4)
                    self.ai_test_status_var.set(f"✖ 通信失败 ({latency_ms:.0f}ms): {msg}")

            self.root.after(0, _update)

        threading.Thread(target=_bg_worker, daemon=True).start()

    def _on_toggle_auto_collect(self) -> None:
        self.runner.set_auto_collect(self.auto_collect_var.get())
        self._save_config()

    def _run_auto_calibrate(self) -> None:
        ok, msg = self.runner.auto_calibrate()
        if ok:
            self._save_config()
            self._refresh_template_status()
            messagebox.showinfo("一键自动校准完成", msg)
        else:
            messagebox.showwarning("未找到游戏窗口", msg)

    def _on_runner_status(self, payload: dict) -> None:
        def _update():
            running = payload.get("running", False)
            self.status_vars["running"].set("● 运行中" if running else "○ 已暂停")
            if running:
                self.btn_toggle.config(text="暂停脚本 (F8)", style="Running.TButton")
            else:
                self.btn_toggle.config(text="启动脚本 (F8)", style="Primary.TButton")

            ai_enabled = payload.get("ai_enabled", False)
            ai_reason = payload.get("ai_reason", "")
            self.ai_mode_var.set(ai_enabled)
            self.status_vars["ai_mode"].set("✦ AI 托管中" if ai_enabled else "规则引擎")

            state = payload.get("state", "IDLE")
            reason = payload.get("takeover_reason", "")
            if ai_enabled and state == "AI_AUTOPILOT":
                self.status_vars["state"].set(f"AI托管: {ai_reason}")
            else:
                self.status_vars["state"].set(f"{state} ({reason})" if reason else state)

            self.status_vars["window"].set(payload.get("window_title", "未绑定（请启动鸣潮）"))
            self.status_vars["preset"].set(payload.get("preset_name", ""))
            quest_text = payload.get("quest_text", "")
            self.status_vars["quest"].set(quest_text if quest_text else "暂无识别到的任务文本")
            tpl_count = payload.get("template_count", 0)
            self.status_vars["templates"].set(f"{tpl_count} 个已就绪（零模板可直接运行）")

            l5_sum = payload.get("l5_summary", {})
            if l5_sum:
                self._update_l5_dashboard(l5_sum)

        self.root.after(0, _update)

    def _apply_l5_master_goal(self) -> None:
        goal_text = self.l5_intent_var.get().strip()
        if not goal_text:
            return
        if hasattr(self.runner, "sm") and self.runner.sm:
            self.runner.sm.set_l5_goal(goal_text)
            self.logger.info(f"🎯 已向 L5 大脑下达全新大目标: 【{goal_text}】")
            summary = self.runner.sm.get_l5_summary()
            self._update_l5_dashboard(summary)
        else:
            self.logger.info(f"🎯 L5 大目标已就绪: 【{goal_text}】(启动脚本后将自动加载)")

    def _update_l5_dashboard(self, summary: dict) -> None:
        try:
            cur_goal = summary.get("current_goal", "就绪")
            prio = summary.get("current_priority", "-")
            paused = summary.get("paused_stack_depth", 0)
            status_txt = f"当前子目标: {cur_goal} (优先级:{prio}) | 暂挂数:{paused}"
            self.l5_status_lbl.config(text=status_txt)
            thoughts = summary.get("latest_thoughts", [])
            if thoughts:
                self.l5_thought_var.set(thoughts[-1])
        except Exception:
            pass

    def _poll_logs(self) -> None:
        flt = self.log_level_filter.get()
        try:
            while True:
                level, msg = self.log_queue.get_nowait()
                self._append_text(self.mini_log_text, msg + "\n", level, max_lines=120)
                if flt == "ALL" or level == flt:
                    self._append_text(self.full_log_text, msg + "\n", level, max_lines=1500)
        except queue.Empty:
            pass
        self.root.after(150, self._poll_logs)

    @staticmethod
    def _append_text(widget: tk.Text, text: str, tag: Optional[str], max_lines: int) -> None:
        widget.config(state=tk.NORMAL)
        if tag and tag in ("INFO", "WARNING", "ERROR"):
            widget.insert(tk.END, text, tag)
        else:
            widget.insert(tk.END, text)
        line_count = int(widget.index("end-1c").split(".")[0])
        if line_count > max_lines:
            widget.delete("1.0", f"{line_count - max_lines}.0")
        widget.see(tk.END)
        widget.config(state=tk.DISABLED)

    def _refresh_preset_lists(self) -> None:
        self._presets_cache = self.preset_manager.list_presets()
        self.preset_listbox.delete(0, tk.END)
        combo_items = []
        active_id = str(self.config.get("active_preset", "default_full_quest"))
        active_idx = 0

        for idx, item in enumerate(self._presets_cache):
            label = f"{item.get('name', item['id'])} ({item['id']})"
            self.preset_listbox.insert(tk.END, label)
            combo_items.append(label)
            if item["id"] == active_id:
                active_idx = idx

        self.quick_preset_combo["values"] = combo_items
        if combo_items:
            self.quick_preset_combo.current(active_idx)
            self.preset_listbox.selection_set(active_idx)
            self.status_vars["preset"].set(self._presets_cache[active_idx].get("name", active_id))

    def _on_preset_select(self, _event=None) -> None:
        sel = self.preset_listbox.curselection()
        if not sel:
            return
        preset = self._presets_cache[sel[0]]
        self._load_preset_into_editor(preset["id"])

    def _load_preset_into_editor(self, preset_id: str) -> None:
        preset = self.preset_manager.get_preset(preset_id)
        self._current_editing_preset_id = preset["id"]
        self.preset_name_var.set(preset.get("name", preset["id"]))
        self.preset_desc_var.set(preset.get("description", ""))

        feats = preset.get("features", {})
        for key, var in self._feature_vars.items():
            var.set(bool(feats.get(key, True)))

        nav = preset.get("navigation", {})
        combat = preset.get("combat", {})
        self.far_dist_var.set(str(nav.get("far_distance_m", 300)))
        self.sprint_var.set(str(nav.get("sprint_hold_sec", 0.45)))
        self.grace_var.set(str(combat.get("exit_grace_sec", 2.5)))
        self.concerto_var.set(bool(combat.get("use_concerto_switch", True)))

        rot_yaml = yaml.safe_dump(combat.get("rotation", []), allow_unicode=True, sort_keys=False)
        self.rotation_text.delete("1.0", tk.END)
        self.rotation_text.insert("1.0", rot_yaml)

    def _save_current_preset(self) -> None:
        try:
            rotation = yaml.safe_load(self.rotation_text.get("1.0", tk.END)) or []
            if not isinstance(rotation, list):
                raise ValueError("战斗连招必须是 YAML 列表格式。")
            far_dist = int(float(self.far_dist_var.get()))
            sprint_sec = float(self.sprint_var.get())
            grace_sec = float(self.grace_var.get())
        except Exception as exc:
            messagebox.showerror("参数格式错误", f"保存失败：{exc}")
            return

        preset = copy.deepcopy(self.preset_manager.get_preset(self._current_editing_preset_id))
        preset["name"] = self.preset_name_var.get().strip() or preset["id"]
        preset["description"] = self.preset_desc_var.get().strip()
        preset["features"] = {k: bool(v.get()) for k, v in self._feature_vars.items()}
        preset.setdefault("navigation", {})
        preset["navigation"]["far_distance_m"] = far_dist
        preset["navigation"]["sprint_hold_sec"] = sprint_sec
        preset.setdefault("combat", {})
        preset["combat"]["exit_grace_sec"] = grace_sec
        preset["combat"]["use_concerto_switch"] = bool(self.concerto_var.get())
        preset["combat"]["rotation"] = rotation

        self.preset_manager.save_preset(preset["id"], preset)
        self.config["active_preset"] = preset["id"]
        self._save_config()
        self.runner.apply_preset(preset["id"])
        self._refresh_preset_lists()
        messagebox.showinfo("保存成功", f"预设 [{preset['name']}] 已保存并立即生效。")

    def _duplicate_preset(self) -> None:
        new_preset = self.preset_manager.duplicate_preset(self._current_editing_preset_id)
        self._refresh_preset_lists()
        self._load_preset_into_editor(new_preset["id"])

    def _delete_preset(self) -> None:
        if not messagebox.askyesno("确认删除", f"确定删除预设 [{self._current_editing_preset_id}] 吗？"):
            return
        if not self.preset_manager.delete_preset(self._current_editing_preset_id):
            messagebox.showwarning("无法删除", "至少需要保留一套操作预设。")
            return
        self._refresh_preset_lists()
        first_id = self._presets_cache[0]["id"]
        self.config["active_preset"] = first_id
        self._save_config()
        self.runner.apply_preset(first_id)
        self._load_preset_into_editor(first_id)

    def _apply_quick_preset(self) -> None:
        idx = self.quick_preset_combo.current()
        if idx < 0 or idx >= len(self._presets_cache):
            return
        preset = self._presets_cache[idx]
        self.config["active_preset"] = preset["id"]
        self._save_config()
        self.runner.apply_preset(preset["id"])
        self._load_preset_into_editor(preset["id"])

    def _clear_log_view(self) -> None:
        self.full_log_text.config(state=tk.NORMAL)
        self.full_log_text.delete("1.0", tk.END)
        self.full_log_text.config(state=tk.DISABLED)

    def _export_logs(self) -> None:
        path = filedialog.asksaveasfilename(
            title="导出日志",
            defaultextension=".log",
            filetypes=[("Log files", "*.log"), ("Text files", "*.txt")],
        )
        if not path:
            return
        content = self.full_log_text.get("1.0", tk.END)
        Path(path).write_text(content, encoding="utf-8")
        messagebox.showinfo("导出成功", f"日志已导出至：{path}")

    def _open_logs_dir(self) -> None:
        logs_dir = self.base_dir / self.config.get("logging", {}).get("save_dir", "logs")
        logs_dir.mkdir(parents=True, exist_ok=True)
        os.startfile(str(logs_dir))

    def _open_assets_dir(self) -> None:
        assets_dir = self.base_dir / "assets"
        assets_dir.mkdir(parents=True, exist_ok=True)
        os.startfile(str(assets_dir))

    def _refresh_template_status(self) -> None:
        for row in self.tpl_tree.get_children():
            self.tpl_tree.delete(row)
        assets_dir = self.base_dir / "assets"
        count = 0
        for name, desc in COMMON_TEMPLATES:
            exists = (assets_dir / name).exists()
            if exists:
                count += 1
            self.tpl_tree.insert("", tk.END, values=(name, desc, "✔ 已就绪" if exists else "○ 自动待收集"))
        self.status_vars["templates"].set(f"{count} 个已就绪（零模板可直接运行）")

    def _python_exe(self) -> str:
        venv_py = self.base_dir / ".venv" / "Scripts" / "python.exe"
        return str(venv_py) if venv_py.exists() else sys.executable

    def _launch_debug_overlay(self) -> None:
        subprocess.Popen([self._python_exe(), "-m", "tools.debug_overlay"], cwd=str(self.base_dir))

    def _capture_selected_template(self) -> None:
        sel = self.tpl_tree.selection()
        if not sel:
            messagebox.showinfo("提示", "请先在列表中选中一个模板。")
            return
        tpl_name = self.tpl_tree.item(sel[0], "values")[0]
        subprocess.Popen(
            [self._python_exe(), "-m", "tools.capture_template", "--name", tpl_name],
            cwd=str(self.base_dir),
        )

    def _on_close(self) -> None:
        self.runner.stop()
        self.root.destroy()

    def run(self) -> None:
        self.logger.info("MingchaoJB V2 已就绪。请切回鸣潮游戏窗口后按 F8 即可启动。")
        self.root.mainloop()


def launch_gui(base_dir: Optional[Path] = None) -> int:
    root_dir = Path(base_dir) if base_dir is not None else Path(__file__).resolve().parent
    app = MingchaoGUI(root_dir)
    app.run()
    return 0


if __name__ == "__main__":
    sys.exit(launch_gui())
