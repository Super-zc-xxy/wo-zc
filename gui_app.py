import copy
import os
import queue
import subprocess
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk

import yaml

from core.bot_runner import BotRunner
from core.logger import setup_logger
from core.presets import PresetManager


FEATURE_LABELS = [
    ("auto_dialogue", "自动跳过剧情与对话选项"),
    ("auto_combat", "自动战斗与队伍轮切"),
    ("auto_navigate", "自动 3D 任务标寻路与脱困"),
    ("auto_teleport", "远距离大地图自动传送"),
    ("auto_switch_quest", "无任务时自动按 J 切换下一任务"),
    ("auto_absorb_echo", "战后自动吸收声骸"),
    ("auto_popup_close", "自动清理弹窗与物品交付"),
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


class MingchaoGUI:
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = Path(base_dir)
        self.config_path = self.base_dir / "config.yaml"
        self.config = self._load_config()

        self.log_queue: "queue.Queue[tuple[str, str]]" = queue.Queue()
        self.logger = setup_logger(self.config, base_dir=self.base_dir, log_queue=self.log_queue)
        self.preset_manager = PresetManager(self.base_dir / "presets")

        self.root = tk.Tk()
        self.root.title("MingchaoJB V2 - 自动任务控制中心")
        self.root.geometry("1040x720")
        self.root.minsize(920, 620)

        self.status_vars = {
            "running": tk.StringVar(value="已暂停"),
            "state": tk.StringVar(value="IDLE"),
            "window": tk.StringVar(value="未绑定（请先启动鸣潮）"),
            "preset": tk.StringVar(value=""),
            "quest": tk.StringVar(value="暂无识别到的任务文本"),
            "templates": tk.StringVar(value="0 个（零模板可直接运行，运行中自动收集）"),
        }
        self.auto_collect_var = tk.BooleanVar(
            value=bool(self.config.get("vision", {}).get("auto_collect_templates", True))
        )

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

    def _build_ui(self) -> None:
        style = ttk.Style()
        try:
            style.theme_use("vista")
        except Exception:
            pass

        top_bar = ttk.Frame(self.root, padding=(12, 10))
        top_bar.pack(fill=tk.X)

        ttk.Label(top_bar, text="MingchaoJB 自动任务助手 V2", font=("Microsoft YaHei UI", 14, "bold")).pack(
            side=tk.LEFT
        )
        ttk.Label(
            top_bar,
            text="  |  F8 启动/暂停   F7 接管恢复（免手动建模板）",
            foreground="#555555",
        ).pack(side=tk.LEFT, padx=(8, 0))

        self.btn_toggle = ttk.Button(top_bar, text="启动脚本 (F8)", command=self.runner.toggle, width=15)
        self.btn_toggle.pack(side=tk.RIGHT, padx=(6, 0))
        ttk.Button(top_bar, text="接管恢复 (F7)", command=self.runner.resume_takeover, width=14).pack(
            side=tk.RIGHT, padx=4
        )
        ttk.Button(top_bar, text="一键自动校准", command=self._run_auto_calibrate, width=14).pack(
            side=tk.RIGHT, padx=4
        )

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        self._build_dashboard_tab()
        self._build_presets_tab()
        self._build_logs_tab()
        self._build_tools_tab()

    def _build_dashboard_tab(self) -> None:
        tab = ttk.Frame(self.notebook, padding=12)
        self.notebook.add(tab, text="控制台")

        status_frame = ttk.LabelFrame(tab, text="运行状态（切回游戏窗口后按 F8 自动执行）", padding=12)
        status_frame.pack(fill=tk.X)

        rows = [
            ("运行状态：", "running"),
            ("当前状态机：", "state"),
            ("绑定游戏窗口：", "window"),
            ("当前操作预设：", "preset"),
            ("已收集模板：", "templates"),
            ("任务栏识别：", "quest"),
        ]
        for idx, (label, key) in enumerate(rows):
            ttk.Label(status_frame, text=label, width=15).grid(row=idx, column=0, sticky=tk.W, pady=3)
            ttk.Label(status_frame, textvariable=self.status_vars[key], font=("Microsoft YaHei UI", 10, "bold")).grid(
                row=idx, column=1, sticky=tk.W, pady=3
            )

        quick_frame = ttk.LabelFrame(tab, text="新手零门槛一键操作", padding=12)
        quick_frame.pack(fill=tk.X, pady=(10, 0))

        ttk.Label(quick_frame, text="选择预设：").pack(side=tk.LEFT)
        self.quick_preset_combo = ttk.Combobox(quick_frame, state="readonly", width=30)
        self.quick_preset_combo.pack(side=tk.LEFT, padx=6)
        ttk.Button(quick_frame, text="应用预设", command=self._apply_quick_preset).pack(side=tk.LEFT, padx=4)
        ttk.Button(quick_frame, text="一键自动校准调试框", command=self._run_auto_calibrate).pack(
            side=tk.LEFT, padx=8
        )
        ttk.Checkbutton(
            quick_frame,
            text="运行中自动收集 UI 模板",
            variable=self.auto_collect_var,
            command=self._on_toggle_auto_collect,
        ).pack(side=tk.RIGHT)

        mini_log_frame = ttk.LabelFrame(tab, text="最近日志（完整日志请切至“日志查看”页）", padding=8)
        mini_log_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        self.mini_log_text = scrolledtext.ScrolledText(
            mini_log_frame,
            height=13,
            state=tk.DISABLED,
            font=("Consolas", 9),
            wrap=tk.WORD,
        )
        self.mini_log_text.pack(fill=tk.BOTH, expand=True)

    def _build_presets_tab(self) -> None:
        tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(tab, text="预设操作")

        left = ttk.Frame(tab, width=250)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left.pack_propagate(False)

        ttk.Label(left, text="预设列表", font=("Microsoft YaHei UI", 10, "bold")).pack(anchor=tk.W, pady=(0, 6))
        self.preset_listbox = tk.Listbox(left, exportselection=False, font=("Microsoft YaHei UI", 10))
        self.preset_listbox.pack(fill=tk.BOTH, expand=True)
        self.preset_listbox.bind("<<ListboxSelect>>", self._on_preset_select)

        btn_row = ttk.Frame(left)
        btn_row.pack(fill=tk.X, pady=(8, 0))
        ttk.Button(btn_row, text="复制新建", command=self._duplicate_preset).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(btn_row, text="删除", command=self._delete_preset).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 0)
        )

        right = ttk.Frame(tab)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        meta = ttk.LabelFrame(right, text="基础信息", padding=10)
        meta.pack(fill=tk.X)
        ttk.Label(meta, text="预设名称：").grid(row=0, column=0, sticky=tk.W, pady=3)
        self.preset_name_var = tk.StringVar()
        ttk.Entry(meta, textvariable=self.preset_name_var, width=42).grid(row=0, column=1, sticky=tk.EW, pady=3)
        ttk.Label(meta, text="说明描述：").grid(row=1, column=0, sticky=tk.W, pady=3)
        self.preset_desc_var = tk.StringVar()
        ttk.Entry(meta, textvariable=self.preset_desc_var, width=60).grid(row=1, column=1, sticky=tk.EW, pady=3)
        meta.columnconfigure(1, weight=1)

        feat_frame = ttk.LabelFrame(right, text="功能开关", padding=10)
        feat_frame.pack(fill=tk.X, pady=(8, 0))
        for idx, (key, label) in enumerate(FEATURE_LABELS):
            var = tk.BooleanVar(value=True)
            self._feature_vars[key] = var
            ttk.Checkbutton(feat_frame, text=label, variable=var).grid(
                row=idx // 2, column=idx % 2, sticky=tk.W, padx=8, pady=3
            )

        param_frame = ttk.LabelFrame(right, text="战斗与寻路参数", padding=10)
        param_frame.pack(fill=tk.X, pady=(8, 0))
        self.far_dist_var = tk.StringVar(value="300")
        self.sprint_var = tk.StringVar(value="0.45")
        self.grace_var = tk.StringVar(value="2.5")
        self.concerto_var = tk.BooleanVar(value=True)

        ttk.Label(param_frame, text="远距传送阈值(米):").grid(row=0, column=0, sticky=tk.W, padx=4)
        ttk.Entry(param_frame, textvariable=self.far_dist_var, width=10).grid(row=0, column=1, sticky=tk.W, padx=4)
        ttk.Label(param_frame, text="冲刺时长(秒):").grid(row=0, column=2, sticky=tk.W, padx=4)
        ttk.Entry(param_frame, textvariable=self.sprint_var, width=10).grid(row=0, column=3, sticky=tk.W, padx=4)
        ttk.Label(param_frame, text="脱战缓冲(秒):").grid(row=0, column=4, sticky=tk.W, padx=4)
        ttk.Entry(param_frame, textvariable=self.grace_var, width=10).grid(row=0, column=5, sticky=tk.W, padx=4)
        ttk.Checkbutton(param_frame, text="协奏满自动切人", variable=self.concerto_var).grid(
            row=1, column=0, columnspan=3, sticky=tk.W, pady=(6, 0)
        )

        rot_frame = ttk.LabelFrame(
            right,
            text="战斗连招预设（YAML 列表，支持 click / hold_click / dodge / key / switch_next）",
            padding=8,
        )
        rot_frame.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
        self.rotation_text = scrolledtext.ScrolledText(rot_frame, height=8, font=("Consolas", 9))
        self.rotation_text.pack(fill=tk.BOTH, expand=True)

        actions = ttk.Frame(right)
        actions.pack(fill=tk.X, pady=(8, 0))
        ttk.Button(actions, text="保存预设并立即生效", command=self._save_current_preset).pack(side=tk.RIGHT)

    def _build_logs_tab(self) -> None:
        tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(tab, text="日志查看")

        toolbar = ttk.Frame(tab)
        toolbar.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(toolbar, text="级别筛选：").pack(side=tk.LEFT)
        self.log_level_filter = ttk.Combobox(
            toolbar,
            values=["ALL", "INFO", "WARNING", "ERROR"],
            state="readonly",
            width=10,
        )
        self.log_level_filter.set("ALL")
        self.log_level_filter.pack(side=tk.LEFT, padx=6)

        ttk.Button(toolbar, text="清空显示", command=self._clear_log_view).pack(side=tk.LEFT, padx=4)
        ttk.Button(toolbar, text="导出当前日志", command=self._export_logs).pack(side=tk.LEFT, padx=4)
        ttk.Button(toolbar, text="打开日志文件夹", command=self._open_logs_dir).pack(side=tk.RIGHT)

        self.full_log_text = scrolledtext.ScrolledText(
            tab,
            state=tk.DISABLED,
            font=("Consolas", 9),
            wrap=tk.WORD,
        )
        self.full_log_text.pack(fill=tk.BOTH, expand=True)
        self.full_log_text.tag_config("WARNING", foreground="#b36b00")
        self.full_log_text.tag_config("ERROR", foreground="#cc0000")
        self.full_log_text.tag_config("INFO", foreground="#1a1a1a")

    def _build_tools_tab(self) -> None:
        tab = ttk.Frame(self.notebook, padding=12)
        self.notebook.add(tab, text="自动校准与模板")

        top = ttk.LabelFrame(tab, text="免手动调试工具", padding=10)
        top.pack(fill=tk.X)
        ttk.Button(top, text="一键自动校准所有调试框", command=self._run_auto_calibrate).pack(
            side=tk.LEFT, padx=(0, 8)
        )
        ttk.Button(top, text="预览调试框 (可选)", command=self._launch_debug_overlay).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="打开模板目录 (assets)", command=self._open_assets_dir).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="刷新状态", command=self._refresh_template_status).pack(side=tk.RIGHT)

        tpl_frame = ttk.LabelFrame(
            tab,
            text="UI 模板状态（默认零模板即可运行，运行中会自动收集；也可手动补采）",
            padding=10,
        )
        tpl_frame.pack(fill=tk.BOTH, expand=True, pady=(12, 0))

        cols = ("name", "desc", "status")
        self.tpl_tree = ttk.Treeview(tpl_frame, columns=cols, show="headings", height=14)
        self.tpl_tree.heading("name", text="模板文件名")
        self.tpl_tree.heading("desc", text="用途说明")
        self.tpl_tree.heading("status", text="状态")
        self.tpl_tree.column("name", width=210)
        self.tpl_tree.column("desc", width=470)
        self.tpl_tree.column("status", width=110, anchor=tk.CENTER)
        self.tpl_tree.pack(fill=tk.BOTH, expand=True)

        bottom = ttk.Frame(tpl_frame)
        bottom.pack(fill=tk.X, pady=(8, 0))
        ttk.Button(bottom, text="手动采集所选模板", command=self._capture_selected_template).pack(side=tk.RIGHT)

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
            self.status_vars["running"].set("运行中" if running else "已暂停")
            self.btn_toggle.config(text="暂停脚本 (F8)" if running else "启动脚本 (F8)")
            state = payload.get("state", "IDLE")
            reason = payload.get("takeover_reason", "")
            self.status_vars["state"].set(f"{state} ({reason})" if reason else state)
            self.status_vars["window"].set(payload.get("window_title", "未绑定（请先启动鸣潮）"))
            self.status_vars["preset"].set(payload.get("preset_name", ""))
            quest_text = payload.get("quest_text", "")
            self.status_vars["quest"].set(quest_text if quest_text else "暂无识别到的任务文本")
            tpl_count = payload.get("template_count", 0)
            self.status_vars["templates"].set(f"{tpl_count} 个（零模板可直接运行，运行中自动收集）")

        self.root.after(0, _update)

    def _poll_logs(self) -> None:
        flt = self.log_level_filter.get()
        try:
            while True:
                level, msg = self.log_queue.get_nowait()
                self._append_text(self.mini_log_text, msg + "\n", None, max_lines=120)
                if flt == "ALL" or level == flt:
                    self._append_text(self.full_log_text, msg + "\n", level, max_lines=1500)
        except queue.Empty:
            pass
        self.root.after(150, self._poll_logs)

    @staticmethod
    def _append_text(widget: scrolledtext.ScrolledText, text: str, tag: str | None, max_lines: int) -> None:
        widget.config(state=tk.NORMAL)
        if tag:
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
        messagebox.showinfo("保存成功", f"预设 [{preset['name']}] 已保存并生效。")

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
            self.tpl_tree.insert("", tk.END, values=(name, desc, "已收集" if exists else "自动待收集"))
        self.status_vars["templates"].set(f"{count} 个（零模板可直接运行，运行中自动收集）")

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
        self.logger.info("MingchaoJB V2 已启动。请先点击“一键自动校准”，然后切回鸣潮游戏窗口按 F8 启动。")
        self.root.mainloop()


def launch_gui(base_dir: Path | None = None) -> int:
    root_dir = Path(base_dir) if base_dir is not None else Path(__file__).resolve().parent
    app = MingchaoGUI(root_dir)
    app.run()
    return 0


if __name__ == "__main__":
    sys.exit(launch_gui())
