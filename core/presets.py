import copy
import re
from pathlib import Path
from typing import Dict, List, Optional

import yaml


DEFAULT_PRESETS: Dict[str, dict] = {
    "default_full_quest": {
        "id": "default_full_quest",
        "name": "全自动清任务（推荐）",
        "description": "自动过剧情、接取切换任务、大地图传送、3D寻路、轮切打怪并战后吸收声骸。",
        "features": {
            "auto_dialogue": True,
            "auto_combat": True,
            "auto_navigate": True,
            "auto_teleport": True,
            "auto_switch_quest": True,
            "auto_absorb_echo": True,
            "auto_popup_close": True,
            "auto_gadget": True,
        },
        "combat": {
            "exit_grace_sec": 2.5,
            "lock_on_key": "middle",
            "use_concerto_switch": True,
            "use_ult_ready": True,
            "rotation": [
                {"action": "click", "count": 3, "interval": 0.16},
                {"action": "key", "key": "e"},
                {"action": "click", "count": 2, "interval": 0.16},
                {"action": "key", "key": "q"},
                {"action": "key", "key": "r"},
                {"action": "switch_next"},
            ],
        },
        "navigation": {
            "sprint_hold_sec": 0.45,
            "turn_sensitivity": 0.42,
            "center_deadzone_px": 70,
            "far_distance_m": 300,
        },
        "quest": {
            "priority_keywords": ["潮汐任务", "伴星任务", "危行任务", "纪闻任务", "每日"],
        },
    },
    "story_only": {
        "id": "story_only",
        "name": "纯自动过剧情 / 对话",
        "description": "仅自动跳过剧情、确认弹窗、点对话选项与 [F] 交互，不主动跑图或打怪。",
        "features": {
            "auto_dialogue": True,
            "auto_combat": False,
            "auto_navigate": False,
            "auto_teleport": False,
            "auto_switch_quest": False,
            "auto_absorb_echo": False,
            "auto_popup_close": True,
            "auto_gadget": False,
        },
        "combat": {
            "exit_grace_sec": 2.0,
            "lock_on_key": "middle",
            "use_concerto_switch": False,
            "use_ult_ready": False,
            "rotation": [],
        },
        "navigation": {
            "sprint_hold_sec": 0.35,
            "turn_sensitivity": 0.40,
            "center_deadzone_px": 75,
            "far_distance_m": 300,
        },
        "quest": {
            "priority_keywords": ["潮汐任务", "伴星任务"],
        },
    },
    "burst_combat": {
        "id": "burst_combat",
        "name": "高爆发轮切打怪 / 刷声骸",
        "description": "专注自动锁敌、穿插闪避、协奏满切人与爆发连招，战后自动吸收声骸。",
        "features": {
            "auto_dialogue": True,
            "auto_combat": True,
            "auto_navigate": True,
            "auto_teleport": False,
            "auto_switch_quest": False,
            "auto_absorb_echo": True,
            "auto_popup_close": True,
            "auto_gadget": True,
        },
        "combat": {
            "exit_grace_sec": 3.0,
            "lock_on_key": "middle",
            "use_concerto_switch": True,
            "use_ult_ready": True,
            "rotation": [
                {"action": "key", "key": "e"},
                {"action": "click", "count": 4, "interval": 0.14},
                {"action": "dodge"},
                {"action": "hold_click", "duration": 0.55},
                {"action": "key", "key": "q"},
                {"action": "key", "key": "r"},
                {"action": "switch_next"},
            ],
        },
        "navigation": {
            "sprint_hold_sec": 0.50,
            "turn_sensitivity": 0.45,
            "center_deadzone_px": 65,
            "far_distance_m": 400,
        },
        "quest": {
            "priority_keywords": ["危行任务", "纪闻任务"],
        },
    },
    "safe_semi_auto": {
        "id": "safe_semi_auto",
        "name": "稳健半自动（远距/切任务提醒）",
        "description": "关闭自动大地图传送与自动切任务，仅处理对话、战斗和近距离寻路，降低误操作风险。",
        "features": {
            "auto_dialogue": True,
            "auto_combat": True,
            "auto_navigate": True,
            "auto_teleport": False,
            "auto_switch_quest": False,
            "auto_absorb_echo": True,
            "auto_popup_close": True,
            "auto_gadget": True,
        },
        "combat": {
            "exit_grace_sec": 2.5,
            "lock_on_key": "middle",
            "use_concerto_switch": True,
            "use_ult_ready": True,
            "rotation": [
                {"action": "click", "count": 3, "interval": 0.18},
                {"action": "key", "key": "e"},
                {"action": "key", "key": "q"},
                {"action": "key", "key": "r"},
                {"action": "switch_next"},
            ],
        },
        "navigation": {
            "sprint_hold_sec": 0.40,
            "turn_sensitivity": 0.38,
            "center_deadzone_px": 80,
            "far_distance_m": 260,
        },
        "quest": {
            "priority_keywords": ["潮汐任务", "伴星任务", "危行任务", "纪闻任务"],
        },
    },
}


def slugify_preset_id(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_\u4e00-\u9fff]+", "_", name.strip()).strip("_")
    return cleaned or "custom_preset"


class PresetManager:
    """管理 presets/ 目录下的多套操作预设，支持增删改查与热切换。"""

    def __init__(self, presets_dir: Path) -> None:
        self.presets_dir = Path(presets_dir)
        self.presets_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_defaults()

    def _ensure_defaults(self) -> None:
        existing = list(self.presets_dir.glob("*.yaml"))
        if existing:
            return
        for preset_id, data in DEFAULT_PRESETS.items():
            self.save_preset(preset_id, copy.deepcopy(data))

    def list_presets(self) -> List[dict]:
        items: List[dict] = []
        for path in sorted(self.presets_dir.glob("*.yaml")):
            try:
                with path.open("r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
                data.setdefault("id", path.stem)
                data.setdefault("name", path.stem)
                data.setdefault("description", "")
                items.append(data)
            except Exception:
                continue
        if not items:
            self._ensure_defaults()
            return self.list_presets()
        return items

    def get_preset(self, preset_id: str) -> dict:
        path = self.presets_dir / f"{preset_id}.yaml"
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            data.setdefault("id", preset_id)
            return data
        all_items = self.list_presets()
        return all_items[0]

    def save_preset(self, preset_id: str, data: dict) -> Path:
        safe_id = slugify_preset_id(preset_id)
        payload = copy.deepcopy(data)
        payload["id"] = safe_id
        path = self.presets_dir / f"{safe_id}.yaml"
        with path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)
        return path

    def delete_preset(self, preset_id: str) -> bool:
        all_items = self.list_presets()
        if len(all_items) <= 1:
            return False
        path = self.presets_dir / f"{preset_id}.yaml"
        if path.exists():
            path.unlink()
            return True
        return False

    def duplicate_preset(self, source_id: str, new_name: Optional[str] = None) -> dict:
        src = copy.deepcopy(self.get_preset(source_id))
        base_name = new_name or f"{src.get('name', source_id)}_副本"
        new_id = slugify_preset_id(base_name)
        idx = 1
        while (self.presets_dir / f"{new_id}.yaml").exists():
            new_id = f"{slugify_preset_id(base_name)}_{idx}"
            idx += 1
        src["id"] = new_id
        src["name"] = base_name
        self.save_preset(new_id, src)
        return src
