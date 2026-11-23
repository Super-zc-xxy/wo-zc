import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


class ExperienceMemory:
    """
    鸣潮任务助手 L5 终身语义反思记忆中枢（Lifelong Episodic & Semantic Memory）：
    1. 成功经验库：记录疑难任务、特殊机关、AI 会诊成功破局方案（0ms 本地秒级回放）；
    2. 反思自省库：记录遭遇失败或卡阻时的原因归因与教训（Counterfactual Reflection）；
    3. 语义模式泛化检索：跨任务名基于几何特征标签（如光线折射、爆炸碎石、压力地锁）秒级复用；
    4. 终身自学习持久化存储（data/experience_db.json）。
    """

    def __init__(
        self,
        db_path: Optional[Path] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.logger = logger
        if db_path is None:
            self.db_path = Path("data") / "experience_db.json"
        else:
            self.db_path = Path(db_path)

        self._memory: Dict[str, dict] = {}
        self._reflections: List[Dict[str, Any]] = []
        self._load()

    def _normalize_key(self, text: str) -> str:
        """清洗文本：剔除多余距离、标点和空白符，提取核心任务签名。"""
        if not text:
            return ""
        cleaned = re.sub(r"\d+\s*[mM米]", "", text)
        cleaned = re.sub(r"[\s\.,，。!！\?:：\[\]【】\(\)（）]+", "", cleaned)
        return cleaned.strip()

    def _load(self) -> None:
        if not self.db_path.exists():
            # 初始化默认内置常见解谜与机关高频经验
            self._memory = {
                "机关解谜连线": {
                    "action": "switch_quest",
                    "reason": "复杂连线拼图机关，经验库策略：暂跳过切换下一任务",
                    "success_count": 1,
                    "created_at": time.time(),
                    "tags": ["puzzle", "line_connect"],
                },
                "吸取爆炸果碎石": {
                    "action": "gadget:levitator",
                    "reason": "控物碎石障碍，经验库策略：切换控物工具吸取投掷",
                    "success_count": 1,
                    "created_at": time.time(),
                    "tags": ["levitator", "explosive", "rock"],
                },
                "追踪虚影足迹": {
                    "action": "gadget:sensor",
                    "reason": "虚影足迹线索，经验库策略：切换感知工具扫描追踪",
                    "success_count": 1,
                    "created_at": time.time(),
                    "tags": ["sensor", "trail"],
                },
            }
            self._reflections = [
                {
                    "context": "陡峭岩壁滑落",
                    "failure_type": "steep_slope",
                    "lesson_learned": "大于65度坡面存在隐形滑坡重力，严禁直冲，应自动转角寻找阶梯或蔓藤攀爬",
                    "timestamp": time.time(),
                }
            ]
            self._save()
            return

        try:
            with open(self.db_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict) and "memory" in data:
                    self._memory = data.get("memory", {})
                    self._reflections = data.get("reflections", [])
                elif isinstance(data, dict):
                    self._memory = data
                    self._reflections = []
        except Exception as e:
            if self.logger:
                self.logger.warning(f"读取经验记忆库失败: {e}，将初始化为空记忆库")
            self._memory = {}
            self._reflections = []

    def _save(self) -> None:
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "memory": self._memory,
                "reflections": self._reflections,
                "version": "2.0_l5",
                "last_updated": time.time(),
            }
            with open(self.db_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception as e:
            if self.logger:
                self.logger.warning(f"保存经验记忆库失败: {e}")

    def query_experience(self, quest_text: str) -> Optional[dict]:
        """查询任务是否存在已记录的自愈或破局经验。"""
        norm = self._normalize_key(quest_text)
        if not norm:
            return None

        # 1. 精确匹配
        if norm in self._memory:
            return self._memory[norm]

        # 2. 核心特征子串模糊匹配
        for k, v in self._memory.items():
            if k in norm or norm in k or any(part in norm for part in k.split(":") if len(part) >= 4):
                return v
            # 3. 乱序核心字符集合重合度匹配（容忍汉字前后缀变化）
            k_chars = set(k)
            common = len(k_chars.intersection(set(norm)))
            if len(k) >= 4 and common >= (len(k) - 1):
                return v

        return None

    def query_semantic_pattern(self, feature_tags: List[str]) -> Optional[dict]:
        """
        跨越任务名字局限，基于物理与几何特征标签进行通用语义模式检索。
        例如在全大世界任何未命名的陌生谜题中，若包含 ['explosive', 'rock']，即可命中控物投掷解法！
        """
        if not feature_tags:
            return None

        tags_set = set(t.lower() for t in feature_tags)
        best_match = None
        max_overlap = 0

        for key, record in self._memory.items():
            rec_tags = set(record.get("tags", []))
            overlap = len(tags_set.intersection(rec_tags))
            if overlap > max_overlap:
                max_overlap = overlap
                best_match = record

        if max_overlap >= 1:
            if self.logger:
                self.logger.info(f"💡 [L5语义模式匹配] 基于特征标签 {feature_tags} 跨场景命中经验: {best_match.get('reason')}")
            return best_match

        return None

    def record_experience(self, quest_text: str, solution: dict, tags: Optional[List[str]] = None) -> None:
        """沉淀并存储一条新的成功解决经验。"""
        norm = self._normalize_key(quest_text)
        if not norm or not solution:
            return

        existing = self._memory.get(norm, {})
        count = int(existing.get("success_count", 0)) + 1

        record = {
            "action": solution.get("act", solution.get("action", "")),
            "reason": solution.get("reason", "来自自愈/AI会诊的成功经验沉淀"),
            "success_count": count,
            "updated_at": time.time(),
            "tags": tags or existing.get("tags", []),
        }
        self._memory[norm] = record
        self._save()
        if self.logger:
            self.logger.info(f"💡 [经验自学习] 已将【{norm}】的解决策略沉淀入本地经验库 (累计生效 {count} 次)")

    def record_failure_reflection(
        self,
        context: str,
        failure_type: str,
        lesson_learned: str,
    ) -> None:
        """
        L5 自省反思归因：记录一次失败或脱困事件，总结教训，防止同一个坑踩两次。
        """
        reflection = {
            "context": context,
            "failure_type": failure_type,
            "lesson_learned": lesson_learned,
            "timestamp": time.time(),
        }
        self._reflections.append(reflection)
        if len(self._reflections) > 50:
            self._reflections.pop(0)
        self._save()
        if self.logger:
            self.logger.warning(f"📝 [L5自省反思] 记录教训: 【{failure_type}】{lesson_learned}")

    def get_memory_stats(self) -> Dict[str, Any]:
        """导出经验库和反思库的统计数据供控制台看板展示。"""
        total_successes = sum(r.get("success_count", 1) for r in self._memory.values())
        return {
            "experience_count": len(self._memory),
            "total_reuse_count": total_successes,
            "reflection_count": len(self._reflections),
            "latest_reflection": self._reflections[-1]["lesson_learned"] if self._reflections else "暂无严重失误",
        }
