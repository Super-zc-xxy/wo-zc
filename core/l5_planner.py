import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class GoalType(Enum):
    DAILY_COMMISSIONS = "daily_commissions"   # 活跃度与 4 个日常委托
    RESIN_DUNGEONS = "resin_dungeons"         # 结晶溶剂与秘境/无音区体力清空
    QUEST_EXPLORATION = "quest_exploration"   # 大世界主支线追踪与探索度推进
    ECHO_HUNTING = "echo_hunting"             # 稀有与高价值声骸巡回捕获
    CHARACTER_BUILD = "character_build"       # 专属角色练度与材料收集
    EMERGENCY_INTERRUPT = "emergency_interrupt" # 突发紧急事件插队（神像满血/闪光声骸）
    CUSTOM_INTENT = "custom_intent"           # 自然语言泛化意图


class SubGoalStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class SubGoal:
    id: str
    title: str
    goal_type: GoalType
    priority: int = 10                        # 优先级：数值越小越优先（0~100）
    params: Dict[str, Any] = field(default_factory=dict)
    status: SubGoalStatus = SubGoalStatus.PENDING
    context_snapshot: Optional[Dict[str, Any]] = None  # 打断暂停时的现场上下文快照
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    reason: str = ""


class L5HierarchicalPlanner:
    """
    鸣潮 L5 级完全自主核心大脑：长程意图与多层级目标规划器（Hierarchical Goal Planner）。
    1. 将高阶自然语言大目标（如“清日常与体力”、“探索乘霄山”、“拉满今汐练度”）智能拆解为有序 DAG 子目标流；
    2. 支持大世界突发高收益事件（闪光声骸、未点亮神像、濒危免费复苏）高优先度动态插队；
    3. 支持断点现场快照保存与无缝无感恢复执行；
    4. 实时向控制台驾驶舱输出'自主思考流（Thought Stream）'。
    """

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self.logger = logger
        self.master_intent: str = "默认自主托管（日常与探索）"
        self.subgoals: List[SubGoal] = []
        self._paused_stack: List[SubGoal] = []
        self._completed_history: List[SubGoal] = []
        self._thought_stream: List[str] = []
        self._init_default_plan()

    def _log_thought(self, thought: str) -> None:
        """记录 AI 思考流日志并推送到驾驶舱。"""
        ts = time.strftime("%H:%M:%S")
        entry = f"[{ts}] 🧠 [L5思考] {thought}"
        self._thought_stream.append(entry)
        if len(self._thought_stream) > 100:
            self._thought_stream.pop(0)
        if self.logger:
            self.logger.info(entry)

    def _init_default_plan(self) -> None:
        """初始化默认多层级自适应执行链。"""
        self.subgoals = [
            SubGoal(
                id="goal_daily",
                title="活跃度日常委托清空 (4/4)",
                goal_type=GoalType.DAILY_COMMISSIONS,
                priority=10,
                params={"max_count": 4},
                reason="每日 5 点刷新高优先级资源",
            ),
            SubGoal(
                id="goal_resin",
                title="结晶溶剂与秘境体力清空",
                goal_type=GoalType.RESIN_DUNGEONS,
                priority=20,
                params={"target": "auto_highest_echo"},
                reason="防止体力溢出造成资源浪费",
            ),
            SubGoal(
                id="goal_quest",
                title="大世界主支线与探索度自动推进",
                goal_type=GoalType.QUEST_EXPLORATION,
                priority=30,
                params={"auto_switch": True},
                reason="推进世界等级与解锁信标",
            ),
            SubGoal(
                id="goal_echo",
                title="沿途全自动声骸定向吸收与开箱",
                goal_type=GoalType.ECHO_HUNTING,
                priority=40,
                params={"rarity": ["gold", "purple"]},
                reason="持续沉淀顶级声骸胚子",
            ),
        ]
        self._log_thought("已规划默认全局执行链路：日常委托 -> 体力清空 -> 大世界主支线 -> 声骸收益")

    def set_master_goal(self, intent_text: str) -> None:
        """
        根据用户自然语言或预设指令动态重构全局规划。
        """
        self.master_intent = intent_text.strip()
        self.subgoals.clear()
        self._paused_stack.clear()

        intent_lower = self.master_intent.lower()
        self._log_thought(f"接收到高阶大目标指令: 【{self.master_intent}】，正在推演最优执行子目标树...")

        if any(name in intent_lower for name in ("今汐", "忌炎", "长离", "椿")):
            char_name = "今汐" if "今汐" in intent_lower else ("忌炎" if "忌炎" in intent_lower else ("椿" if "椿" in intent_lower else "长离"))
            self.subgoals.append(
                SubGoal(f"goal_build_{char_name}", f"【{char_name}】专属突破素材与声骸定向收集", GoalType.CHARACTER_BUILD, priority=5, params={"character": char_name})
            )
            if "体力" in intent_lower or "秘境" in intent_lower or "无音区" in intent_lower:
                self.subgoals.append(
                    SubGoal(f"goal_resin_{char_name}", f"清空体力刷取【{char_name}】天赋与突破素材", GoalType.RESIN_DUNGEONS, priority=10, params={"character": char_name})
                )
            else:
                self.subgoals.append(
                    SubGoal("goal_quest_follow", "材料满足后继续大世界探索", GoalType.QUEST_EXPLORATION, priority=20)
                )
        elif "体力" in intent_lower or "秘境" in intent_lower or "无音区" in intent_lower:
            self.subgoals.append(
                SubGoal("goal_resin_prio", "优先清空当前结晶溶剂体力", GoalType.RESIN_DUNGEONS, priority=5)
            )
            self.subgoals.append(
                SubGoal("goal_quest_follow", "体力耗尽后推进大世界探索", GoalType.QUEST_EXPLORATION, priority=20)
            )
        elif "声骸" in intent_lower or "刷怪" in intent_lower:
            self.subgoals.append(
                SubGoal("goal_echo_prio", "全大世界高价值金色声骸巡回猎杀", GoalType.ECHO_HUNTING, priority=5)
            )
        elif "日常" in intent_lower:
            self.subgoals.append(
                SubGoal("goal_daily_prio", "极速完成 4 个活跃度日常任务", GoalType.DAILY_COMMISSIONS, priority=5)
            )
            self.subgoals.append(
                SubGoal("goal_quest_follow", "日常完成后巡航大世界任务", GoalType.QUEST_EXPLORATION, priority=20)
            )
        else:
            self._init_default_plan()

        self._log_thought(f"子目标树推演完成，当前队列待办数: {len(self.subgoals)}")

    def get_current_subgoal(self) -> Optional[SubGoal]:
        """获取当前正在执行或下一个待执行的最高优先级子目标。"""
        if not self.subgoals:
            return None
        # 按照优先级升序排列（数值越小越紧急）
        self.subgoals.sort(key=lambda g: g.priority)
        current = self.subgoals[0]
        if current.status == SubGoalStatus.PENDING:
            current.status = SubGoalStatus.IN_PROGRESS
            self._log_thought(f"开始执行子目标: [{current.title}] (优先级: {current.priority})")
        return current

    def trigger_interrupt(
        self,
        title: str,
        reason: str,
        priority: int = 1,
        action_params: Optional[Dict[str, Any]] = None,
        context_snapshot: Optional[Dict[str, Any]] = None,
    ) -> SubGoal:
        """
        突发事件高优先度插队（例如：发现闪光声骸、附近发现未点亮界碑神像、濒死需免费满血）。
        将当前进行中的任务暂挂并保存现场快照，插队执行突发任务。
        """
        action_params = action_params or {}
        context_snapshot = context_snapshot or {}

        if self.subgoals:
            active_goal = self.subgoals[0]
            if active_goal.status in (SubGoalStatus.IN_PROGRESS, SubGoalStatus.PENDING):
                active_goal.status = SubGoalStatus.PAUSED
                active_goal.context_snapshot = context_snapshot
                self._paused_stack.append(active_goal)
                self.subgoals.pop(0)
                self._log_thought(
                    f"⚠️ 触发动态插队！已暂挂当前目标 [{active_goal.title}] 并封存现场快照: {list(context_snapshot.keys())}"
                )

        interrupt_goal = SubGoal(
            id=f"interrupt_{int(time.time() * 1000)}",
            title=title,
            goal_type=GoalType.EMERGENCY_INTERRUPT,
            priority=priority,
            params=action_params,
            status=SubGoalStatus.IN_PROGRESS,
            reason=reason,
        )
        self.subgoals.insert(0, interrupt_goal)
        self._log_thought(f"🚨 突发高优先目标已置顶插队: 【{title}】 原因: {reason} (优先级: {priority})")
        return interrupt_goal

    def complete_current_subgoal(self, status: SubGoalStatus = SubGoalStatus.COMPLETED) -> Optional[SubGoal]:
        """
        当前最高优先级子目标达成，自动出队；若有被暂挂的被打断目标，自动无缝恢复现场！
        """
        if not self.subgoals:
            return None

        finished = self.subgoals.pop(0)
        finished.status = status
        finished.completed_at = time.time()
        self._completed_history.append(finished)
        self._log_thought(f"✅ 子目标顺利达成: [{finished.title}] (耗时: {finished.completed_at - finished.created_at:.1f}s)")

        # 若存在被暂停打断的历史现场，立即恢复
        if self._paused_stack:
            resumed_goal = self._paused_stack.pop()
            resumed_goal.status = SubGoalStatus.IN_PROGRESS
            self.subgoals.insert(0, resumed_goal)
            self._log_thought(f"🔄 现场恢复：已自动唤醒此前被暂挂的目标 [{resumed_goal.title}]，恢复断点继续推进！")
            return resumed_goal

        return self.get_current_subgoal()

    def get_thought_stream(self) -> List[str]:
        """获取最近的思考流记录。"""
        return list(self._thought_stream)

    def get_plan_summary(self) -> Dict[str, Any]:
        """导出当前多层级规划运行状态概要，供控制台看板可视化渲染。"""
        current = self.subgoals[0] if self.subgoals else None
        return {
            "master_intent": self.master_intent,
            "current_goal": current.title if current else "无（全部目标已达成）",
            "current_priority": current.priority if current else None,
            "pending_count": len(self.subgoals),
            "paused_stack_depth": len(self._paused_stack),
            "completed_count": len(self._completed_history),
            "latest_thoughts": self._thought_stream[-5:],
        }
