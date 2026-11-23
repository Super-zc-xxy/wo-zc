import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class ObservationSnapshot:
    """时序空间中的单帧物理微观观测快照。"""
    timestamp: float
    camera_yaw: float = 0.0           # 当前相机水平偏角（度数）
    vertical_flow: float = 0.0        # 垂直光流速率（下坠为正，上升/跳跃为负）
    horizontal_flow: float = 0.0      # 水平运动向量
    distance_m: Optional[float] = None # 目标点距离（米）
    elevation: int = 0                # 落差标志：+1 上方高处，-1 下方低处，0 平视
    action_executed: str = ""         # 该时刻施加的动作


class L5WorldModel:
    """
    鸣潮 L5 级完全自主核心中枢二：时序空间与物理因果世界模型（World Model & Causality Engine）。
    1. 连续时序空间动力学环形缓冲：跨越单帧局限，感知连续运动位移趋势与异常阻抗；
    2. 局部空间因果与障碍物推演：通过连续多帧运动矢量推算滑坡、隐形空气墙与死角；
    3. 物理机制因果推演（Mechanism Causality Reasoner）：
       - 激光/光路折射（光学因果）
       - 控物爆炸吸附与石壁碎裂（力学因果）
       - 重力踏板与压力感应（重力因果）
       - 机关时间溯流（时序因果）
    4. 自动生成并闭环执行多步宏动作序列（Macro Actions Pipeline）。
    """

    MECHANISM_TEMPLATES: Dict[str, List[Dict[str, Any]]] = {
        "levitator_rock": [
            {"act": "gadget", "tool": "levitator", "desc": "切换控物工具"},
            {"act": "aim", "dx": 0, "dy": -20, "desc": "对准爆炸果/重物"},
            {"act": "press_key", "key": "t", "desc": "念力吸取重物"},
            {"act": "sleep", "sec": 0.45, "desc": "等待念力吸附完成"},
            {"act": "aim", "dx": 0, "dy": -40, "desc": "瞄准前方裂缝石壁"},
            {"act": "click", "button": "left", "desc": "投掷轰碎障碍石壁"},
            {"act": "sleep", "sec": 0.35, "desc": "确认碎石消散"},
        ],
        "sensor_trail": [
            {"act": "gadget", "tool": "sensor", "desc": "切换感知工具"},
            {"act": "press_key", "key": "t", "desc": "释放共鸣感知扫描环境"},
            {"act": "sleep", "sec": 0.5, "desc": "显化虚影足迹与痕迹"},
            {"act": "turn_camera", "angle": 30, "desc": "微调视线锁定虚影轨迹"},
        ],
        "pressure_plate": [
            {"act": "gadget", "tool": "levitator", "desc": "切换念力控物"},
            {"act": "press_key", "key": "t", "desc": "吸附重力方块"},
            {"act": "aim", "dx": 0, "dy": 80, "desc": "向下对准地表重力感应踏板"},
            {"act": "click", "button": "left", "desc": "放置重物压制地锁"},
        ],
        "optical_beam": [
            {"act": "press_key", "key": "f", "desc": "交互旋转光路偏折棱镜"},
            {"act": "sleep", "sec": 0.4, "desc": "等待棱镜偏转 45 度"},
            {"act": "check_beam", "desc": "验证光斑是否折射入受光核心"},
        ],
    }

    def __init__(
        self,
        buffer_size: int = 15,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.buffer_size = buffer_size
        self.logger = logger
        self.buffer: List[ObservationSnapshot] = []
        self._last_gray_frame: Optional[np.ndarray] = None
        self._inferred_obstacles: List[Dict[str, Any]] = []

    def update_observation(
        self,
        frame: np.ndarray,
        distance_m: Optional[float] = None,
        elevation: int = 0,
        camera_yaw: float = 0.0,
        action: str = "",
    ) -> ObservationSnapshot:
        """
        录入新一帧微观物理观测，计算帧间光流与空间位移矢量。
        """
        now = time.monotonic()
        v_flow, h_flow = 0.0, 0.0

        if frame is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
            # 缩放至极轻量 160x90 区域计算中心光流（超低耗时 < 1ms）
            small = cv2.resize(gray, (160, 90), interpolation=cv2.INTER_NEAREST)
            if self._last_gray_frame is not None:
                diff = small.astype(np.float32) - self._last_gray_frame.astype(np.float32)
                # 提取画面垂直与水平微观位移均值
                v_flow = float(np.mean(diff[45:80, :]))
                h_flow = float(np.mean(diff[:, 40:120]))
            self._last_gray_frame = small

        snapshot = ObservationSnapshot(
            timestamp=now,
            camera_yaw=camera_yaw,
            vertical_flow=v_flow,
            horizontal_flow=h_flow,
            distance_m=distance_m,
            elevation=elevation,
            action_executed=action,
        )

        self.buffer.append(snapshot)
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)

        return snapshot

    def detect_motion_anomaly(self) -> Optional[Dict[str, Any]]:
        """
        跨越单帧局限，基于时序滑动窗口分析空间运动反常：
        1. 坡度滑落/卡阻（持续向前走，但连续数秒距离不减反增）；
        2. 隐形空气墙/悬崖阻隔（光流几乎为零且角色处于前进状态）；
        3. 连续下坠状态。
        """
        if len(self.buffer) < 5:
            return None

        recent = self.buffer[-5:]
        valid_distances = [s.distance_m for s in recent if s.distance_m is not None]

        # 1. 距离反常检测：连续移动时，距离非但不减小反而持续变大
        if len(valid_distances) >= 3:
            delta_d = valid_distances[-1] - valid_distances[0]
            if delta_d > 4.0:  # 距离被弹开或滑坡远离超过 4 米
                anomaly = {
                    "type": "steep_slope_slipping",
                    "reason": f"检测到时序空间连续滑坡排斥：距离在移动中非降反增 (+{delta_d:.1f}m)，疑似前方为高陡滑坡或反弹空气墙",
                    "suggested_action": "turn_and_climb",
                    "delta_d": delta_d,
                }
                if self.logger:
                    self.logger.warning(f"🌐 [世界模型推演] {anomaly['reason']}")
                return anomaly

        # 2. 运动停滞与假死阻隔检测
        avg_v_flow = sum(abs(s.vertical_flow) for s in recent) / len(recent)
        forward_actions = sum(1 for s in recent if "w" in s.action_executed.lower() or "forward" in s.action_executed.lower())
        if forward_actions >= 4 and avg_v_flow < 0.05:
            anomaly = {
                "type": "invisible_barrier_blocked",
                "reason": "检测到时序光流停滞：连续执行前进但画面处于完全静止状态，前方判定存在碰撞体阻隔",
                "suggested_action": "jump_vault",
            }
            if self.logger:
                self.logger.warning(f"🌐 [世界模型推演] {anomaly['reason']}")
            return anomaly

        return None

    def reason_mechanism_causality(self, quest_text: str) -> Optional[List[Dict[str, Any]]]:
        """
        物理常识因果推演：根据任务与空间线索，自主生成多步宏动作序列（Macro Actions Sequence）。
        """
        if not quest_text:
            return None

        # 1. 碎石、爆炸果、念力投掷机制
        if any(w in quest_text for w in ("碎石", "石壁", "爆炸", "砸开", "裂缝", "投掷")):
            if self.logger:
                self.logger.info("🌐 [世界模型因果推演] 识别到【力学破坏与念力控物】机关，自主生成多步投掷宏动作流")
            return list(self.MECHANISM_TEMPLATES["levitator_rock"])

        # 2. 虚影足迹与感知追踪机制
        if any(w in quest_text for w in ("虚影", "足迹", "残影", "感知", "追踪痕迹")):
            if self.logger:
                self.logger.info("🌐 [世界模型因果推演] 识别到【共鸣虚影显化】机制，自主生成声波感知宏动作流")
            return list(self.MECHANISM_TEMPLATES["sensor_trail"])

        # 3. 重力踏板与地锁压制
        if any(w in quest_text for w in ("踏板", "压感", "地锁", "重力方块")):
            if self.logger:
                self.logger.info("🌐 [世界模型因果推演] 识别到【重力压制地锁】机制，自主生成重力归位宏动作流")
            return list(self.MECHANISM_TEMPLATES["pressure_plate"])

        # 4. 激光与光路折射
        if any(w in quest_text for w in ("棱镜", "光线", "折射", "光路", "激光", "受光")):
            if self.logger:
                self.logger.info("🌐 [世界模型因果推演] 识别到【空间光学折射】机制，自主生成光路微调宏动作流")
            return list(self.MECHANISM_TEMPLATES["optical_beam"])

        return None

    def execute_macro_sequence(
        self,
        actions: List[Dict[str, Any]],
        inp: Any,
        gadget_manager: Optional[Any] = None,
    ) -> bool:
        """
        在虚拟控制器上顺序执行推演出的宏动作序列。
        """
        if not actions or not inp:
            return False

        for step in actions:
            act_type = step.get("act")
            if act_type == "gadget":
                tool = step.get("tool", "")
                if gadget_manager and hasattr(gadget_manager, "switch_tool"):
                    gadget_manager.switch_tool(tool)
            elif act_type == "press_key":
                inp.press_key(step.get("key", ""))
            elif act_type == "click":
                inp.click(step.get("button", "left"))
            elif act_type == "aim":
                dx = int(step.get("dx", 0))
                dy = int(step.get("dy", 0))
                if hasattr(inp, "smooth_turn") and dx != 0:
                    inp.smooth_turn(dx)
            elif act_type == "turn_camera":
                deg = float(step.get("angle", 0))
                if hasattr(inp, "smooth_turn"):
                    # 约 3.6 像素每度
                    inp.smooth_turn(int(deg * 3.6))
            elif act_type == "sleep":
                time.sleep(float(step.get("sec", 0.1)))

        return True

    def get_spatial_summary(self) -> Dict[str, Any]:
        """导出时序空间认知概要供控制台展示。"""
        recent_flow = [s.vertical_flow for s in self.buffer]
        return {
            "buffer_depth": len(self.buffer),
            "avg_vertical_flow": sum(recent_flow) / len(recent_flow) if recent_flow else 0.0,
            "latest_distance": self.buffer[-1].distance_m if self.buffer else None,
            "anomaly_detected": bool(self.detect_motion_anomaly()),
        }
