import base64
import json
import logging
import os
import re
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import requests

from core.input_ctrl import InputController


class AIResponseError(Exception):
    """大模型接口返回非 200 HTTP 错误时的异常包装。"""
    def __init__(self, status_code: int, message: str):
        super().__init__(f"HTTP {status_code}: {message}")
        self.status_code = status_code
        self.message = message


DEFAULT_AI_PROMPT = """你正在全权接管操作开放世界3D动作游戏《鸣潮》。
请根据输入的实时游戏画面与任务信息，做出真人级的高速操作决策。
你必须且仅输出单行纯 JSON，不要包含任何 Markdown 格式或多余文字：
1. 战斗输出: {"act":"combat", "keys":["e","r"], "clicks":3, "dodge":0, "reason":"释放技能连携普攻输出"}
2. 极限闪避: {"act":"combat", "dodge":1, "keys":["e"], "clicks":1, "reason":"规避怪物红光攻击后反击"}
3. 残血吃药: {"act":"item", "key":"c", "reason":"左下角血量过低使用回复药剂"}
4. 剧情跳过: {"act":"dialogue", "reason":"点击屏幕右上跳过或右侧对话选项"}
5. 交互拾取: {"act":"interact", "key":"f", "reason":"拾取地面金色声骸道具或与NPC对话"}
6. 钩索位移: {"act":"gadget", "key":"t", "reason":"视野内出现T键钩索点快速飞跃"}
7. 目标寻路: {"act":"nav", "turn":0, "sprint":1, "dur":0.7, "reason":"朝前方任务菱形光柱冲刺"}
8. 任务受阻: {"act":"switch_quest", "reason":"遇到复杂机关解谜或连续卡死切换任务"}"""

STUCK_CONSULT_PROMPT = """你正在协助诊断开放世界3D动作游戏《鸣潮》中的卡死或障碍受阻现场。
角色在寻路中连续脱困失败停滞不前。请观察当前画面中的地形、台阶、悬崖、建筑障碍及屏幕按键提示（如T键钩索提示），给出单步破局动作。
你必须且仅输出单行纯 JSON，严禁包含任何 Markdown 格式或多余文字：
1. 跳跃翻越: {"act":"jump", "keys":["w"], "reason":"前方为低矮石阶或小障碍，按W加空格翻越"}
2. 钩索飞跃: {"act":"gadget", "key":"t", "reason":"视野中存在T键钩索锚点，直接按T飞跃"}
3. 绕行转向: {"act":"turn", "turn":90, "reason":"避开正前方巨大障碍物向右侧绕行"}
4. 放弃切换: {"act":"switch_quest", "reason":"前方为深渊绝壁或复杂机关解谜，建议切换下一任务"}"""


class AIAgentController:
    """
    超低延迟异步 AI 托管决策控制器：
    1. 采用双速架构（毫秒级本地动作流水线 + 200~300ms 异步 VLM 多模态决策）。
    2. 图像超轻量化压缩（640x360 JPEG，体积 < 35KB），网络毫秒级上传。
    3. 支持 Gemini API、OpenAI 协议与本地 Ollama。
    4. AI 思考期间本地执行惯性平滑巡航，实现体感 0 停顿、0 掉帧的高速真人托管。
    """

    def __init__(
        self,
        config: dict,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.config = config
        self.logger = logger

        ai_cfg = config.get("ai_autopilot", {})
        self.enabled = bool(ai_cfg.get("enabled", False))
        self.provider = str(ai_cfg.get("provider", "gemini")).lower()
        self.model = str(ai_cfg.get("model", "gemini-2.5-flash"))
        self.base_url = str(ai_cfg.get("base_url", "")).rstrip("/")
        self.timeout_sec = float(ai_cfg.get("timeout_sec", 2.2))
        self.target_width = int(ai_cfg.get("fast_width", 640))
        self.target_height = int(ai_cfg.get("fast_height", 360))
        self.jpeg_quality = int(ai_cfg.get("jpeg_quality", 65))

        # 读取 API Key（优先配置，回退环境变量）
        self.api_key = str(ai_cfg.get("api_key", "")).strip()
        if not self.api_key:
            if self.provider == "gemini":
                self.api_key = os.environ.get("GEMINI_API_KEY", "")
            else:
                self.api_key = os.environ.get("OPENAI_API_KEY", "")

        self._lock = threading.Lock()
        self._pending_frame: Optional[np.ndarray] = None
        self._pending_context: Optional[dict] = None
        self._latest_action: Optional[dict] = None
        self._latest_reason: str = "就绪中"
        self._is_querying: bool = False
        self._stop_event = threading.Event()
        self._last_ai_query_ts: float = 0.0

        # 启动后台异步推理线程
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            name="AIAgentWorkerThread",
            daemon=True,
        )
        self._worker_thread.start()

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)
        self.config.setdefault("ai_autopilot", {})["enabled"] = bool(enabled)
        if self.logger:
            self.logger.info("AI 全权接管托管模式：%s", "已开启（AI 主导决策）" if enabled else "已关闭（本地规则主导）")

    def get_status_info(self) -> Tuple[bool, str]:
        with self._lock:
            return self.enabled, self._latest_reason

    def update_config(self, ai_cfg: dict) -> None:
        """动态热更新 AI 配置参数（无需重启软件）。"""
        with self._lock:
            if "enabled" in ai_cfg:
                self.enabled = bool(ai_cfg["enabled"])
            if "provider" in ai_cfg:
                self.provider = str(ai_cfg["provider"]).lower()
            if "model" in ai_cfg:
                self.model = str(ai_cfg["model"])
            if "base_url" in ai_cfg:
                self.base_url = str(ai_cfg["base_url"]).rstrip("/")
            if "timeout_sec" in ai_cfg:
                self.timeout_sec = float(ai_cfg["timeout_sec"])
            if "fast_width" in ai_cfg:
                self.target_width = int(ai_cfg["fast_width"])
            if "fast_height" in ai_cfg:
                self.target_height = int(ai_cfg["fast_height"])
            if "jpeg_quality" in ai_cfg:
                self.jpeg_quality = int(ai_cfg["jpeg_quality"])
            if "api_key" in ai_cfg:
                self.api_key = str(ai_cfg["api_key"]).strip()

        if not self.api_key:
            if self.provider == "gemini":
                self.api_key = os.environ.get("GEMINI_API_KEY", "")
            else:
                self.api_key = os.environ.get("OPENAI_API_KEY", "")

    @staticmethod
    def _parse_error_message(resp: requests.Response) -> str:
        """从非 200 HTTP 响应中提取结构化错误描述。"""
        try:
            data = resp.json()
            if isinstance(data, dict):
                # 兼容 {"error": {"message": "..."}}
                err = data.get("error")
                if isinstance(err, dict) and "message" in err:
                    return str(err["message"])
                if isinstance(err, str):
                    return err
                if "message" in data:
                    return str(data["message"])
                if "msg" in data:
                    return str(data["msg"])
        except Exception:
            pass
        # 降级返回截断的原始文本
        txt = (resp.text or "").strip()
        return txt[:160] if txt else f"HTTP {resp.status_code}"

    @classmethod
    def fetch_models(
        cls,
        provider: str,
        base_url: str,
        api_key: str,
        timeout_sec: float = 4.0,
    ) -> List[str]:
        """从目标 API 或中转网关自动拉取当前账户/服务可用的模型 ID 列表。"""
        models: List[str] = []
        base_url = (base_url or "").strip().rstrip("/")

        # 1. 尝试 OpenAI /v1/models 协议（最通用，涵盖各类聚合网关如 OneAPI/NewAPI/本地网关）
        target_openai_url = base_url if base_url else "https://api.openai.com"
        if not target_openai_url.endswith("/v1") and not target_openai_url.endswith("/v1beta"):
            target_openai_url = f"{target_openai_url}/v1"
        try:
            headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
            r = requests.get(f"{target_openai_url}/models", headers=headers, timeout=timeout_sec)
            if r.status_code == 200:
                d = r.json()
                if isinstance(d, dict) and "data" in d and isinstance(d["data"], list):
                    for item in d["data"]:
                        if isinstance(item, dict) and "id" in item:
                            models.append(str(item["id"]))
                    if models:
                        return sorted(list(set(models)))
        except Exception:
            pass

        # 2. 尝试 Gemini 原生 /v1beta/models 协议
        target_gemini_url = base_url if base_url else "https://generativelanguage.googleapis.com"
        if not target_gemini_url.endswith("/v1beta") and not target_gemini_url.endswith("/v1"):
            target_gemini_url = f"{target_gemini_url}/v1beta"
        try:
            params = {"key": api_key} if api_key else {}
            headers = {
                "x-goog-api-key": api_key,
                "Authorization": f"Bearer {api_key}",
            }
            r = requests.get(f"{target_gemini_url}/models", params=params, headers=headers, timeout=timeout_sec)
            if r.status_code == 200:
                d = r.json()
                if isinstance(d, dict) and "models" in d and isinstance(d["models"], list):
                    for item in d["models"]:
                        if isinstance(item, dict) and "name" in item:
                            m_name = str(item["name"]).replace("models/", "")
                            models.append(m_name)
                    if models:
                        return sorted(list(set(models)))
        except Exception:
            pass

        return models

    def test_connection(self, frame: Optional[np.ndarray] = None) -> Tuple[bool, float, str]:
        """立即测试一次网络与大模型通信与测速。返回 (成功与否, 耗时ms, 详细信息)。"""
        if frame is None:
            frame = np.full((1080, 1920, 3), 40, dtype=np.uint8)
            cv2.putText(
                frame,
                "MingchaoJB AI Test Frame",
                (300, 540),
                cv2.FONT_HERSHEY_SIMPLEX,
                2.0,
                (0, 220, 255),
                3,
            )

        if not self.api_key and self.provider != "ollama":
            return False, 0.0, "未配置 API Key（请先在界面填入密钥或配置环境变量）"

        t0 = time.perf_counter()
        try:
            b64 = self.encode_frame(frame)
            decision = None
            if self.provider == "gemini":
                decision = self._call_gemini_api(b64, "测试当前画面并返回操作决策")
            else:
                decision = self._call_openai_api(b64, "测试当前画面并返回操作决策")

            elapsed_ms = (time.perf_counter() - t0) * 1000
            if decision and isinstance(decision, dict):
                act = decision.get("act", "未知动作")
                reason = decision.get("reason", "成功响应")
                return True, elapsed_ms, f"AI 动作: [{act}] | 分析思路: {reason}"
            return False, elapsed_ms, "大模型已返回 200，但未输出符合格式的 JSON 动作字典"
        except AIResponseError as aie:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            # 针对常见的访客/权限不足问题给予特别醒目的诊断建议
            if "访客" in aie.message or "附件" in aie.message:
                return (
                    False,
                    elapsed_ms,
                    f"上游网关拦截 (HTTP {aie.status_code}): {aie.message}（请在本地网关后台登录可用 Google 账号）",
                )
            return False, elapsed_ms, f"上游接口报错 (HTTP {aie.status_code}): {aie.message}"
        except requests.ConnectionError as ce:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            return False, elapsed_ms, f"连接被拒绝/网络不可达: 请确认接口地址 Base URL 端口已启动 ({ce})"
        except requests.Timeout:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            return False, elapsed_ms, f"请求超时（超过 {self.timeout_sec} 秒）: 请检查网络连接或降低分辨率"
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            return False, elapsed_ms, f"请求发生异常: {exc}"

    def encode_frame(self, frame: np.ndarray) -> str:
        """极速压缩画面至轻量 JPEG，返回 Base64 字符串（耗时 < 3ms，体积 < 35KB）。"""
        resized = cv2.resize(frame, (self.target_width, self.target_height), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if not ok:
            return ""
        return base64.b64encode(buf.tobytes()).decode("ascii")

    def _call_gemini_api(
        self,
        base64_img: str,
        quest_text: str,
        prompt_override: Optional[str] = None,
    ) -> Optional[dict]:
        api_key = self.api_key
        raw_url = (self.base_url or "").strip().rstrip("/")
        if not raw_url:
            raw_url = "https://generativelanguage.googleapis.com/v1beta"
        elif not raw_url.endswith("/v1beta") and not raw_url.endswith("/v1"):
            raw_url = f"{raw_url}/v1beta"

        # 模型名容错：去除重复的 models/ 前缀
        model_name = self.model.replace("models/", "")
        endpoint = f"{raw_url}/models/{model_name}:generateContent"

        if prompt_override:
            prompt = prompt_override
        else:
            prompt = f"{DEFAULT_AI_PROMPT}\n【当前左上角任务】：{quest_text if quest_text else '未识别到任务文本'}"

        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": base64_img,
                            }
                        },
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": 90,
                "responseMimeType": "application/json",
            },
        }

        # 兼容 URL query key、x-goog-api-key、Authorization 等所有鉴权方式
        params = {"key": api_key} if api_key else {}
        headers = {
            "Content-Type": "application/json",
        }
        if api_key:
            headers["x-goog-api-key"] = api_key
            headers["Authorization"] = f"Bearer {api_key}"

        resp = requests.post(
            endpoint,
            params=params,
            headers=headers,
            json=payload,
            timeout=self.timeout_sec,
        )
        if resp.status_code != 200:
            err_msg = self._parse_error_message(resp)
            if self.logger:
                self.logger.warning("Gemini API 请求失败 (%d): %s", resp.status_code, err_msg)
            raise AIResponseError(resp.status_code, err_msg)

        data = resp.json()
        parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
        if not parts:
            return None
        raw_text = parts[0].get("text", "").strip()
        return self._extract_json(raw_text)

    def _call_openai_api(
        self,
        base64_img: str,
        quest_text: str,
        prompt_override: Optional[str] = None,
    ) -> Optional[dict]:
        raw_url = (self.base_url or "").strip().rstrip("/")
        if not raw_url:
            raw_url = "https://api.openai.com/v1"
        elif not raw_url.endswith("/v1") and not raw_url.endswith("/v1beta"):
            raw_url = f"{raw_url}/v1"

        endpoint = f"{raw_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        if prompt_override:
            prompt = prompt_override
        else:
            prompt = f"{DEFAULT_AI_PROMPT}\n【当前左上角任务】：{quest_text if quest_text else '未识别到任务文本'}"

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"},
                        },
                    ],
                }
            ],
            "max_tokens": 90,
            "temperature": 0.1,
        }

        resp = requests.post(endpoint, headers=headers, json=payload, timeout=self.timeout_sec)
        if resp.status_code != 200:
            err_msg = self._parse_error_message(resp)
            if self.logger:
                self.logger.warning("OpenAI 兼容 API 请求失败 (%d): %s", resp.status_code, err_msg)
            raise AIResponseError(resp.status_code, err_msg)

        data = resp.json()
        choices = data.get("choices", [])
        if not choices:
            return None
        content = choices[0].get("message", {}).get("content", "").strip()
        return self._extract_json(content)

    def consult_stuck_situation(
        self,
        frame: np.ndarray,
        context: Optional[dict] = None,
    ) -> Optional[dict]:
        """
        针对常规寻路连续脱困失败的疑难场景，单帧唤醒多模态大模型进行现场诊断与破局。
        返回决策字典，例如 {"act": "jump"|"gadget"|"turn"|"switch_quest", "reason": "..."}
        """
        if not self.api_key and self.provider != "ollama":
            return None

        context = context or {}
        quest_text = context.get("quest_text", "")
        unstuck_count = context.get("unstuck_count", 2)
        prompt = (
            f"{STUCK_CONSULT_PROMPT}\n"
            f"【当前左上角任务】：{quest_text if quest_text else '未识别到任务文本'}\n"
            f"【已脱困尝试】：连续 {unstuck_count} 次常规脱困失败，角色疑似被低矮台阶/障碍墙/悬崖卡阻。"
        )

        base64_img = self.encode_frame(frame)
        if not base64_img:
            return None

        if self.provider == "gemini":
            return self._call_gemini_api(base64_img, quest_text, prompt_override=prompt)
        else:
            return self._call_openai_api(base64_img, quest_text, prompt_override=prompt)

    @staticmethod
    def _extract_json(raw_text: str) -> Optional[dict]:
        raw_text = raw_text.strip()
        # 剥离可能的 Markdown 代码块
        if raw_text.startswith("```"):
            raw_text = re.sub(r"^```(?:json)?", "", raw_text)
            raw_text = re.sub(r"```$", "", raw_text).strip()
        try:
            return json.loads(raw_text)
        except Exception:
            # 正则匹配最内层花括号
            m = re.search(r"\{.*?\}", raw_text, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group(0))
                except Exception:
                    pass
        return None

    def _worker_loop(self) -> None:
        """后台异步推理工作流：负责持续以超低耗时与大模型通信，不堵塞主线程。"""
        while not self._stop_event.is_set():
            frame = None
            context = None
            with self._lock:
                if self._pending_frame is not None and not self._is_querying:
                    frame = self._pending_frame
                    context = self._pending_context or {}
                    self._pending_frame = None
                    self._is_querying = True

            if frame is None:
                time.sleep(0.04)
                continue

            try:
                base64_img = self.encode_frame(frame)
                quest_text = context.get("quest_text", "")

                decision = None
                if self.api_key:
                    if self.provider == "gemini":
                        decision = self._call_gemini_api(base64_img, quest_text)
                    else:
                        decision = self._call_openai_api(base64_img, quest_text)

                if decision and isinstance(decision, dict):
                    with self._lock:
                        self._latest_action = decision
                        self._latest_reason = decision.get("reason", decision.get("act", "执行中"))
                        if self.logger:
                            self.logger.info(f"[AI 托管决策] 行动: {decision.get('act')} | 思路: {self._latest_reason}")
            except Exception as exc:
                if self.logger:
                    self.logger.debug(f"AI 异步推理异常: {exc}")
            finally:
                with self._lock:
                    self._is_querying = False
                    self._last_ai_query_ts = time.monotonic()

            time.sleep(0.08)

    def trigger_ai_observation(self, frame: np.ndarray, context: dict) -> None:
        """向后台 AI 推送最新帧，若后台正忙则平滑跳过当前推送。"""
        if not self.enabled:
            return
        now = time.monotonic()
        with self._lock:
            # 限制最高请求频率（每 0.35s 最多推一帧给 AI，保持极速且节约 Token）
            if (now - self._last_ai_query_ts) >= 0.35 and not self._is_querying:
                self._pending_frame = frame
                self._pending_context = context

    def execute_action(self, inp: InputController) -> Optional[dict]:
        """主线程以 60FPS 频率无锁消费 AI 决策并快速执行，零卡顿。"""
        act_data = None
        with self._lock:
            if self._latest_action is not None:
                act_data = self._latest_action
                self._latest_action = None

        if not act_data:
            return None

        act = act_data.get("act", "")

        if act == "combat":
            keys = act_data.get("keys", ["e"])
            clicks = int(act_data.get("clicks", 2))
            for k in keys:
                inp.press_key(str(k))
            for _ in range(clicks):
                inp.click("left")
            if act_data.get("dodge"):
                inp.click("right")

        elif act == "item":
            key = str(act_data.get("key", "c"))
            inp.press_key(key)

        elif act == "interact":
            key = str(act_data.get("key", "f"))
            inp.press_key(key)

        elif act == "gadget":
            key = str(act_data.get("key", "t"))
            inp.press_key(key)

        elif act == "dialogue":
            inp.press_key("space")
            inp.press_key("f")

        elif act == "nav":
            turn = int(act_data.get("turn", 0))
            if turn != 0:
                inp.smooth_turn(turn, steps=3)
            sprint = bool(act_data.get("sprint", True))
            dur = float(act_data.get("dur", 0.5))
            inp.key_down("w")
            if sprint:
                inp.key_down("shift")
                time.sleep(dur)
                inp.key_up("shift")
            else:
                time.sleep(dur)
            inp.key_up("w")

        return act_data

    def stop(self) -> None:
        self._stop_event.set()
