
import json
import logging
import os
from pathlib import Path
from typing import Any, Optional
import requests
from datetime import datetime
from omegaconf import DictConfig
from rlinf.data.io_struct import RolloutResult
from .base_api_worker import BaseAPIWorker


class HttpAPIWorker(BaseAPIWorker):
    """基于 HTTP 的 API 工作器，处理 Rollout 输出、奖励结果与环境指标并记录数据"""

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.api_url = cfg.apiworkers.api_url
        self.api_key = cfg.apiworkers.get("api_key")
        self.timeout = cfg.apiworkers.get("timeout", 30)
        self.headers = self._init_headers()
        
        # 数据记录配置
        self.record_enabled = cfg.apiworkers.get("record_enabled", True)
        self.record_dir = self._init_record_dir(cfg)
        
        # 缓冲区
        self.reward_buffer = {}
        self.env_metrics_buffer = {}

    def _init_record_dir(self, cfg: DictConfig) -> Path:
        """初始化记录文件保存目录"""
        if not self.record_enabled:
            return None
            
        # 从配置获取路径，默认当前目录下"api_worker_records"文件夹
        base_path = Path(cfg.apiworkers.get("record_dir", "./api_worker_records"))
        # 创建带时间戳的子目录，避免覆盖
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        record_dir = base_path / f"api_records_{timestamp}"
        record_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"数据记录已启用，保存路径：{record_dir}")
        return record_dir

    def _init_headers(self) -> dict:
        """初始化 HTTP 请求头"""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def init_worker(self):
        """初始化工作器：验证 API 连接性"""
        try:
            response = requests.get(
                f"{self.api_url}/health",
                headers=self.headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            logging.info(f"Successfully connected to API at {self.api_url}")
        except Exception as e:
            logging.error(f"API connection test failed: {str(e)}")
            raise  # 初始化失败时终止

    def _format_rollout_data(self, rollout_result: RolloutResult, 
                            reward_result: Optional[dict] = None,
                            env_metrics: Optional[dict] = None) -> dict:
        """将 RolloutResult、奖励结果和环境指标格式化为 API 所需输入"""
        return {
            "num_sequence": rollout_result.num_sequence,
            "group_size": rollout_result.group_size,
            "prompts": rollout_result.prompt_texts,
            "responses": rollout_result.response_texts,
            "rewards": {
                "scores": rollout_result.rewards.tolist() if rollout_result.rewards is not None else None,
                "details": reward_result
            },
            "env_metrics": env_metrics,
            "timestamps": {
                "rollout": [self._get_timestamp()] * rollout_result.num_sequence,
                "processed": self._get_timestamp()
            }
        }

    def _get_timestamp(self) -> str:
        """获取当前时间戳（UTC）"""
        return datetime.utcnow().isoformat()

    def _record_data(self, data_type: str, data: dict):
        """
        记录数据到JSONL文件
        :param data_type: 数据类型（input/receive）
        :param data: 要记录的数据
        """
        if not self.record_enabled or not self.record_dir:
            return
            
        # 按数据类型分文件记录
        filename = f"{data_type}_records.jsonl"
        record_path = self.record_dir / filename
        
        # 添加记录元数据
        record = {
            "timestamp": self._get_timestamp(),
            "data_type": data_type,
            "content": data
        }
        
        # 写入JSONL文件（追加模式）
        with open(record_path, "a", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False)
            f.write("\n")

    def process_rollout(self, rollout_result: RolloutResult,
                       reward_result: Optional[dict] = None,
                       env_metrics: Optional[dict] = None) -> Any:
        """处理完整数据（轨迹+奖励+环境指标）并调用外部 API，同时记录数据"""
        try:
            # 格式化输入数据
            api_input = self._format_rollout_data(rollout_result, reward_result, env_metrics)
            
            # 记录发送到API的数据
            self._record_data(
                data_type="sent",
                data={
                    "rollout_id": id(rollout_result),
                    "api_input": api_input,
                    "api_url": self.api_url
                }
            )
            
            # 调用外部 API
            response = requests.post(
                self.api_url,
                headers=self.headers,
                data=json.dumps({"inputs": api_input}),
                timeout=self.timeout
            )
            response.raise_for_status()
            api_result = response.json()
            
            # 记录从API接收的数据
            self._record_data(
                data_type="received",
                data={
                    "rollout_id": id(rollout_result),
                    "api_response": api_result,
                    "status_code": response.status_code,
                    "response_time": response.elapsed.total_seconds()
                }
            )
            
            return api_result

        except requests.exceptions.Timeout:
            error_msg = "API request timed out"
            logging.error(error_msg)
            self._record_data(
                data_type="error",
                data={
                    "rollout_id": id(rollout_result),
                    "error_type": "timeout",
                    "message": error_msg
                }
            )
            return {"error": "timeout", "data": None}
        except Exception as e:
            error_msg = f"API request failed: {str(e)}"
            logging.error(error_msg)
            self._record_data(
                data_type="error",
                data={
                    "rollout_id": id(rollout_result),
                    "error_type": "exception",
                    "message": error_msg,
                    "exception_type": type(e).__name__
                }
            )
            return {"error": str(e), "data": None}

    def run(self):
        """主循环：整合处理数据并记录接收的原始数据"""
        assert self._input_channel is not None, "Input channel not set"
        assert self._output_channel is not None, "Output channel not set"
        assert self._reward_channel is not None, "Reward channel not set"
        assert self._env_channel is not None, "Environment channel not set"

        logging.info("Starting HttpAPIWorker main loop with data recording support")
        while True:
            # 1. 接收rollout生成的轨迹并记录
            rollout_result: RolloutResult = self._input_channel.get()
            rollout_id = id(rollout_result)
            self._record_data(
                data_type="rollout_received",
                data={
                    "rollout_id": rollout_id,
                    "num_sequence": rollout_result.num_sequence,
                    "prompts": rollout_result.prompt_texts,
                    "responses": rollout_result.response_texts,
                    "rewards": rollout_result.rewards.tolist() if rollout_result.rewards is not None else None
                }
            )
            
            # 2. 接收奖励计算结果并记录
            reward_result = self._reward_channel.get()
            self.reward_buffer[rollout_id] = reward_result
            self._record_data(
                data_type="reward_received",
                data={
                    "rollout_id": rollout_id,
                    "reward_result": reward_result
                }
            )
            
            # 假设接收的是环境交互后的infos字典（而非EnvResult）
            # 1. 接收环境返回的infos（包含指标数据）
            env_infos = self._env_channel.get()  # 替换EnvResult接收，实际类型为dict

            # 2. 提取指标数据（根据环境类的实现，指标通常在infos["episode"]中）
            # 不同环境的指标字段可能不同，需根据实际环境适配（参考各Env类的_record_metrics方法）
            env_metrics = env_infos.get("episode", {})  # 例如包含success_once、return、episode_len等

            # 3. 存储指标到缓冲区
            self.env_metrics_buffer[rollout_id] = env_metrics

            # 4. 记录数据（保持原记录逻辑，适配实际指标结构）
            self._record_data(
                data_type="env_metrics_received",
                data={
                    "rollout_id": rollout_id,
                    "env_metrics": env_metrics  # 直接使用提取的指标字典
                }
            )
            
            # 4. 处理完整数据
            if rollout_id in self.reward_buffer and rollout_id in self.env_metrics_buffer:
                api_result = self.process_rollout(
                    rollout_result=rollout_result,
                    reward_result=self.reward_buffer.pop(rollout_id),
                    env_metrics=self.env_metrics_buffer.pop(rollout_id)
                )
                
                # 记录处理结果
                self._record_data(
                    data_type="processed",
                    data={
                        "rollout_id": rollout_id,
                        "api_result_summary": "success" if "error" not in api_result else api_result["error"]
                    }
                )
                
                # 发送到输出通道
                self._output_channel.put({
                    "rollout_id": rollout_id,
                    "api_result": api_result,
                    "timestamp": self._get_timestamp()
                })