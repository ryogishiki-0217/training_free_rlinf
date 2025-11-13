# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC, abstractmethod
from typing import Any, Optional
from omegaconf import DictConfig
from rlinf.scheduler import Worker
from rlinf.data.io_struct import RolloutResult


class BaseAPIWorker(Worker, ABC):
    """基础 API 工作器抽象类，定义核心接口"""

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self._input_channel = None  # 接收 Rollout 输出的通道
        self._output_channel = None  # 发送 API 结果的通道

    @abstractmethod
    def init_worker(self):
        """初始化工作器（连接通道、配置客户端等）"""
        pass

    @abstractmethod
    def process_rollout(self, rollout_result: RolloutResult) -> Any:
        """
        处理 Rollout 输出并与外部 API 交互
        :param rollout_result: Rollout 生成的轨迹数据
        :return: 外部 API 返回的结果
        """
        pass

    @abstractmethod
    def run(self):
        """主循环：从通道接收数据 -> 处理 -> 发送结果"""
        pass

    def set_channels(self, input_channel, output_channel):
        """设置输入输出通道"""
        self._input_channel = input_channel
        self._output_channel = output_channel