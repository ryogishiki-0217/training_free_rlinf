# 修正导入语句，添加 abstractmethod
from abc import ABC, abstractmethod  # 关键：导入 abstractmethod
from typing import Any, Optional
from omegaconf import DictConfig
from rlinf.scheduler import Worker
from rlinf.data.io_struct import RolloutResult

# 确保之前添加的元类定义正确（解决元类冲突）
from rlinf.scheduler.worker.worker import WorkerMeta  # 导入Worker的元类
class APIMeta(WorkerMeta, type(ABC)):  # 合并元类（ABC的元类是ABCMeta）
    pass

class BaseAPIWorker(Worker, ABC, metaclass=APIMeta):
    """基础 API 工作器抽象类，定义核心接口"""

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self._input_channel = None  # 接收 Rollout 输出的通道
        self._output_channel = None  # 发送 API 结果的通道

    @abstractmethod  # 现在可以正常使用该装饰器
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