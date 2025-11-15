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

import os
import logging  # 新增：引入日志模块

from omegaconf.dictconfig import DictConfig
from tqdm import tqdm

from rlinf.utils.distributed import ScopedTimer
from rlinf.utils.metric_logger import MetricLogger
from rlinf.utils.metric_utils import compute_evaluate_metrics
from rlinf.utils.runner_utils import check_progress
from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor
from rlinf.workers.env.env_worker import EnvWorker
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker

import json
from pathlib import Path
from datetime import datetime

class FrozenModelRunner:
    def __init__(
        self,
        cfg: DictConfig,
        actor: EmbodiedFSDPActor,
        rollout: MultiStepRolloutWorker,
        env: EnvWorker,
        critic=None,
        reward=None,
        run_timer=None,
    ):
        self.cfg = cfg
        self.actor = actor
        self.rollout = rollout
        self.env = env
        self.critic = critic
        self.reward = reward

        # 计时器用于检查是否应停止运行
        self.run_timer = run_timer

        self.consumed_samples = 0
        self.global_step = 0

        # 配置最大步数
        self.set_max_steps()

        self.timer = ScopedTimer(reduction="max", sync_cuda=False)
        self.metric_logger = MetricLogger(cfg)

        # 新增：标记是否需要冻结模型，延迟到init_workers后执行
        self.need_freeze = True

        # 移除：__init__中不再直接调用冻结逻辑
        # self.freeze_model_parameters()
        
        self.record_cfg = cfg.runner.get("record", {})
        self.record_enabled = self.record_cfg.get("enable", False)
        self.record_buffer = []  # 缓冲区减少IO操作
        if self.record_enabled:
            self._init_record_dir()
            # 记录需要包含的字段（从配置读取，默认全记录）
            self.include_fields = self.record_cfg.get("include", {
                "env_input": True,
                "rollout_trajectories": True,
                "actor_advantages": True
            })


    def init_workers(self):
        # 初始化worker，按顺序创建以减少内存占用峰值
        self.actor.init_worker().wait()
        self.rollout.init_worker().wait()
        self.env.init_worker().wait()

        # 恢复检查点（如果需要）
        if self.cfg.runner.get("resume_dir", None) is not None:
            self.global_step = int(self.cfg.runner.resume_dir.split("global_step_")[-1])
            logging.info(f"从检查点恢复，全局步数设置为: {self.global_step}")

        # 新增：在所有worker初始化完成后执行冻结
        if self.need_freeze:
            self.freeze_model_parameters()
            self.need_freeze = False  # 确保只冻结一次

    def freeze_model_parameters(self):
        """冻结所有模型参数，禁止梯度更新（通过Worker接口分发指令）"""
        try:
            # 关键修复：通过WorkerGroup的apply_all分发冻结指令到子Worker
            logging.info("开始冻结Actor模型参数...")
            actor_futures = self.actor.freeze_parameters()
            actor_futures.wait()  # 等待所有Actor Worker完成冻结
            logging.info("Actor模型参数冻结完成")

            # 如果有critic模型也冻结其参数
            if self.critic is not None:
                logging.info("开始冻结Critic模型参数...")
                critic_futures = self.critic.freeze_parameters()
                critic_futures.wait()
                logging.info("Critic模型参数冻结完成")
        except Exception as e:
            logging.error(f"参数冻结失败: {str(e)}", exc_info=True)
            raise  # 重新抛出异常，确保错误被捕获

    def evaluate(self):
        """评估当前模型性能"""
        env_futures = self.env.evaluate()
        rollout_futures = self.rollout.evaluate()
        env_results = env_futures.wait()
        rollout_futures.wait()
        
        eval_metrics_list = [results for results in env_results if results is not None]
        eval_metrics = compute_evaluate_metrics(eval_metrics_list)
        return eval_metrics

    def _init_record_dir(self):
        """初始化记录文件保存目录"""
        base_path = Path(self.record_cfg.get("path", "./grpo_records"))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.record_dir = base_path / f"frozen_run_{timestamp}"
        self.record_dir.mkdir(parents=True, exist_ok=True)
        print(f"JSONL记录已启用，保存路径：{self.record_dir}")

    def _write_records(self, step: int):
        """将缓冲区的记录写入JSONL文件"""
        if not self.record_buffer:
            return
        # 按step分文件存储（如 step_0-9.jsonl）
        start_step = self.record_buffer[0]["global_step"]
        end_step = self.record_buffer[-1]["global_step"]
        record_file = self.record_dir / f"records_step_{start_step}-{end_step}.jsonl"
        
        with open(record_file, "a", encoding="utf-8") as f:
            for item in self.record_buffer:
                json.dump(item, f, ensure_ascii=False)
                f.write("\n")  # 每条记录占一行
        self.record_buffer.clear()

    def generate_rollouts(self):
        """生成rollout数据，同时收集记录所需的原始数据"""
        env_futures = self.env.interact()
        rollout_futures = self.rollout.generate()
        actor_futures = self.actor.recv_rollout_batch()
        
        # 等待所有异步操作完成（获取原始数据而非仅指标）
        env_results = env_futures.wait()  # 环境输入/输出原始数据
        actor_results = actor_futures.wait()  # 包含轨迹数据
        rollout_results = rollout_futures.wait()  # rollout生成的原始结果

        # 计算环境指标（原有逻辑）
        env_results_list = [results for results in env_results if results is not None]
        env_metrics = compute_evaluate_metrics(env_results_list)

        # 新增：收集记录数据（仅在启用时执行）
        self.current_rollout_data = {
            "env_input": env_results,          # 环境输入（观测、任务描述等）
            "rollout_trajectories": rollout_results,  # rollout生成的轨迹
            "actor_batch": actor_results       # actor接收的batch数据
        }

        return env_metrics

    def run(self):
        """主运行循环：生成rollout、计算优势值，并新增JSONL记录"""
        start_step = self.global_step
        global_pbar = tqdm(
            initial=start_step,
            total=self.max_steps,
            desc="Global Step",
            ncols=800,
        )

        for _step in range(start_step, self.max_steps):
            # 设置全局步数
            self.actor.set_global_step(self.global_step)
            self.rollout.set_global_step(self.global_step)
            eval_metrics = {}

            # 定期评估（原有逻辑）
            if (
                _step % self.cfg.runner.val_check_interval == 0
                and self.cfg.runner.val_check_interval > 0
            ):
                with self.timer("eval"):
                    eval_metrics = self.evaluate()
                    eval_metrics = {f"eval/{k}": v for k, v in eval_metrics.items()}
                    self.metric_logger.log(data=eval_metrics, step=_step)

            with self.timer("step"):
                # 生成rollout数据（会收集原始数据到current_rollout_data）
                with self.timer("generate_rollouts"):
                    env_metrics = self.generate_rollouts()

                # 计算优势值和回报（获取原始计算结果）
                with self.timer("cal_adv_and_returns"):
                    actor_futures = self.actor.compute_advantages_and_returns()
                    actor_rollout_metrics = actor_futures.wait()
                    # 假设返回结果中包含原始优势值数据（需根据实际actor实现调整）
                    self.current_adv_data = actor_futures.get_advantages_data()

                self.global_step += 1

                # 检查保存和结束条件（原有逻辑）
                run_val, save_model, is_train_end = check_progress(
                    self.global_step,
                    self.max_steps,
                    self.cfg.runner.val_check_interval,
                    self.cfg.runner.save_interval,
                    1.0,
                    run_time_exceeded=False,
                )

                if save_model:
                    self._save_checkpoint()

                # 新增：构建记录并加入缓冲区
                if self.record_enabled:
                    record = {
                        "global_step": _step,
                        "timestamp": datetime.now().isoformat(),
                        "env_input": self._format_env_input() if self.include_fields["env_input"] else None,
                        "rollout_trajectories": self._format_rollout_trajectories() if self.include_fields["rollout_trajectories"] else None,
                        "actor_advantages": self._format_advantages() if self.include_fields["actor_advantages"] else None
                    }
                    self.record_buffer.append(record)

            # 新增：每N步写入一次记录（避免频繁IO）
            if self.record_enabled and (
                _step % self.record_cfg.get("write_interval", 10) == 0 
                or _step == self.max_steps - 1
            ):
                self._write_records(_step)

            # 收集并记录所有指标（原有逻辑）
            time_metrics = self.timer.consume_durations()
            time_metrics = {f"time/{k}": v for k, v in time_metrics.items()}
            rollout_metrics = {
                f"rollout/{k}": v for k, v in actor_rollout_metrics[0].items()
            }
            env_metrics = {f"env/{k}": v for k, v in env_metrics.items()}

            self.metric_logger.log(env_metrics, _step)
            self.metric_logger.log(rollout_metrics, _step)
            self.metric_logger.log(time_metrics, _step)

            # 更新进度条
            logging_metrics = {**time_metrics,** eval_metrics, **env_metrics,** rollout_metrics}
            global_pbar.set_postfix(logging_metrics)
            global_pbar.update(1)

            if is_train_end:
                break

        # 最后一步确保所有记录都被写入
        if self.record_enabled and self.record_buffer:
            self._write_records(self.max_steps - 1)

        self.metric_logger.finish()

    # 新增：格式化记录字段（根据实际数据结构调整）
    def _format_env_input(self):
        """格式化环境输入数据（观测、任务描述等）"""
        formatted = []
        for env_ep in self.current_rollout_data["env_input"]:
            if env_ep is None:
                continue
            formatted.append({
                "obs_shape": env_ep["obs"].shape if "obs" in env_ep else None,
                "dones": env_ep["dones"].tolist() if "dones" in env_ep else None,
                "rewards": env_ep["rewards"].tolist() if "rewards" in env_ep else None,
                # 补充其他需要的环境输入字段
            })
        return formatted

    def _format_rollout_trajectories(self):
        """格式化rollout生成的轨迹数据"""
        formatted = []
        for traj in self.current_rollout_data["rollout_trajectories"]:
            formatted.append({
                "actions": traj["actions"].tolist() if "actions" in traj else None,
                "log_probs": traj["log_probs"].tolist() if "log_probs" in traj else None,
                "length": len(traj["actions"]) if "actions" in traj else 0,
                # 补充其他轨迹相关字段
            })
        return formatted

    def _format_advantages(self):
        """格式化优势值计算结果"""
        adv_data = self.current_adv_data
        return {
            "advantages": adv_data["advantages"].tolist() if "advantages" in adv_data else None,
            "returns": adv_data["returns"].tolist() if "returns" in adv_data else None,
            "values": adv_data["values"].tolist() if "values" in adv_data else None,
            # 补充其他优势值相关字段
        }

    def _save_checkpoint(self):
        """保存检查点（仅保存必要的rollout数据和状态，不包含模型权重更新）"""
        base_output_dir = os.path.join(
            self.cfg.runner.logger.log_path,
            self.cfg.runner.logger.experiment_name,
            f"checkpoints/global_step_{self.global_step}",
        )
        # 保存rollout相关状态
        rollout_save_path = os.path.join(base_output_dir, "rollout")
        os.makedirs(rollout_save_path, exist_ok=True)
        self.rollout.save_checkpoint(rollout_save_path, self.global_step).wait()

        # 保存环境状态
        env_save_path = os.path.join(base_output_dir, "env")
        os.makedirs(env_save_path, exist_ok=True)
        self.env.save_checkpoint(env_save_path, self.global_step).wait()

        logging.info(f"检查点已保存至: {base_output_dir}")

    def set_max_steps(self):
        """设置最大运行步数"""
        self.num_steps_per_epoch = 1
        self.max_steps = self.num_steps_per_epoch * self.cfg.runner.max_epochs

        if (max_steps := self.cfg.runner.get("max_steps", -1)) >= 0:
            self.max_steps = min(self.max_steps, max_steps)
        logging.info(f"最大训练步数设置为: {self.max_steps}")

    @property
    def epoch(self):
        return self.global_step // self.num_steps_per_epoch