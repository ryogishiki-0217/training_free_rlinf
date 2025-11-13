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

from omegaconf.dictconfig import DictConfig
from tqdm import tqdm

from rlinf.utils.distributed import ScopedTimer
from rlinf.utils.metric_logger import MetricLogger
from rlinf.utils.metric_utils import compute_evaluate_metrics
from rlinf.utils.runner_utils import check_progress
from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor
from rlinf.workers.env.env_worker import EnvWorker
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


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

        # 冻结所有模型参数
        self.freeze_model_parameters()

    def init_workers(self):
        # 初始化worker，按顺序创建以减少内存占用峰值
        self.actor.init_worker().wait()
        self.rollout.init_worker().wait()
        self.env.init_worker().wait()

        # 恢复检查点（如果需要）
        if self.cfg.runner.get("resume_dir", None) is not None:
            self.global_step = int(self.cfg.runner.resume_dir.split("global_step_")[-1])

    def freeze_model_parameters(self):
        """冻结所有模型参数，禁止梯度更新"""
        # 冻结actor模型参数
        for param in self.actor.model.parameters():
            param.requires_grad = False
        
        # 如果有critic模型也冻结其参数
        if self.critic is not None:
            for param in self.critic.model.parameters():
                param.requires_grad = False

    def generate_rollouts(self):
        """生成rollout数据，不涉及权重更新"""
        env_futures = self.env.interact()
        rollout_futures = self.rollout.generate()
        actor_futures = self.actor.recv_rollout_batch()
        
        # 等待所有异步操作完成
        env_results = env_futures.wait()
        actor_futures.wait()
        rollout_futures.wait()

        # 计算环境相关指标
        env_results_list = [results for results in env_results if results is not None]
        env_metrics = compute_evaluate_metrics(env_results_list)
        return env_metrics

    def evaluate(self):
        """评估当前模型性能"""
        env_futures = self.env.evaluate()
        rollout_futures = self.rollout.evaluate()
        env_results = env_futures.wait()
        rollout_futures.wait()
        
        eval_metrics_list = [results for results in env_results if results is not None]
        eval_metrics = compute_evaluate_metrics(eval_metrics_list)
        return eval_metrics

    def run(self):
        """主运行循环：仅生成rollout和计算优势值，不更新模型权重"""
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

            # 定期评估（如果配置）
            if (
                _step % self.cfg.runner.val_check_interval == 0
                and self.cfg.runner.val_check_interval > 0
            ):
                with self.timer("eval"):
                    eval_metrics = self.evaluate()
                    eval_metrics = {f"eval/{k}": v for k, v in eval_metrics.items()}
                    self.metric_logger.log(data=eval_metrics, step=_step)

            with self.timer("step"):
                # 生成rollout数据
                with self.timer("generate_rollouts"):
                    env_metrics = self.generate_rollouts()

                # 计算优势值和回报（不更新模型）
                with self.timer("cal_adv_and_returns"):
                    actor_futures = self.actor.compute_advantages_and_returns()
                    actor_rollout_metrics = actor_futures.wait()

                self.global_step += 1

                # 检查是否需要保存（仅保存rollout数据相关检查点）
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

            # 收集并记录所有指标
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

        self.metric_logger.finish()

    def _save_checkpoint(self):
        """保存检查点（仅保存必要的rollout数据和状态，不包含模型权重更新）"""
        base_output_dir = os.path.join(
            self.cfg.runner.logger.log_path,
            self.cfg.runner.logger.experiment_name,
            f"checkpoints/global_step_{self.global_step}",
        )
        # 保存rollout相关状态（不包含模型权重更新）
        rollout_save_path = os.path.join(base_output_dir, "rollout")
        os.makedirs(rollout_save_path, exist_ok=True)
        self.rollout.save_checkpoint(rollout_save_path, self.global_step).wait()

        # 保存环境状态
        env_save_path = os.path.join(base_output_dir, "env")
        os.makedirs(env_save_path, exist_ok=True)
        self.env.save_checkpoint(env_save_path, self.global_step).wait()

    def set_max_steps(self):
        """设置最大运行步数"""
        self.num_steps_per_epoch = 1
        self.max_steps = self.num_steps_per_epoch * self.cfg.runner.max_epochs

        if (max_steps := self.cfg.runner.get("max_steps", -1)) >= 0:
            self.max_steps = min(self.max_steps, max_steps)

    @property
    def epoch(self):
        return self.global_step // self.num_steps_per_epoch