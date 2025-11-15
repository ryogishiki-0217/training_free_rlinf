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

import gc

import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from tqdm import tqdm

from datetime import datetime  # 新增：导入datetime模块
from pathlib import Path  # 新增：用于路径处理
import json  # 新增：导入json模块

from rlinf.data.io_struct import EmbodiedRolloutResult
from rlinf.models import get_model, get_vla_model_config_and_processor
from rlinf.scheduler import Cluster, Worker
from rlinf.utils.metric_utils import compute_split_num
from rlinf.utils.placement import HybridComponentPlacement


class MultiStepRolloutWorker(Worker):
    def __init__(self, cfg: DictConfig):
        Worker.__init__(self)

        self.cfg = cfg
        self._env_group_name = cfg.env.group_name
        self._actor_group_name = cfg.actor.group_name
        self.device = torch.cuda.current_device()

        self._obs_queue_name = cfg.env.channel.queue_name
        self._action_queue_name = cfg.rollout.channel.queue_name
        self._replay_buffer_name = cfg.actor.channel.queue_name
        self.stage_num = cfg.rollout.pipeline_stage_num

        self._component_placement = HybridComponentPlacement(cfg, Cluster())
        self.channel = self.connect_channel(cfg.rollout.channel.name)

        # 新增：初始化记录相关参数
        self.record_enabled = cfg.rollout.get("record", False)  # 从配置读取开关
        self.record_path = self._init_record_path()  # 初始化记录路径
        self.rollout_records = []  # 存储记录的列表

    def _init_record_path(self):
        """初始化记录文件保存路径"""
        if not self.record_enabled:
            return None
        # 从配置获取路径，默认当前目录下"rollout_records"文件夹
        base_path = Path(self.cfg.rollout.get("record_path", "./rollout_records"))
        # 创建带时间戳的子目录，避免覆盖
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        record_dir = base_path / f"rank_{self._rank}_{timestamp}"
        record_dir.mkdir(parents=True, exist_ok=True)
        return record_dir

    def init_worker(self):
        # NOTE:
        # because pi series have some different dtype params, we can not call `to`
        # after get_model, here we simply change actor.model.precision to rollout.precision
        # and after get_model we change it back. THIS CODE SHOULD BE REFACTORED SOON.
        with open_dict(self.cfg):
            original_precision = self.cfg.actor.model.precision
            self.cfg.actor.model.precision = self.cfg.rollout.precision
        self.hf_model = get_model(self.cfg.rollout.model_dir, self.cfg.actor.model)
        with open_dict(self.cfg):
            self.cfg.actor.model.precision = original_precision

        if self.cfg.actor.model.model_name in ["openvla", "openvla_oft"]:
            model_config, input_processor = get_vla_model_config_and_processor(
                self.cfg.actor
            )
            self.hf_model.setup_config_and_processor(
                model_config, self.cfg, input_processor
            )

        self.hf_model.eval()

        self.setup_sample_params()
        if self.cfg.rollout.get("enable_offload", False):
            self.offload_model()
            
        # 新增：如果启用记录，打印提示
        if self.record_enabled:
            self.log_info(f"Rollout记录已启用，保存路径：{self.record_path}")
            
    def setup_sample_params(self):
        # length parameters for rollout
        self._length_params = OmegaConf.to_container(
            self.cfg.algorithm.length_params, resolve=True
        )
        # sampling parameters for rollout
        self._sampling_params = OmegaConf.to_container(
            self.cfg.algorithm.sampling_params, resolve=True
        )
        self._train_sampling_params = {
            "temperature": self._sampling_params["temperature_train"],
            "top_k": self._sampling_params["top_k"],
            "top_p": self._sampling_params["top_p"],
            "max_new_tokens": self._length_params["max_new_token"],
            "use_cache": True,
        }

        self._eval_sampling_params = {
            "temperature": self._sampling_params["temperature_eval"],
            "top_k": self._sampling_params["top_k"],
            "top_p": self._sampling_params["top_p"],
            "max_new_tokens": self._length_params["max_new_token"],
        }

    def predict(self, env_obs, do_sample=True, mode="train"):
        kwargs = (
            self._train_sampling_params
            if mode == "train"
            else self._eval_sampling_params
        )
        kwargs["do_sample"] = do_sample

        if self.cfg.actor.model.model_name in ["openpi"]:
            kwargs = {"mode": mode}

        with torch.no_grad():
            actions, result = self.hf_model.predict_action_batch(
                env_obs=env_obs,
                **kwargs,
            )

        return actions, result

    def update_env_output(self, i, env_output):
        # first step for env_batch
        if env_output["rewards"] is None:
            self.buffer_list[i].dones.append(env_output["dones"].contiguous().cpu())
            return

        self.buffer_list[i].rewards.append(env_output["rewards"].cpu().contiguous())
        self.buffer_list[i].dones.append(env_output["dones"].bool().cpu().contiguous())

        # Note: currently this is not correct for chunk-size>1 with partial reset
        if env_output["dones"].any() and self.cfg.env.train.auto_reset:
            if hasattr(self.hf_model, "value_head"):
                dones = env_output["dones"]

                final_obs = env_output["final_obs"]
                with torch.no_grad():
                    actions, result = self.predict(final_obs)
                    if "prev_values" in result:
                        _final_values = result["prev_values"]
                    else:
                        _final_values = torch.zeros_like(actions[:, 0])
                final_values = torch.zeros_like(_final_values[:, 0])  # [bsz, ]
                last_step_dones = dones[:, -1]  # [bsz, ]

                final_values[last_step_dones] = _final_values[:, 0][last_step_dones]

                self.buffer_list[i].rewards[-1][:, -1] += (
                    self.cfg.algorithm.gamma * final_values.cpu()
                )

    def generate(self):
        # 新增：每次生成前清空记录列表
        if self.record_enabled:
            self.rollout_records.clear()
            
        if self.cfg.rollout.get("enable_offload", False):
            self.reload_model()
        self.buffer_list = [EmbodiedRolloutResult() for _ in range(self.stage_num)]

        for epoch in tqdm(
            range(self.cfg.algorithm.rollout_epoch),
            desc="Generating Rollout Epochs",
            disable=(self._rank != 0),
        ):
            for chunk in range(self.cfg.algorithm.n_chunk_steps):
                for i in range(self.stage_num):
                    env_output = self.recv_env_output()
                    self.update_env_output(i, env_output)
                    actions, result = self.predict(env_output["obs"])

                    # 新增：记录输入输出（根据任务类型调整字段）
                    if self.record_enabled:
                        self._record_step(
                            epoch=epoch,
                            chunk=chunk,
                            stage=i,
                            env_obs=env_output["obs"],  # 问题/观测输入
                            actions=actions,  # 模型输出
                            result=result  # 额外结果（如logits等）
                        )

                    self.buffer_list[i].append_result(result)

                    self.send_chunk_actions(actions)

            for i in range(self.stage_num):
                env_output = self.recv_env_output()
                self.update_env_output(i, env_output)
                actions, result = self.predict(env_output["obs"])
                if "prev_values" in result:
                    self.buffer_list[i].prev_values.append(
                        result["prev_values"].cpu().contiguous()
                    )

        for i in range(self.stage_num):
            self.send_rollout_batch(i)

        if self.cfg.rollout.get("enable_offload", False):
            self.offload_model()
            
        # 新增：生成结束后保存记录
        if self.record_enabled:
            self._save_records()

    def _record_step(self, epoch, chunk, stage, env_obs, actions, result):
        """记录单步的输入输出，确保所有数据可JSON序列化"""
        # 处理观测数据中的 Tensor，转换为列表
        processed_obs = {}
        if env_obs is not None:
            for key, value in env_obs.items():
                if isinstance(value, torch.Tensor):
                    # Tensor 转换为列表（若维度较大，可只记录形状）
                    processed_obs[key] = value.tolist()
                    # 可选：仅记录形状以节省空间（适用于图像等大张量）
                    # processed_obs[f"{key}_shape"] = list(value.shape)
                elif isinstance(value, (list, dict, str, int, float, bool)):
                    # 其他可序列化类型直接保留
                    processed_obs[key] = value
                else:
                    # 未知类型转换为字符串描述
                    processed_obs[key] = str(value)
        
        record = {
            "epoch": epoch,
            "chunk": chunk,
            "stage": stage,
            "timestamp": datetime.now().isoformat(),
            "input": {
                "obs": processed_obs  # 使用处理后的观测数据
            },
            "output": {
                "actions": actions.tolist(),
                "action_shape": list(actions.shape),  # 元组转换为列表
                "values": result["prev_values"].tolist() if "prev_values" in result else None
            }
        }
        self.rollout_records.append(record)

    # def _record_step(self, epoch, chunk, stage, env_obs, actions, result):
    #     """记录单步的输入输出"""
    #     # 根据实际任务类型解析obs（这里以具身任务为例，可根据需求修改）
    #     record = {
    #         "epoch": epoch,
    #         "chunk": chunk,
    #         "stage": stage,
    #         "timestamp": datetime.now().isoformat(),
    #         "input": {
    #             # 示例：解析观测数据（根据env_obs的实际结构调整）
    #             "obs": env_obs if env_obs else None,
    #         },
    #         "output": {
    #             "actions": actions.tolist(),  # 模型生成的动作/输出
    #             "action_shape": actions.shape,
    #             # 可选：记录额外信息（如价值估计、logprobs等）
    #             "values": result["prev_values"].tolist() if "prev_values" in result else None
    #         }
    #     }
    #     self.rollout_records.append(record)

    def _save_records(self):
        """将记录保存为JSON文件"""
        if not self.rollout_records:
            self.log_warn("没有记录可保存")
            return

        # 保存为JSON Lines格式（每行一个JSON对象，便于处理大文件）
        record_file = self.record_path / "rollout_records.jsonl"
        with open(record_file, "w", encoding="utf-8") as f:
            for record in self.rollout_records:
                json.dump(record, f, ensure_ascii=False)
                f.write("\n")

        self.log_info(f"已保存{len(self.rollout_records)}条记录到：{record_file}")

    def evaluate(self):
        if self.cfg.rollout.get("enable_offload", False):
            self.reload_model()

        for _ in range(self.cfg.algorithm.n_eval_chunk_steps):
            for _ in range(self.stage_num):
                env_output = self.recv_env_output()
                actions, _ = self.predict(env_output["obs"], mode="eval")
                self.send_chunk_actions(actions)

        if self.cfg.rollout.get("enable_offload", False):
            self.offload_model()

    def offload_model(self):
        self.hf_model = self.hf_model.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()

    def reload_model(self):
        self.hf_model = self.hf_model.to(self.device)

    def sync_model_from_actor(self):
        # print("即将进入断点所在行")
        # print("-----------------------------------------------------------")
        # import pdb; pdb.set_trace() 
        param_state_dict = self.recv(self._actor_group_name, src_rank=self._rank)

        self.hf_model.load_state_dict(param_state_dict)
        del param_state_dict
        gc.collect()
        torch.cuda.empty_cache()

    def recv_env_output(self):
        env_output = self.channel.get(
            key=f"{self._obs_queue_name}_{self._rank}",
        )
        return env_output

    def send_chunk_actions(self, chunk_actions):
        self.channel.put(
            item=chunk_actions,
            key=f"{self._action_queue_name}_{self._rank}",
        )

    def send_rollout_batch(self, stage_id):
        # send rollout_batch to actor
        send_num = self._component_placement.get_world_size("rollout") * self.stage_num
        recv_num = self._component_placement.get_world_size("actor")
        split_num = compute_split_num(recv_num, send_num)
        splited_rollout_result = self.buffer_list[stage_id].to_splited_dict(split_num)
        for i in range(split_num):
            self.channel.put(
                item=splited_rollout_result[i],
                key=self._replay_buffer_name,
            )

    def set_global_step(self, global_step):
        if hasattr(self.hf_model, "set_global_step"):
            self.hf_model.set_global_step(global_step)
