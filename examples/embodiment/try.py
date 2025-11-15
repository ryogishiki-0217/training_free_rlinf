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

import json
import torch.multiprocessing as mp
import hydra
from omegaconf.omegaconf import OmegaConf

from rlinf.config import validate_cfg
from rlinf.runners.try_runner import FrozenModelRunner  # 对应FrozenModelRunner
from rlinf.scheduler import Cluster
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor
from rlinf.workers.env.env_worker import EnvWorker
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker

# 设置多进程启动方式
mp.set_start_method("spawn", force=True)


@hydra.main(
    version_base="1.1", 
    config_path="config",  # 复用现有配置路径
    config_name="api"  # 可根据需要修改配置文件名
)
def main(cfg) -> None:
    # 验证配置
    cfg = validate_cfg(cfg)
    # 打印配置（便于调试）
    print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))

    # 初始化集群和组件部署策略
    cluster = Cluster(num_nodes=cfg.cluster.num_nodes)
    component_placement = HybridComponentPlacement(cfg, cluster)

    # 创建Actor Worker组
    actor_placement = component_placement.get_strategy("actor")
    actor_group = EmbodiedFSDPActor.create_group(cfg).launch(
        cluster, name=cfg.actor.group_name, placement_strategy=actor_placement
    )

    # 创建Rollout Worker组
    rollout_placement = component_placement.get_strategy("rollout")
    rollout_group = MultiStepRolloutWorker.create_group(cfg).launch(
        cluster, name=cfg.rollout.group_name, placement_strategy=rollout_placement
    )

    # 创建Env Worker组
    env_placement = component_placement.get_strategy("env")
    env_group = EnvWorker.create_group(cfg).launch(
        cluster, name=cfg.env.group_name, placement_strategy=env_placement
    )

    # 初始化FrozenModelRunner
    runner = FrozenModelRunner(
        cfg=cfg,
        actor=actor_group,
        rollout=rollout_group,
        env=env_group,
    )

    # 初始化Workers并启动运行
    runner.init_workers()
    runner.run()


if __name__ == "__main__":
    main()