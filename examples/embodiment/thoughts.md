generate_rollouts()
├─ 1. 环境交互
│  └─ env.interact()  # 生成观测，等待动作
│     └─ 内部：初始化环境/执行动作 → 返回观测、奖励、终止信号
├─ 2. 轨迹生成
│  └─ rollout.generate()  # 生成动作，记录轨迹
│     ├─ 从env获取观测 → 调用actor.sample_actions()生成动作
│     ├─ 将动作发送给env执行
│     └─ 记录轨迹数据（obs, actions, rewards, dones）
├─ 3. 数据接收
│  └─ actor.recv_rollout_batch()  # 接收rollout发送的轨迹
├─ 4. 同步等待
│  ├─ env_futures.wait()
│  ├─ actor_futures.wait()
│  └─ rollout_futures.wait()
└─ 5. 指标计算
   └─ compute_evaluate_metrics()  # 统计奖励、成功率等