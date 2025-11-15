#!/bin/bash

# 配置环境变量（根据实际环境调整）
export MUJOCO_GL="egl"
export PYOPENGL_PLATFORM="egl"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TOKENIZERS_PARALLELISM=false
# 新增：设置 EMBODIED_PATH 环境变量（关键修复）
export EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"

# 获取项目根路径（基于 EMBODIED_PATH 计算，与原脚本风格一致）
export REPO_PATH=$(dirname $(dirname "$EMBODIED_PATH"))
export PYTHONPATH=${REPO_PATH}:$PYTHONPATH

# 环境特定路径配置（按需修改）
export OMNIGIBSON_DATA_PATH=${OMNIGIBSON_DATA_PATH:-"/path/to/omnigibson/data"}
export OMNIGIBSON_DATASET_PATH=${OMNIGIBSON_DATASET_PATH:-"${OMNIGIBSON_DATA_PATH}/behavior-1k-assets/"}
export OMNIGIBSON_KEY_PATH=${OMNIGIBSON_KEY_PATH:-"${OMNIGIBSON_DATA_PATH}/omnigibson.key"}
export ISAAC_PATH=${ISAAC_PATH:-"/path/to/isaac-sim"}
export EXP_PATH=${EXP_PATH:-"${ISAAC_PATH}/apps"}
export CARB_APP_PATH=${CARB_APP_PATH:-"${ISAAC_PATH}/kit"}

# 配置文件名称（默认使用openvlaoft配置）
if [ -z "$1" ]; then
    CONFIG_NAME="api"
else
    CONFIG_NAME=$1
fi

# 启动训练脚本
python ${REPO_PATH}/examples/embodiment/try.py \
    --config-path ${EMBODIED_PATH}/config/ \
    --config-name $CONFIG_NAME