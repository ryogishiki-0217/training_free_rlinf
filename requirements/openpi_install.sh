#!/bin/bash

# 定义环境路径和目标
VENV_PATH="$HOME/RLinf/venv_openpi"
TARGET="openpi"
PYTHON_VERSION="3.11.10"
EMBODIED_TARGET=("openvla" "openvla-oft" "openpi")

# 检查是否已存在环境
if [ -d "$VENV_PATH" ]; then
    echo "环境 $VENV_PATH 已存在，是否覆盖？[y/N]"
    read -r response
    if [[ "$response" != "y" && "$response" != "Y" ]]; then
        echo "安装终止"
        exit 0
    fi
    rm -rf "$VENV_PATH"
fi

# 创建并激活虚拟环境
echo "创建 $VENV_PATH 虚拟环境..."
uv venv --python="$PYTHON_VERSION" "$VENV_PATH"
source "$VENV_PATH/bin/activate" || {
    echo "虚拟环境激活失败，请检查路径是否正确：$VENV_PATH"
    exit 1
}

# 安装基础依赖（提前安装 setuptools 解决 tree 包依赖）
echo "安装基础依赖和 setuptools..."
UV_TORCH_BACKEND=auto uv sync
uv pip install setuptools  # 新增：安装 setuptools

# 安装具身智能相关依赖
if [[ " ${EMBODIED_TARGET[*]} " == *" $TARGET "* ]]; then
    echo "安装具身智能依赖..."
    uv sync --extra embodied
    uv pip uninstall pynvml  # 修正：移除 -y 参数（uv 不支持）
    bash requirements/install_embodied_deps.sh
    
    # 安装LIBERO并配置环境变量
    # mkdir -p /opt && git clone https://github.com/RLinf/LIBERO.git /opt/libero
    echo "export PYTHONPATH=/opt/libero:\$PYTHONPATH" >> "$VENV_PATH/bin/activate"
fi

# 安装openpi特有依赖（保留 --no-build-isolation 确保 setuptools 生效）
echo "安装openpi专用依赖..."
UV_TORCH_BACKEND=auto GIT_LFS_SKIP_SMUDGE=1 uv pip install -r requirements/openpi.txt --no-build-isolation

# 复制transformers补丁
TRANSFORMERS_REPLACE_DIR="$VENV_PATH/lib/python3.11/site-packages/openpi/models_pytorch/transformers_replace/"
if [ -d "$TRANSFORMERS_REPLACE_DIR" ]; then
    cp -r "$TRANSFORMERS_REPLACE_DIR"* "$VENV_PATH/lib/python3.11/site-packages/transformers/"
else
    echo "警告：未找到transformers补丁目录，可能影响功能"
fi

# 下载tokenizer模型
echo "下载paligemma_tokenizer.model到~/openpi_weights/..."
TOKENIZER_DIR=~/openpi_weights/
mkdir -p "$TOKENIZER_DIR"
if command -v gsutil &> /dev/null; then
    gsutil -m cp -r gs://big_vision/paligemma_tokenizer.model "$TOKENIZER_DIR"
else
    echo "警告：gsutil未安装，无法自动下载tokenizer，请手动下载并放置到 $TOKENIZER_DIR"
    echo "手动下载地址：https://storage.googleapis.com/big_vision/paligemma_tokenizer.model"
fi

# 完成提示
deactivate
echo "----------------------------------------"
echo "openpi环境安装完成：$VENV_PATH"
echo "激活命令：source $VENV_PATH/bin/activate"
echo "退出命令：deactivate"