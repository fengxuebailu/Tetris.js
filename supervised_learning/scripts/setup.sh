#!/bin/bash

# 设置虚拟环境并安装依赖

# 检查是否已存在虚拟环境
if [ -d ".venv" ]; then
    echo "发现已存在的虚拟环境，是否重新创建？(y/n)"
    read answer
    if [ "$answer" = "y" ]; then
        rm -rf .venv
    else
        echo "将使用现有虚拟环境"
        source .venv/bin/activate
        python -m pip install -r requirements.txt
        exit
    fi
fi

# 创建新的虚拟环境
echo "创建新的虚拟环境..."
python -m venv .venv

# 激活虚拟环境
source .venv/bin/activate

# 更新pip
python -m pip install --upgrade pip

# 安装依赖
echo "安装依赖包..."
python -m pip install -r requirements.txt

echo "环境设置完成！"
echo "使用 'python start.py' 开始训练或评估模型"
