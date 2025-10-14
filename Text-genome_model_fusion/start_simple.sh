#!/bin/bash

# 简单API服务启动脚本

echo "🚀 启动DNA序列问答API服务..."

# 设置环境变量
export PYTHONPATH=$(pwd)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# 检查checkpoint
if [ ! -d "checkpoints" ]; then
    echo "❌ checkpoints目录不存在"
    exit 1
fi

echo "🌐 启动服务..."
echo "   服务地址: http://localhost:8000"
echo "   API文档: http://localhost:8000/docs"
echo "   健康检查: http://localhost:8000/health"
echo ""

# 启动服务
python simple_app.py
