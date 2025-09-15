# 详细部署步骤和配置

## GUI-Owl模型部署

### 步骤1: 下载模型文件

#### 方法1: 使用Hugging Face Hub
```python
from huggingface_hub import snapshot_download
import os

# 创建模型目录
os.makedirs("./models", exist_ok=True)

# 下载GUI-Owl-7B (约14GB)
print("正在下载GUI-Owl-7B模型...")
snapshot_download(
    repo_id="mPLUG/GUI-Owl-7B",
    local_dir="./models/GUI-Owl-7B",
    local_dir_use_symlinks=False
)

# 下载GUI-Owl-32B (约64GB，可选)
print("正在下载GUI-Owl-32B模型...")
snapshot_download(
    repo_id="mPLUG/GUI-Owl-32B", 
    local_dir="./models/GUI-Owl-32B",
    local_dir_use_symlinks=False
)
```

#### 方法2: 使用Git LFS
```bash
# 安装Git LFS
git lfs install

# 克隆GUI-Owl-7B
git clone https://huggingface.co/mPLUG/GUI-Owl-7B ./models/GUI-Owl-7B

# 克隆GUI-Owl-32B (可选)
git clone https://huggingface.co/mPLUG/GUI-Owl-32B ./models/GUI-Owl-32B
```

### 步骤2: 部署VLLM推理服务

#### 2.1 安装VLLM
```bash
# 安装VLLM
pip install vllm

# 验证安装
python -c "import vllm; print('VLLM安装成功')"
```

#### 2.2 启动GUI-Owl-7B服务
```bash
# 启动7B模型服务 (需要约12-16GB显存)
python -m vllm.entrypoints.openai.api_server \
    --model ./models/GUI-Owl-7B \
    --served-model-name gui-owl-7b \
    --host 0.0.0.0 \
    --port 8000 \
    --gpu-memory-utilization 0.8 \
    --max-model-len 8192 \
    --trust-remote-code

# 后台运行
nohup python -m vllm.entrypoints.openai.api_server \
    --model ./models/GUI-Owl-7B \
    --served-model-name gui-owl-7b \
    --host 0.0.0.0 \
    --port 8000 \
    --gpu-memory-utilization 0.8 \
    --max-model-len 8192 \
    --trust-remote-code > vllm_7b.log 2>&1 &
```

#### 2.3 启动GUI-Owl-32B服务 (可选)
```bash
# 启动32B模型服务 (需要约48-64GB显存或多GPU)
python -m vllm.entrypoints.openai.api_server \
    --model ./models/GUI-Owl-32B \
    --served-model-name gui-owl-32b \
    --host 0.0.0.0 \
    --port 8001 \
    --gpu-memory-utilization 0.9 \
    --tensor-parallel-size 2 \
    --max-model-len 8192 \
    --trust-remote-code

# 多GPU配置示例 (4张GPU)
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m vllm.entrypoints.openai.api_server \
    --model ./models/GUI-Owl-32B \
    --served-model-name gui-owl-32b \
    --host 0.0.0.0 \
    --port 8001 \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.95 \
    --max-model-len 8192 \
    --trust-remote-code
```

#### 2.4 验证服务启动
```bash
# 检查服务状态
curl http://localhost:8000/v1/models

# 预期输出类似:
# {
#   "object": "list",
#   "data": [
#     {
#       "id": "gui-owl-7b",
#       "object": "model",
#       "created": 1234567890,
#       "owned_by": "vllm"
#     }
#   ]
# }
```

## Android设备配置

### 真实设备设置

#### 步骤1: 启用开发者选项
```bash
# 在Android设备上:
# 1. 进入 设置 > 关于手机
# 2. 连续点击"版本号"7次启用开发者选项
# 3. 返回设置，进入 开发者选项
# 4. 启用"USB调试"
# 5. 如果是HyperOS/MIUI系统，同时启用"USB调试(安全设置)"
```

#### 步骤2: 连接和验证设备
```bash
# 1. 用USB线连接设备到电脑
# 2. 在设备上选择"传输文件"模式
# 3. 授权电脑的USB调试请求

# 验证设备连接
adb devices
# 应该显示类似输出:
# List of devices attached
# XXXXXXXXXX    device
```

#### 步骤3: 安装ADB键盘
```bash
# 下载ADB键盘APK
wget https://github.com/senzhk/ADBKeyBoard/raw/master/ADBKeyboard.apk

# 安装到设备
adb install ADBKeyboard.apk

# 在设备上手动设置:
# 设置 > 系统 > 语言和输入法 > 虚拟键盘 > 管理键盘
# 启用"ADB Keyboard"并设为默认输入法
```

### Android模拟器设置

#### 创建优化的模拟器启动脚本
```bash
# 创建优化的模拟器启动脚本
cat > start_emulator_optimized.sh << 'EOF'
#!/bin/bash

AVD_NAME="AndroidWorldAvd"

echo "启动优化的Android模拟器..."

# 检查AVD是否存在
if ! emulator -list-avds | grep -q "$AVD_NAME"; then
    echo "错误: AVD '$AVD_NAME' 不存在"
    echo "可用的AVD:"
    emulator -list-avds
    exit 1
fi

# 启动模拟器 (优化配置)
emulator -avd "$AVD_NAME" \
    -gpu swiftshader_indirect \
    -memory 4096 \
    -cores 4 \
    -cache-size 1024 \
    -no-snapshot-save \
    -no-snapshot-load \
    -wipe-data \
    -no-audio \
    -netdelay none \
    -netspeed full \
    -qemu -enable-kvm &

echo "等待模拟器启动..."
adb wait-for-device

echo "模拟器启动完成"
adb shell input keyevent 82  # 解锁屏幕
adb shell settings put global window_animation_scale 0.0
adb shell settings put global transition_animation_scale 0.0
adb shell settings put global animator_duration_scale 0.0

echo "模拟器优化完成"
EOF

chmod +x start_emulator_optimized.sh
```

## GUI-Critic-R1部署

### 步骤1: 安装依赖
```bash
cd GUI-Critic-R1
pip install -r requirement.txt
```

### 步骤2: 配置API密钥
```python
# 编辑 statistic.py 文件
# 配置Qwen-72B API (用于建议有效性计算)

# 示例配置:
import dashscope
dashscope.api_key = "your_dashscope_api_key"

# 或者设置环境变量
export DASHSCOPE_API_KEY="your_dashscope_api_key"
```

### 步骤3: 下载测试数据
```bash
# 测试文件已包含在项目中:
# - test_files/gui_i.jsonl (GUI-I数据集)
# - test_files/gui_s.jsonl (GUI-S数据集) 
# - test_files/gui_web.jsonl (GUI-Web数据集)

# 如需完整数据集，请联系作者获取
```

## 优化配置

### VLLM服务优化
```bash
# 创建优化的VLLM启动脚本
cat > start_vllm_optimized.sh << 'EOF'
#!/bin/bash

MODEL_PATH="./models/GUI-Owl-7B"
MODEL_NAME="gui-owl-7b"
PORT=8000

# 检测GPU数量
GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
echo "检测到 $GPU_COUNT 个GPU"

# 根据GPU数量调整配置
if [ $GPU_COUNT -ge 2 ]; then
    TENSOR_PARALLEL_SIZE=2
    GPU_MEMORY_UTIL=0.9
else
    TENSOR_PARALLEL_SIZE=1
    GPU_MEMORY_UTIL=0.8
fi

echo "启动VLLM服务..."
echo "模型: $MODEL_PATH"
echo "端口: $PORT"
echo "并行度: $TENSOR_PARALLEL_SIZE"
echo "GPU内存利用率: $GPU_MEMORY_UTIL"

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name "$MODEL_NAME" \
    --host 0.0.0.0 \
    --port $PORT \
    --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
    --gpu-memory-utilization $GPU_MEMORY_UTIL \
    --max-model-len 8192 \
    --max-num-seqs 32 \
    --max-num-batched-tokens 8192 \
    --trust-remote-code \
    --disable-log-stats \
    --quantization awq \
    --enable-prefix-caching \
    --use-v2-block-manager
EOF

chmod +x start_vllm_optimized.sh
```

### 环境变量配置
```bash
# 创建环境配置文件
cat > .env << 'EOF'
# VLLM配置
VLLM_HOST=0.0.0.0
VLLM_PORT=8000
VLLM_MODEL_PATH=./models/GUI-Owl-7B
VLLM_MODEL_NAME=gui-owl-7b

# Android配置
ADB_PATH=/opt/platform-tools/adb
ANDROID_AVD_HOME=~/.android/avd
ANDROID_SDK_ROOT=~/Android/Sdk

# API配置
API_KEY=your_api_key_here
BASE_URL=http://localhost:8000/v1

# 其他配置
CUDA_VISIBLE_DEVICES=0
PYTHONPATH=.
EOF

# 加载环境变量
source .env
```

## 验证部署

### 完整部署验证脚本
```bash
# 创建部署验证脚本
cat > verify_deployment.sh << 'EOF'
#!/bin/bash

echo "=== Mobile-Agent-v3 部署验证 ==="

# 1. 检查Python环境
echo "1. 检查Python环境..."
python --version
pip list | grep -E "(torch|vllm|qwen|transformers)"

# 2. 检查CUDA
echo "2. 检查CUDA环境..."
nvidia-smi
python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}')"

# 3. 检查ADB
echo "3. 检查ADB连接..."
adb version
adb devices

# 4. 检查模型文件
echo "4. 检查模型文件..."
ls -la ./models/

# 5. 检查VLLM服务
echo "5. 检查VLLM服务..."
curl -s http://localhost:8000/v1/models || echo "VLLM服务未启动"

# 6. 检查项目文件
echo "6. 检查项目文件..."
ls -la Mobile-Agent-v3/
ls -la GUI-Critic-R1/

echo "=== 验证完成 ==="
EOF

chmod +x verify_deployment.sh
./verify_deployment.sh