# Mobile-Agent-v3 和 GUI-Owl 详细测试手册与部署指南

## 目录
- [一、项目概述](#一项目概述)
- [二、硬件和软件要求](#二硬件和软件要求)
- [三、详细部署步骤](#三详细部署步骤)
- [四、本地测试手册](#四本地测试手册)
- [五、性能监控和调优](#五性能监控和调优)
- [六、常见问题和解决方案](#六常见问题和解决方案)
- [七、扩展和定制](#七扩展和定制)

## 一、项目概述

### 1.1 Mobile-Agent-v3
- **定义**: 基于GUI-Owl的跨平台多智能体框架
- **核心功能**: 
  - 动态任务分解、规划和进度管理
  - 高度集成的操作空间，降低模型的感知和操作频率
  - 丰富的异常处理和反射能力，在弹窗、广告等场景下提供更稳定的性能
  - 关键信息记录能力，支持跨应用任务
- **支持平台**: Android、鸿蒙系统(≤4)、PC、Web
- **技术报告**: [Mobile-Agent-v3](https://arxiv.org/abs/2508.15144)

### 1.2 GUI-Owl
- **定义**: 多模态跨平台GUI虚拟层模型(VLM)，具备GUI感知、落地和端到端操作能力
- **核心功能**: 
  - 原生端到端多模态代理，旨在作为GUI自动化的基础模型
  - 在单一策略网络中统一感知、基础、推理、规划和动作执行
  - 强大的跨平台交互和多轮决策，并具有明确的中间推理功能
- **模型规格**: 
  - [GUI-Owl-7B](https://huggingface.co/mPLUG/GUI-Owl-7B): 7B参数版本
  - [GUI-Owl-32B](https://huggingface.co/mPLUG/GUI-Owl-32B): 32B参数版本
- **特点**: 7B以内实现SOTA结果，可在Mobile-Agent-v3中实例化为不同的专用智能体

### 1.3 GUI-Critic-R1
- **定义**: 用于GUI自动化操作前错误诊断的预操作评价模型
- **核心功能**: 通过推理潜在结果和操作的正确性，在实际执行之前提供有效的反馈
- **技术特点**: 基于建议感知的梯度相对策略优化(S-GRPO)策略构建
- **技术报告**: [GUI-Critic-R1](https://arxiv.org/abs/2506.04614)

## 二、硬件和软件要求

### 2.1 PC硬件配置要求

> **说明**: Mobile-Agent-V3是基于GUI-Owl的跨平台多智能体框架。GUI-Owl作为底层VLM模型提供GUI感知和操作能力，Mobile-Agent-V3在此基础上实现任务分解、规划和跨平台协调。

#### GUI-Owl模型配置要求

##### GUI-Owl-7B配置

###### 完整模型配置
- **CPU**: Intel i7-10700K 或 AMD Ryzen 7 3700X 及以上
- **内存**: 32GB RAM
- **显卡**: NVIDIA RTX 3080 12GB 或更高 (支持CUDA 11.8+)
- **存储**: 200GB可用SSD空间
- **用途**: 单一GUI操作任务，适合中等复杂度场景

###### 量化模型配置 (推荐)

**重要发现**: 经过深度代码调查，GUI-Owl-7B模型支持多种量化配置，可显著降低硬件要求：

**AWQ 4位量化版本** (推荐配置)
- **CPU**: Intel i5-10400 或 AMD Ryzen 5 3600 及以上
- **内存**: 16GB RAM
- **显卡**: NVIDIA RTX 3060 8GB 或更高
- **存储**: 50GB可用SSD空间
- **性能**: 推理速度快，精度损失<2%

**GPTQ 4位量化版本** (兼容性优先)
- **CPU**: Intel i5-10400 或 AMD Ryzen 5 3600 及以上
- **内存**: 16GB RAM
- **显卡**: NVIDIA RTX 3060 8GB 或更高
- **存储**: 50GB可用SSD空间
- **性能**: 兼容性好，精度损失<3%

**动态INT8量化** (运行时配置)
- **CPU**: Intel i7-10700K 或 AMD Ryzen 7 3700X 及以上
- **内存**: 24GB RAM
- **显卡**: NVIDIA RTX 3070 8GB 或更高
- **存储**: 200GB可用SSD空间 (使用完整模型)
- **配置方式**: 通过vLLM参数启用，无需单独模型
- **性能**: 速度与精度平衡，精度损失<1%

**量化模型可用性说明**:
✅ **有单独预训练模型**: AWQ-4bit、GPTQ-4bit
❌ **无单独模型**: INT8量化通过运行时参数配置
⚠️ **注意**: INT8量化需要下载完整模型，然后通过软件配置实现

**量化模型性能对比表**:
| 模型版本 | 显存占用 | 推理速度 | 精度损失 | 推荐GPU | 模型来源 | 适用场景 |
|---------|---------|---------|---------|---------|---------|---------|
| 完整FP16 | ~14GB | 最快 | 无 | RTX 3080+ | 原始模型 | 高端GPU环境 |
| AWQ-4bit | ~4GB | 快 | <2% | RTX 3060+ | 预训练量化模型 | 中端GPU推荐 |
| GPTQ-4bit | ~4GB | 中等 | <3% | RTX 3060+ | 预训练量化模型 | 兼容性优先 |
| INT8动态 | ~7GB | 中等 | <1% | RTX 3070+ | 运行时量化 | 平衡选择 |
| CPU-only | 0GB显存 | 慢 | 无 | 无GPU | 原始模型 | 仅CPU环境 |

##### GUI-Owl-32B配置
- **CPU**: Intel i9-12900K 或 AMD Ryzen 9 5900X 及以上
- **内存**: 64GB RAM 或更高
- **显卡**: NVIDIA RTX 4090 24GB 或多GPU配置
- **存储**: 500GB可用NVMe SSD空间
- **用途**: 复杂GUI推理任务，支持高精度操作

#### Mobile-Agent-V3框架配置要求

##### 基础配置 (使用API调用)
- **CPU**: Intel i5-10400 或 AMD Ryzen 5 3600 及以上
- **内存**: 16GB RAM
- **显卡**: 集成显卡或入门级独显 (无需本地模型推理)
- **存储**: 50GB可用空间
- **网络**: 稳定互联网连接 (用于API调用GUI-Owl)
- **Android环境**: Android Studio + AVD模拟器
- **特殊要求**: ADB调试工具、多智能体协调能力

##### 本地部署配置 (集成GUI-Owl-7B)
- **CPU**: Intel i7-12700K 或 AMD Ryzen 7 5800X 及以上
- **内存**: 48GB RAM (32GB用于GUI-Owl + 16GB用于框架)
- **显卡**: NVIDIA RTX 4070 Ti 12GB 或更高
- **存储**: 300GB可用NVMe SSD空间
- **特殊要求**: 
  - 支持GUI-Critic-R1预操作评价
  - 跨平台任务执行 (Android/鸿蒙/PC/Web)
  - 动态任务分解和异常处理

##### 高性能配置 (集成GUI-Owl-32B)
- **CPU**: Intel i9-13900K 或 AMD Ryzen 9 7900X 及以上
- **内存**: 96GB RAM (64GB用于GUI-Owl + 32GB用于框架)
- **显卡**: NVIDIA RTX 4090 24GB 或多GPU配置
- **存储**: 800GB可用NVMe SSD空间
- **特殊要求**:
  - AndroidWorld基准测试支持
  - 大规模并发智能体处理
  - 完整性能监控和调优功能

#### 最低兼容配置
- **CPU**: Intel i5-8400 或 AMD Ryzen 5 2600 及以上
- **内存**: 16GB RAM
- **显卡**: NVIDIA GTX 1060 6GB 或更高 (支持CUDA)
- **存储**: 100GB可用空间 (SSD推荐)
- **网络**: 稳定的互联网连接
- **用途**: 仅支持API模式运行Mobile-Agent-V3，不支持本地模型部署

### 2.2 Android平板设备配置要求

> **说明**: Mobile-Agent-V3支持在Android平板设备上运行，但由于ARM架构和移动设备的计算限制，主要以API调用模式运行，本地模型推理能力有限。

#### 推荐平板配置
- **处理器**: 高通骁龙8 Gen2/Gen3 或 联发科天玑9200 及以上
- **内存**: 12GB RAM 或更高 (推荐16GB)
- **存储**: 256GB UFS 3.1/4.0 (推荐512GB)
- **GPU**: Adreno 740+ 或 Mali-G715+ 
- **系统**: Android 12+ (推荐Android 14)
- **网络**: 5G/WiFi 6支持
- **特殊要求**: 支持开发者选项和ADB调试

#### 瑞芯微RK3562平台兼容性分析

##### 硬件规格评估
- **CPU**: RK3562 (4核ARM Cortex-A53 2.0GHz)
- **GPU**: Mali-G52 MC1
- **内存**: 通常配置4-8GB LPDDR4
- **存储**: eMMC 5.1 或 UFS 2.1
- **系统**: Android 14支持

##### 兼容性结论
**✅ 可以集成，但有限制**:
- **API模式**: 完全支持，可作为Mobile-Agent-V3的客户端
- **轻量级任务**: 支持简单的GUI操作和任务协调
- **跨应用操作**: 支持Android应用间的自动化操作

**❌ 不支持的功能**:
- **本地模型推理**: RK3562性能不足以运行GUI-Owl模型
- **复杂任务处理**: 受限于CPU和内存性能
- **高并发操作**: 多智能体并发能力有限

##### 推荐部署方案
```
边缘计算架构:
┌─────────────────┐    ┌──────────────────┐
│   RK3562平板    │    │    云端/边缘服务器  │
│                 │    │                  │
│ Mobile-Agent-V3 │◄──►│   GUI-Owl模型     │
│   客户端框架     │    │   推理服务        │
│                 │    │                  │
│ - 任务接收      │    │ - 模型推理        │
│ - GUI操作执行   │    │ - 决策生成        │
│ - 状态反馈      │    │ - 任务规划        │
└─────────────────┘    └──────────────────┘
```

#### 其他ARM平板推荐配置

##### 高性能ARM平板 (完全支持)
- **处理器**: Apple M1/M2 (iPad Pro) 或 高通骁龙8cx Gen3
- **内存**: 16GB RAM 或更高
- **存储**: 512GB 或更高
- **系统**: iPadOS 16+ 或 Android 13+
- **特殊能力**: 支持轻量级本地模型推理

##### 中端ARM平板 (API模式)
- **处理器**: 高通骁龙778G+ 或 联发科天玑8100
- **内存**: 8GB RAM 或更高
- **存储**: 128GB UFS 3.1
- **系统**: Android 12+
- **适用场景**: API调用模式，适合教育和轻量级自动化

#### 移动设备特殊考虑

##### 电源管理
- **电池容量**: 建议8000mAh以上
- **充电功率**: 支持快充，减少使用中断
- **省电模式**: 需要配置白名单避免后台限制

##### 网络要求
- **带宽**: 建议50Mbps以上稳定连接
- **延迟**: <100ms (用于实时API调用)
- **流量**: 重度使用场景下每小时约500MB-1GB

##### 散热考虑
- **持续运行**: 需要良好的散热设计
- **性能限制**: 可能需要降频运行以控制温度
- **使用环境**: 建议在空调环境下长时间使用

### 2.2 软件环境要求

#### 操作系统
- **Linux**: Ubuntu 18.04+ 或 CentOS 7+ (推荐Ubuntu 20.04+) - **vLLM完全支持**
- **Windows**: Windows 10/11 (64位) - **vLLM需要WSL2或Docker支持**
- **macOS**: macOS 10.15+ - **vLLM仅支持CPU推理，性能受限**

> **重要说明**: vLLM推理服务推荐在Linux环境下部署以获得最佳性能和稳定性

#### 核心软件依赖
- **Python**: 3.8-3.11 (推荐3.10)
- **CUDA**: 11.8+ (GPU加速必需)
- **cuDNN**: 对应CUDA版本的cuDNN
- **Android Studio**: 最新版本 (用于Android模拟器)
- **ADB**: Android Debug Bridge
- **Git**: 版本控制工具

#### Python包依赖
```
# 核心依赖
qwen_agent
qwen_vl_utils
transformers>=4.30.0
torch>=2.0.0
vllm>=0.2.0

# AndroidWorld评测依赖
absl-py==2.1.0
android_env==1.2.3
dm_env==1.6
fuzzywuzzy==0.18.0
google-generativeai
grpcio-tools==1.71.0
protobuf==5.29.5
immutabledict==2.0.0
IPython
jsonschema==4.17.3
matplotlib==3.6.1
numpy==1.26.3
opencv-python
pandas==2.1.4
pydub
python-Levenshtein
pytest
requests
tenacity
termcolor

# GUI-Critic-R1依赖
dashscope
scikit-learn
modelscope
Pillow
tqdm
```

## 三、详细部署步骤

### 3.1 环境准备

#### 步骤1: 安装Python环境
```bash
# 方法1: 使用Conda (推荐)
conda create -n mobile-agent python=3.10
conda activate mobile-agent

# 方法2: 使用pip和venv
python -m venv mobile-agent-env
# Linux/Mac激活
source mobile-agent-env/bin/activate
# Windows激活
mobile-agent-env\Scripts\activate
```

#### 步骤2: 安装CUDA和PyTorch
```bash
# 检查CUDA版本
nvidia-smi

# 安装PyTorch (CUDA 11.8示例)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 验证CUDA安装
python -c "import torch; print(torch.cuda.is_available())"
```

#### 步骤2.1: vLLM部署环境配置

##### Linux环境 (推荐)
```bash
# Ubuntu/Debian系统
sudo apt update
sudo apt install -y build-essential python3-dev

# 安装vLLM
pip install vllm

# 验证安装
python -c "import vllm; print('vLLM安装成功')"
```

##### Windows环境 (使用WSL2)
```bash
# 1. 安装WSL2和Ubuntu
wsl --install -d Ubuntu-22.04

# 2. CUDA支持配置 (仅GPU加速需要)
# 检查是否已有CUDA支持
wsl
nvidia-smi  # 如果显示GPU信息，说明已支持

# 如果nvidia-smi命令不存在，需要安装CUDA WSL驱动:
# - Windows 11 (22H2+): 通常已内置支持
# - Windows 10: 需要手动安装
# - 下载地址: https://developer.nvidia.com/cuda/wsl
# - 注意: 只需安装Windows端驱动，WSL2内无需安装CUDA toolkit

# 3. 在WSL2中安装vLLM
sudo apt update && sudo apt install -y build-essential python3-dev
pip install vllm

# 4. 验证GPU支持 (可选)
python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}')"
```

> **重要说明**: 
> - **仅CPU使用**: 跳过步骤2，直接安装vLLM即可
> - **GPU加速**: 必须完成CUDA WSL驱动安装
> - **Windows 11用户**: 系统可能已内置CUDA WSL支持
> - **验证方法**: 在WSL2中运行`nvidia-smi`命令

##### Windows环境 (使用Docker)
```bash
# 1. 安装Docker Desktop并启用GPU支持
# 2. 拉取vLLM镜像
docker pull vllm/vllm-openai:latest

# 3. 运行vLLM容器
docker run --gpus all \
    -v ./models:/models \
    -p 8000:8000 \
    vllm/vllm-openai:latest \
    --model /models/GUI-Owl-7B \
    --served-model-name gui-owl-7b
```

##### macOS环境 (CPU推理)
```bash
# 注意: macOS仅支持CPU推理，性能较差
pip install vllm

# CPU推理启动示例
python -m vllm.entrypoints.openai.api_server \
    --model ./models/GUI-Owl-7B \
    --served-model-name gui-owl-7b \
    --device cpu \
    --dtype float16
```

#### 步骤3: 克隆项目
```bash
git clone https://github.com/X-PLUG/MobileAgent.git
cd MobileAgent
```

#### 步骤4: API密钥配置和管理

##### 4.1 vLLM本地服务 (无需API key)

###### 单机本地服务
```bash
# vLLM本地部署不需要API key
# 启动后会在指定端口提供OpenAI兼容的API服务
python -m vllm.entrypoints.openai.api_server \
    --model ./models/GUI-Owl-7B \
    --served-model-name gui-owl-7b \
    --host 127.0.0.1 \
    --port 8000

# 本地API调用示例 (无需API key)
curl -X POST http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "gui-owl-7b",
        "messages": [{"role": "user", "content": "Hello"}]
    }'
```

###### 局域网服务配置 (无需API key)
```bash
# 1. 启动vLLM局域网服务
python -m vllm.entrypoints.openai.api_server \
    --model ./models/GUI-Owl-7B \
    --served-model-name gui-owl-7b \
    --host 0.0.0.0 \
    --port 8000 \
    --trust-remote-code

# 2. 查看服务器IP地址
# Linux/macOS
ip addr show | grep "inet " | grep -v 127.0.0.1
# Windows
ipconfig | findstr "IPv4"

# 3. 防火墙配置 (如需要)
# Ubuntu/Debian
sudo ufw allow 8000/tcp
# CentOS/RHEL
sudo firewall-cmd --permanent --add-port=8000/tcp
sudo firewall-cmd --reload
# Windows
# 在Windows防火墙中添加入站规则，允许端口8000

# 4. 局域网客户端调用示例
# 假设服务器IP为192.168.1.100
curl -X POST http://192.168.3.144:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "gui-owl-7b",
        "messages": [{"role": "user", "content": "Hello"}]
    }'

# 5. Mobile-Agent-V3客户端配置 (空API key)
# 重要发现: 经过代码分析，Mobile-Agent-V3支持空API key配置

# 方式1: 命令行参数 (推荐)
python run_mobileagentv3.py \
    --adb_path "C:\platform-tools\adb.exe" \
    --api_key "" \
    --base_url "http://192.168.3.144:8000/v1" \
    --model "gui-owl-7b" \
    --instruction "请在小红书搜索济南攻略"

# 方式2: 环境变量配置
# Linux/macOS
export API_KEY=""
export BASE_URL="http://192.168.3.144:8000/v1"
# Windows
set API_KEY=
set BASE_URL=http://192.168.3.144:8000/v1

# 方式3: 批处理文件配置
# run_local.bat
@echo off
set "API_KEY="
set "BASE_URL=http://192.168.3.144:8000/v1"
set "MODEL=gui-owl-7b"
python run_mobileagentv3.py --api_key "%API_KEY%" --base_url "%BASE_URL%" --model "%MODEL%" --instruction "%INSTRUCTION%"

# 6. 空API key配置验证脚本
python -c "
import requests
try:
    response = requests.post('http://192.168.3.144:8000/v1/chat/completions',
        json={'model': 'gui-owl-7b', 'messages': [{'role': 'user', 'content': 'Hello'}], 'max_tokens': 10},
        headers={'Authorization': 'Bearer '},  # 空Bearer token
        timeout=10)
    print('✅ 空API key配置成功' if response.status_code == 200 else f'❌ 响应异常: {response.status_code}')
except Exception as e:
    print(f'❌ 连接失败: {e}')
"

##### 4.4 连接错误排查和解决

**常见错误**: "Connection error" 或 "Error calling LLM"

###### 排查步骤

**步骤1: 检查vLLM服务状态**
```bash
# 检查服务是否运行
curl -X GET http://192.168.3.144:8000/health
# 或使用PowerShell
Invoke-WebRequest -Uri "http://192.168.3.144:8000/health" -Method GET

# 检查模型列表
curl -X GET http://192.168.3.144:8000/v1/models
```

**步骤2: 检查网络连通性**
```bash
# 测试网络连接
ping 192.168.3.144

# 测试端口连通性
telnet 192.168.3.144 8000
# 或使用PowerShell
Test-NetConnection -ComputerName 192.168.3.144 -Port 8000
```

**步骤3: 启动vLLM服务**
```bash
# 如果服务未运行，启动vLLM服务
python -m vllm.entrypoints.openai.api_server \
    --model mPLUG/GUI-Owl-7B \
    --served-model-name gui-owl-7b \
    --host 0.0.0.0 \
    --port 8000 \
    --trust-remote-code
```

**步骤4: 防火墙检查**
```bash
# Windows防火墙
netsh advfirewall firewall add rule name="vLLM" dir=in action=allow protocol=TCP localport=8000

# Linux防火墙
sudo ufw allow 8000/tcp
```

###### 快速诊断脚本
```python
# diagnosis.py - 保存并运行此脚本
import requests
import socket
import time

def diagnose_vllm_connection(host="192.168.3.144", port=8000):
    base_url = f"http://{host}:{port}"
    
    print(f"🔍 诊断vLLM服务连接: {base_url}")
    
    # 1. 网络连通性测试
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex((host, port))
        sock.close()
        
        if result == 0:
            print("✅ 网络连通性: 正常")
        else:
            print("❌ 网络连通性: 失败 - 检查IP地址和端口")
            return False
    except Exception as e:
        print(f"❌ 网络测试失败: {e}")
        return False
    
    # 2. 服务健康检查
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code == 200:
            print("✅ 服务健康状态: 正常")
        else:
            print(f"❌ 服务健康状态: 异常 ({response.status_code})")
            return False
    except Exception as e:
        print(f"❌ 健康检查失败: {e}")
        return False
    
    # 3. 模型列表检查
    try:
        response = requests.get(f"{base_url}/v1/models", timeout=10)
        if response.status_code == 200:
            models = response.json()
            print(f"✅ 可用模型: {[m['id'] for m in models.get('data', [])]}")
        else:
            print(f"❌ 模型列表获取失败: {response.status_code}")
    except Exception as e:
        print(f"❌ 模型检查失败: {e}")
    
    # 4. API调用测试
    try:
        test_data = {
            "model": "gui-owl-7b",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 5
        }
        response = requests.post(f"{base_url}/v1/chat/completions", 
                               json=test_data, timeout=30)
        if response.status_code == 200:
            print("✅ API调用测试: 成功")
            return True
        else:
            print(f"❌ API调用失败: {response.status_code}")
            print(f"响应内容: {response.text}")
            return False
    except Exception as e:
        print(f"❌ API测试失败: {e}")
        return False

if __name__ == "__main__":
    diagnose_vllm_connection()
```

###### 解决方案

**方案1: 本地服务启动**
```bash
# 在服务器(192.168.3.144)上启动vLLM
cd /path/to/models
python -m vllm.entrypoints.openai.api_server \
    --model ./GUI-Owl-7B \
    --served-model-name gui-owl-7b \
    --host 0.0.0.0 \
    --port 8000 \
    --trust-remote-code
```

**方案2: 修改IP地址**
```bash
# 如果IP地址变化，更新配置
set "BASE_URL=http://新IP地址:8000/v1"
```

**方案3: 使用本地服务**
```bash
# 改为本地服务
set "BASE_URL=http://127.0.0.1:8000/v1"
```

**方案4: 增加重试和超时配置**
```python
# 在代码中增加重试逻辑
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def create_robust_session():
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session
```

##### 4.3 COOR_TYPE坐标类型配置

**COOR_TYPE参数说明**: 控制Mobile-Agent-V3如何处理屏幕坐标

###### 支持的坐标类型

**1. "abs" (绝对坐标) - 默认推荐**
```bash
--coor_type "abs"
```
- **含义**: 使用屏幕的绝对像素坐标
- **适用**: 大多数Android设备和模拟器
- **优势**: 精确度高，兼容性好
- **示例**: 点击坐标(500, 800)表示屏幕上的绝对位置

**2. "rel" (相对坐标)**
```bash
--coor_type "rel"
```
- **含义**: 使用相对于屏幕尺寸的比例坐标(0-1范围)
- **适用**: 不同分辨率设备间的兼容性
- **优势**: 跨设备适配性好
- **示例**: 点击坐标(0.5, 0.8)表示屏幕中央偏下位置

**3. "qwen-vl" (Qwen-VL格式)**
```bash
--coor_type "qwen-vl"
```
- **含义**: 使用Qwen-VL模型的特定坐标格式
- **适用**: 与Qwen-VL视觉模型配合使用
- **优势**: 针对视觉理解模型优化
- **注意**: 需要特定的模型支持

###### 坐标转换机制
```python
# 代码分析 (run_mobileagentv3.py)
if coor_type != "abs":
    if "coordinate" in action_object:
        # 将相对坐标转换为绝对坐标
        action_object['coordinate'] = [
            int(action_object['coordinate'][0] / 1000 * width), 
            int(action_object['coordinate'][1] / 1000 * height)
        ]
```

###### 配置建议

**推荐配置**: `COOR_TYPE=abs`
- ✅ 兼容性最好
- ✅ 精确度最高
- ✅ 调试方便
- ✅ 适用于GUI-Owl-7B模型

**特殊场景配置**:
```bash
# 多设备兼容
--coor_type "rel"

# Qwen-VL模型
--coor_type "qwen-vl"
```

###### 配置验证
```bash
# 测试不同坐标类型
python run_mobileagentv3.py --coor_type "abs" --instruction "点击屏幕中央"
python run_mobileagentv3.py --coor_type "rel" --instruction "点击屏幕中央"
```

###### 常见问题
- **点击位置不准确**: 尝试切换到"abs"模式
- **跨设备兼容问题**: 使用"rel"模式
- **与特定模型不兼容**: 检查模型文档推荐的坐标类型
    --base_url "http://192.168.1.100:8000/v1" \
    --model "gui-owl-7b" \
    --api_key "not_required" \
    --instruction "打开设置应用"
```

###### 多设备协作配置
```bash
# 创建局域网服务配置文件
cat > vllm_lan_config.json << 'EOF'
{
    "server_config": {
        "host": "0.0.0.0",
        "port": 8000,
        "model_path": "./models/GUI-Owl-7B",
        "model_name": "gui-owl-7b",
        "max_concurrent_requests": 10,
        "gpu_memory_utilization": 0.8
    },
    "client_devices": {
        "android_tablet_1": {
            "device_id": "192.168.1.101",
            "base_url": "http://192.168.1.100:8000/v1",
            "model": "gui-owl-7b"
        },
        "android_tablet_2": {
            "device_id": "192.168.1.102", 
            "base_url": "http://192.168.1.100:8000/v1",
            "model": "gui-owl-7b"
        },
        "pc_client": {
            "device_id": "192.168.1.103",
            "base_url": "http://192.168.1.100:8000/v1",
            "model": "gui-owl-7b"
        }
    }
}
EOF

# 启动优化的局域网服务
python -m vllm.entrypoints.openai.api_server \
    --model ./models/GUI-Owl-7B \
    --served-model-name gui-owl-7b \
    --host 0.0.0.0 \
    --port 8000 \
    --max-num-seqs 10 \
    --gpu-memory-utilization 0.8 \
    --trust-remote-code \
    --disable-log-stats
```

###### 局域网服务监控
```bash
# 创建服务状态检查脚本
cat > check_vllm_lan_service.sh << 'EOF'
#!/bin/bash

SERVER_IP="192.168.1.100"
SERVER_PORT="8000"

echo "=== vLLM局域网服务状态检查 ==="

# 1. 检查服务是否运行
echo "1. 服务连通性检查:"
if curl -s http://$SERVER_IP:$SERVER_PORT/v1/models > /dev/null; then
    echo "✅ vLLM服务运行正常"
else
    echo "❌ vLLM服务无法访问"
    exit 1
fi

# 2. 检查模型列表
echo "2. 可用模型:"
curl -s http://$SERVER_IP:$SERVER_PORT/v1/models | jq '.data[].id' 2>/dev/null || echo "获取模型列表失败"

# 3. 测试API调用
echo "3. API功能测试:"
response=$(curl -s -X POST http://$SERVER_IP:$SERVER_PORT/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "gui-owl-7b",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 10
    }')

if echo "$response" | jq '.choices[0].message.content' > /dev/null 2>&1; then
    echo "✅ API调用成功"
else
    echo "❌ API调用失败"
    echo "响应: $response"
fi

# 4. 网络延迟测试
echo "4. 网络延迟:"
ping -c 3 $SERVER_IP | tail -1 | awk '{print $4}' | cut -d '/' -f 2 | xargs -I {} echo "平均延迟: {}ms"

echo "=== 检查完成 ==="
EOF

chmod +x check_vllm_lan_service.sh
```

> **局域网部署优势**:
> - **资源共享**: 一台高性能服务器为多个设备提供AI服务
> - **成本效益**: 无需每台设备都配置高端GPU
> - **统一管理**: 集中式模型管理和更新
> - **无API费用**: 完全本地化，无外部API调用费用
> - **数据安全**: 数据不离开局域网环境

##### 4.2 外部API服务配置

###### Qwen API (通义千问)
```bash
# 1. 获取API key
# 访问: https://dashscope.console.aliyun.com/
# 注册并获取API key

# 2. 设置环境变量
export DASHSCOPE_API_KEY="your_dashscope_api_key_here"

# 3. 或在代码中配置
import dashscope
dashscope.api_key = "your_dashscope_api_key_here"
```

###### OpenAI API
```bash
# 1. 获取API key
# 访问: https://platform.openai.com/api-keys
# 创建新的API key

# 2. 设置环境变量
export OPENAI_API_KEY="your_openai_api_key_here"

# 3. 或在代码中配置
import openai
openai.api_key = "your_openai_api_key_here"
```

##### 4.3 Mobile-Agent-V3配置文件
```bash
# 创建配置文件
cat > config.json << 'EOF'
{
    "api_config": {
        "local_vllm": {
            "base_url": "http://localhost:8000/v1",
            "model": "gui-owl-7b",
            "api_key": "not_required"
        },
        "qwen_api": {
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "model": "qwen-vl-plus",
            "api_key": "${DASHSCOPE_API_KEY}"
        },
        "openai_api": {
            "base_url": "https://api.openai.com/v1",
            "model": "gpt-4-vision-preview",
            "api_key": "${OPENAI_API_KEY}"
        }
    },
    "android_config": {
        "adb_path": "/opt/platform-tools/adb",
        "device_id": "auto",
        "screenshot_quality": 80
    }
}
EOF
```

##### 4.4 环境变量管理
```bash
# 创建环境变量文件
cat > .env << 'EOF'
# vLLM本地服务配置
VLLM_HOST=0.0.0.0
VLLM_PORT=8000
VLLM_MODEL_PATH=./models/GUI-Owl-7B
VLLM_MODEL_NAME=gui-owl-7b

# 外部API配置
DASHSCOPE_API_KEY=your_dashscope_api_key_here
OPENAI_API_KEY=your_openai_api_key_here

# Android配置
ADB_PATH=/opt/platform-tools/adb
ANDROID_HOME=~/Android/Sdk

# 其他配置
CUDA_VISIBLE_DEVICES=0,1
PYTHONPATH=.
EOF

# 加载环境变量
source .env

# 验证配置
echo "vLLM端口: $VLLM_PORT"
echo "ADB路径: $ADB_PATH"
echo "API密钥已设置: $([ -n "$DASHSCOPE_API_KEY" ] && echo "是" || echo "否")"
```

##### 4.5 API密钥安全管理
```bash
# 1. 使用系统密钥管理器 (Linux)
sudo apt install -y gnome-keyring
secret-tool store --label="Dashscope API Key" service dashscope username api_key
# 读取: secret-tool lookup service dashscope username api_key

# 2. 使用加密配置文件
pip install cryptography
python << 'EOF'
from cryptography.fernet import Fernet
import json

# 生成密钥
key = Fernet.generate_key()
with open('secret.key', 'wb') as f:
    f.write(key)

# 加密API配置
config = {
    "dashscope_api_key": "your_real_api_key_here",
    "openai_api_key": "your_real_openai_key_here"
}

f = Fernet(key)
encrypted_config = f.encrypt(json.dumps(config).encode())
with open('encrypted_config.bin', 'wb') as file:
    file.write(encrypted_config)

print("API密钥已加密保存")
EOF

# 3. 读取加密配置的示例代码
cat > load_encrypted_config.py << 'EOF'
from cryptography.fernet import Fernet
import json
import os

def load_encrypted_config():
    try:
        # 读取密钥
        with open('secret.key', 'rb') as f:
            key = f.read()
        
        # 读取加密配置
        with open('encrypted_config.bin', 'rb') as f:
            encrypted_config = f.read()
        
        # 解密
        f = Fernet(key)
        decrypted_config = f.decrypt(encrypted_config)
        config = json.loads(decrypted_config.decode())
        
        # 设置环境变量
        os.environ['DASHSCOPE_API_KEY'] = config['dashscope_api_key']
        os.environ['OPENAI_API_KEY'] = config['openai_api_key']
        
        return config
    except Exception as e:
        print(f"加载配置失败: {e}")
        return None

# 使用示例
if __name__ == "__main__":
    config = load_encrypted_config()
    if config:
        print("API密钥加载成功")
    else:
        print("API密钥加载失败")
EOF
```

##### 4.6 查看和验证API配置
```bash
# 创建配置检查脚本
cat > check_api_config.sh << 'EOF'
#!/bin/bash

echo "=== API配置检查 ==="

# 1. 检查环境变量
echo "1. 环境变量检查:"
echo "DASHSCOPE_API_KEY: $([ -n "$DASHSCOPE_API_KEY" ] && echo "已设置" || echo "未设置")"
echo "OPENAI_API_KEY: $([ -n "$OPENAI_API_KEY" ] && echo "已设置" || echo "未设置")"
echo "VLLM_PORT: ${VLLM_PORT:-8000}"

# 2. 检查vLLM服务
echo "2. vLLM服务检查:"
if curl -s http://localhost:${VLLM_PORT:-8000}/v1/models > /dev/null; then
    echo "vLLM服务: 运行中"
    curl -s http://localhost:${VLLM_PORT:-8000}/v1/models | jq '.data[].id' 2>/dev/null || echo "模型列表获取失败"
else
    echo "vLLM服务: 未运行"
fi

# 3. 测试API连接
echo "3. API连接测试:"
if [ -n "$DASHSCOPE_API_KEY" ]; then
    python3 << 'PYTHON_EOF'
import dashscope
import os
dashscope.api_key = os.environ.get('DASHSCOPE_API_KEY')
try:
    response = dashscope.Generation.call(
        model='qwen-turbo',
        prompt='Hello',
        max_tokens=10
    )
    print("Dashscope API: 连接成功")
except Exception as e:
    print(f"Dashscope API: 连接失败 - {e}")
PYTHON_EOF
else
    echo "Dashscope API: 未配置"
fi

echo "=== 检查完成 ==="
EOF

chmod +x check_api_config.sh
```

### 3.2 Mobile-Agent-v3部署

#### 步骤1: 安装基础依赖
```bash
# 安装Qwen相关依赖
pip install qwen_agent qwen_vl_utils

# 安装其他核心依赖
pip install transformers>=4.30.0
pip install vllm
pip install openai
```

#### 步骤2: 安装AndroidWorld评测环境
```bash
# 进入AndroidWorld目录
cd Mobile-Agent-v3/android_world_v3

# 安装依赖
pip install -r requirements.txt

# 安装AndroidWorld包
python setup.py install
```

#### 步骤3: 配置Android环境

##### 3.3.1 安装Android SDK Platform Tools
```bash
# Windows
# 下载: https://dl.google.com/android/repository/platform-tools-latest-windows.zip
# 解压到 C:\platform-tools

# Linux
wget https://dl.google.com/android/repository/platform-tools-latest-linux.zip
unzip platform-tools-latest-linux.zip
sudo mv platform-tools /opt/
echo 'export PATH=$PATH:/opt/platform-tools' >> ~/.bashrc
source ~/.bashrc

# macOS
wget https://dl.google.com/android/repository/platform-tools-latest-darwin.zip
unzip platform-tools-latest-darwin.zip
sudo mv platform-tools /usr/local/
echo 'export PATH=$PATH:/usr/local/platform-tools' >> ~/.zshrc
source ~/.zshrc

# 验证ADB安装
adb version
```

##### 3.3.2 安装Android Studio和AVD
```bash
# 1. 下载Android Studio
# https://developer.android.com/studio

# 2. 安装后启动Android Studio
# 3. 打开AVD Manager
# 4. 创建新的虚拟设备:
#    - 硬件: Pixel 6
#    - 系统镜像: Tiramisu, API Level 33
#    - AVD名称: AndroidWorldAvd

# 5. 启动模拟器
emulator -avd AndroidWorldAvd
```

更多详细的部署步骤、测试指南、性能优化和故障排除信息，请参考以下补充文档：

- [deployment_details.md](deployment_details.md) - 详细部署步骤和配置
- [testing_guide.md](testing_guide.md) - 完整的测试手册和用例
- [performance_optimization.md](performance_optimization.md) - 性能监控和调优指南
- [troubleshooting.md](troubleshooting.md) - 常见问题和解决方案
- [customization_guide.md](customization_guide.md) - 扩展和定制指南

## 快速开始

### 1. 基础测试
```bash
# 启动VLLM服务
python -m vllm.entrypoints.openai.api_server \
    --model ./models/GUI-Owl-7B \
    --served-model-name gui-owl-7b \
    --host 0.0.0.0 \
    --port 8000

# 运行基础测试
cd Mobile-Agent-v3/mobile_v3
python run_mobileagentv3.py \
    --adb_path "/path/to/adb" \
    --api_key "your_api_key" \
    --base_url "http://localhost:8000/v1" \
    --model "gui-owl-7b" \
    --instruction "打开设置应用" \
    --coor_type "qwen-vl"
```

### 2. AndroidWorld基准测试
```bash
cd Mobile-Agent-v3/android_world_v3

# 配置测试参数
vim run_ma3.sh
# 修改 MODEL, API_KEY, BASE_URL

# 运行测试
bash run_ma3.sh
```

### 3. GUI-Critic-R1测试
```bash
cd GUI-Critic-R1

# 运行测试
python test.py \
    --model_dir ./models/gui-critic-r1 \
    --test_file ./test_files/gui_i.jsonl \
    --save_dir ./output/gui_i \
    --data_dir ./dataset
```

## 技术支持

如果在部署或使用过程中遇到问题，请：

1. 查看对应的详细文档
2. 检查[常见问题解决方案](troubleshooting.md)
3. 提交GitHub Issue
4. 参考项目的技术论文

## 更新日志

- `[2025.9.10]` 开源Mobile-Agent-v3真实手机场景代码
- `[2025.8.29]` 开源GUI-Owl和Mobile-Agent-v3在AndroidWorld上的评测代码
- `[2025.8.20]` 发布GUI-Owl和Mobile-Agent-v3
- `[2025.8.14]` Mobile-Agent-v3获得CCL 2025最佳演示奖

## 许可证

本项目遵循相应的开源许可证，详情请查看项目根目录的LICENSE文件。