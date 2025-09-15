# 常见问题和解决方案

## 部署相关问题

### 1. CUDA和GPU问题

#### 问题：CUDA版本不兼容
**症状：** `RuntimeError: CUDA error: no kernel image is available for execution on the device`

**解决方案：**
```bash
# 1. 检查CUDA版本
nvidia-smi
nvcc --version

# 2. 重新安装匹配的PyTorch
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 3. 验证安装
python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"
```

#### 问题：GPU内存不足
**症状：** `torch.cuda.OutOfMemoryError: CUDA out of memory`

**解决方案：**
```bash
# 1. 降低GPU内存利用率
--gpu-memory-utilization 0.7

# 2. 减少序列长度
--max-model-len 2048

# 3. 启用量化
--quantization awq

# 4. 使用更小的模型
# 使用GUI-Owl-7B而不是32B
```

### 2. 依赖安装问题

#### 问题：Python包冲突
**症状：** `ImportError: cannot import name 'xxx' from 'yyy'`

**解决方案：**
```bash
# 1. 创建新的虚拟环境
conda create -n mobile-agent-clean python=3.10
conda activate mobile-agent-clean

# 2. 按顺序安装依赖
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install vllm
pip install qwen_agent qwen_vl_utils
pip install -r requirements.txt
```

### 3. ADB和Android问题

#### 问题：ADB设备连接失败
**症状：** `adb: no devices/emulators found`

**解决方案：**
```bash
# 1. 重启ADB服务
adb kill-server
adb start-server

# 2. 检查设备授权
adb devices
# 如果显示"unauthorized"，在设备上点击"允许"

# 3. 检查USB调试设置
# 确保在Android设备上启用了USB调试
```

## 运行时问题

### 1. VLLM服务问题

#### 问题：VLLM服务启动失败
**解决方案：**
```bash
# 1. 检查端口占用
lsof -i :8000

# 2. 检查模型路径
ls -la ./models/GUI-Owl-7B/

# 3. 查看详细错误日志
python -m vllm.entrypoints.openai.api_server --model ./models/GUI-Owl-7B --verbose
```

#### 问题：API请求超时
**解决方案：**
```python
# 增加超时时间
import requests
response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json=payload,
    timeout=120  # 增加到120秒
)
```

### 2. Mobile-Agent-v3运行问题

#### 问题：坐标识别错误
**解决方案：**
```bash
# 1. 调整坐标类型
--coor_type qwen-vl

# 2. 检查屏幕分辨率
adb shell wm size

# 3. 启用调试模式
--debug True
```

## 性能问题

### 1. 推理速度慢

**诊断脚本：**
```python
import time
import requests

def test_inference_speed():
    start_time = time.time()
    response = requests.post("http://localhost:8000/v1/chat/completions", json={
        "model": "gui-owl-7b",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 10
    })
    latency = time.time() - start_time
    print(f"推理延迟: {latency:.2f}秒")
    
    if latency > 5:
        print("延迟过高，检查GPU利用率和模型配置")

test_inference_speed()
```

### 2. 内存使用过高

**内存优化：**
```python
import gc
import torch

def optimize_memory():
    # Python垃圾回收
    gc.collect()
    
    # GPU内存清理
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    print("内存清理完成")

optimize_memory()
```

## 快速排查清单

### 基础环境检查
- [ ] Python版本 (3.8-3.11)
- [ ] CUDA版本 (11.8+)
- [ ] GPU驱动版本
- [ ] 虚拟环境激活
- [ ] 依赖包安装完整

### 服务状态检查
- [ ] VLLM服务运行状态
- [ ] 端口占用情况 (8000)
- [ ] 模型文件完整性
- [ ] API响应正常

### Android环境检查
- [ ] ADB连接正常
- [ ] 设备授权状态
- [ ] USB调试启用
- [ ] ADB键盘安装

### 性能检查
- [ ] GPU利用率
- [ ] 内存使用率
- [ ] 磁盘空间
- [ ] 网络连接

## 日志收集脚本

```bash
#!/bin/bash
# collect_debug_info.sh

LOG_DIR="debug_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "收集调试信息到: $LOG_DIR"

# 系统信息
echo "=== 系统信息 ===" > "$LOG_DIR/system.txt"
uname -a >> "$LOG_DIR/system.txt"
python --version >> "$LOG_DIR/system.txt"

# GPU信息
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi > "$LOG_DIR/gpu.txt"
fi

# 进程信息
ps aux | grep -E "(vllm|python)" > "$LOG_DIR/processes.txt"

# 网络连接
netstat -tulpn | grep :8000 > "$LOG_DIR/network.txt"

echo "调试信息收集完成"
```

## 联系支持

如果问题仍未解决，请：

1. 运行日志收集脚本
2. 提供详细的错误信息
3. 说明系统配置和操作步骤
4. 提交GitHub Issue或联系技术支持