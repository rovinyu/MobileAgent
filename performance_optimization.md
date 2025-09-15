# 性能监控和调优指南

## 系统资源监控

### GPU监控

#### 基础GPU监控
```bash
# 实时GPU监控
watch -n 2 nvidia-smi

# GPU使用率监控
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw --format=csv -l 5

# GPU进程监控
nvidia-smi pmon -c 10
```

#### GPU性能分析脚本
```python
# gpu_monitor.py
import subprocess
import time
import json

def monitor_gpu_performance(duration=300, interval=10):
    """监控GPU性能"""
    print(f"开始监控GPU性能 {duration}秒...")
    
    samples = []
    start_time = time.time()
    
    while time.time() - start_time < duration:
        try:
            result = subprocess.run([
                'nvidia-smi', '--query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True)
            
            timestamp = time.time()
            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = line.split(', ')
                    sample = {
                        'timestamp': timestamp,
                        'gpu_index': int(parts[0]),
                        'gpu_util': int(parts[1]) if parts[1] != '[Not Supported]' else 0,
                        'memory_util': int(parts[2]) if parts[2] != '[Not Supported]' else 0,
                        'memory_used': int(parts[3]),
                        'memory_total': int(parts[4]),
                        'temperature': int(parts[5]) if parts[5] != '[Not Supported]' else 0,
                        'power_draw': float(parts[6]) if parts[6] != '[Not Supported]' else 0
                    }
                    samples.append(sample)
                    
                    print(f"GPU {sample['gpu_index']}: "
                          f"GPU={sample['gpu_util']}%, "
                          f"MEM={sample['memory_util']}%, "
                          f"TEMP={sample['temperature']}°C")
            
        except Exception as e:
            print(f"监控错误: {e}")
        
        time.sleep(interval)
    
    # 保存结果
    with open(f'gpu_performance_{int(time.time())}.json', 'w') as f:
        json.dump(samples, f, indent=2)
    
    return samples

if __name__ == "__main__":
    monitor_gpu_performance()
```

### 系统资源监控

#### 系统监控脚本
```bash
#!/bin/bash
# system_monitor.sh

LOG_FILE="system_monitor_$(date +%Y%m%d_%H%M%S).log"

echo "系统监控开始 - 日志: $LOG_FILE"

while true; do
    {
        echo "=== $(date) ==="
        
        # CPU和内存
        echo "CPU和内存使用:"
        top -bn1 | head -5
        
        # 磁盘使用
        echo "磁盘使用:"
        df -h
        
        # 网络连接
        echo "VLLM服务连接:"
        netstat -an | grep :8000 | wc -l
        
        # 关键进程
        echo "VLLM进程:"
        ps aux | grep vllm | grep -v grep
        
        echo "================================"
        
    } >> "$LOG_FILE"
    
    sleep 30
done
```

## VLLM服务优化

### 自动配置生成器
```python
# vllm_config_generator.py
import subprocess
import os

class VLLMConfigGenerator:
    def __init__(self):
        self.gpu_info = self.detect_gpus()
        self.system_info = self.detect_system()
    
    def detect_gpus(self):
        """检测GPU配置"""
        try:
            result = subprocess.run([
                'nvidia-smi', '--query-gpu=index,name,memory.total',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True)
            
            gpus = []
            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = line.split(', ')
                    gpus.append({
                        'index': int(parts[0]),
                        'name': parts[1],
                        'memory_mb': int(parts[2])
                    })
            return gpus
        except:
            return []
    
    def detect_system(self):
        """检测系统配置"""
        cpu_count = os.cpu_count()
        
        try:
            with open('/proc/meminfo', 'r') as f:
                meminfo = f.read()
            total_memory_kb = int([line for line in meminfo.split('\n') 
                                 if 'MemTotal' in line][0].split()[1])
            total_memory_gb = total_memory_kb / 1024 / 1024
        except:
            total_memory_gb = 16
        
        return {
            'cpu_count': cpu_count,
            'total_memory_gb': total_memory_gb
        }
    
    def generate_config(self, model_size='7B'):
        """生成VLLM配置"""
        config = {
            'model_path': f'./models/GUI-Owl-{model_size}',
            'served_model_name': f'gui-owl-{model_size.lower()}',
            'host': '0.0.0.0',
            'port': 8000
        }
        
        # GPU配置
        gpu_count = len(self.gpu_info)
        total_gpu_memory = sum(gpu['memory_mb'] for gpu in self.gpu_info)
        
        # 确定并行度
        if model_size == '32B' and gpu_count >= 4:
            config['tensor_parallel_size'] = 4
        elif gpu_count >= 2:
            config['tensor_parallel_size'] = 2
        else:
            config['tensor_parallel_size'] = 1
        
        # 内存利用率
        required_memory = 14000 if model_size == '7B' else 64000
        if total_gpu_memory >= required_memory * 1.5:
            config['gpu_memory_utilization'] = 0.9
        else:
            config['gpu_memory_utilization'] = 0.8
        
        # 序列长度
        if total_gpu_memory >= 40000:
            config['max_model_len'] = 8192
            config['max_num_seqs'] = 64
        else:
            config['max_model_len'] = 4096
            config['max_num_seqs'] = 32
        
        return config
    
    def create_startup_script(self, config, filename='start_vllm.sh'):
        """创建启动脚本"""
        script = f"""#!/bin/bash

# VLLM启动脚本 - 自动生成
# GPU数量: {len(self.gpu_info)}
# 总显存: {sum(gpu['memory_mb'] for gpu in self.gpu_info)/1024:.1f}GB

python -m vllm.entrypoints.openai.api_server \\
    --model {config['model_path']} \\
    --served-model-name {config['served_model_name']} \\
    --host {config['host']} \\
    --port {config['port']} \\
    --tensor-parallel-size {config['tensor_parallel_size']} \\
    --gpu-memory-utilization {config['gpu_memory_utilization']} \\
    --max-model-len {config['max_model_len']} \\
    --max-num-seqs {config['max_num_seqs']} \\
    --trust-remote-code \\
    --disable-log-stats \\
    --enable-prefix-caching
"""
        
        with open(filename, 'w') as f:
            f.write(script)
        
        os.chmod(filename, 0o755)
        print(f"启动脚本已创建: {filename}")
        return filename

def main():
    generator = VLLMConfigGenerator()
    
    print("=== 系统配置 ===")
    print(f"CPU核心: {generator.system_info['cpu_count']}")
    print(f"内存: {generator.system_info['total_memory_gb']:.1f}GB")
    print(f"GPU数量: {len(generator.gpu_info)}")
    
    for gpu in generator.gpu_info:
        print(f"  GPU {gpu['index']}: {gpu['name']} ({gpu['memory_mb']/1024:.1f}GB)")
    
    # 生成7B配置
    config_7b = generator.generate_config('7B')
    script_7b = generator.create_startup_script(config_7b, 'start_vllm_7b.sh')
    
    # 如果GPU足够，生成32B配置
    total_memory = sum(gpu['memory_mb'] for gpu in generator.gpu_info)
    if total_memory >= 48000:  # 48GB+
        config_32b = generator.generate_config('32B')
        script_32b = generator.create_startup_script(config_32b, 'start_vllm_32b.sh')
        print("32B配置也已生成")

if __name__ == "__main__":
    main()
```

### 性能调优参数

#### 关键参数说明
```yaml
# VLLM性能参数配置指南

# GPU相关
tensor_parallel_size: 
  - 1: 单GPU
  - 2: 双GPU并行
  - 4: 四GPU并行 (32B模型推荐)

gpu_memory_utilization:
  - 0.7: 保守设置，适合多任务
  - 0.8: 平衡设置，推荐
  - 0.9: 激进设置，单任务使用

# 序列和批处理
max_model_len:
  - 2048: 低显存配置
  - 4096: 标准配置
  - 8192: 高显存配置

max_num_seqs:
  - 16: 低显存
  - 32: 标准
  - 64: 高显存

# 优化选项
enable_prefix_caching: true  # 启用前缀缓存
use_v2_block_manager: true   # 使用V2块管理器
disable_log_stats: true      # 禁用日志统计
```

## 推理性能优化

### 批处理优化
```python
# batch_inference.py
import asyncio
import aiohttp
import json
import time

class BatchInferenceOptimizer:
    def __init__(self, base_url="http://localhost:8000/v1"):
        self.base_url = base_url
    
    async def batch_inference(self, requests, batch_size=4):
        """批量推理优化"""
        results = []
        
        async with aiohttp.ClientSession() as session:
            for i in range(0, len(requests), batch_size):
                batch = requests[i:i+batch_size]
                
                # 并发发送批次请求
                tasks = [
                    self.single_request(session, req) 
                    for req in batch
                ]
                
                batch_results = await asyncio.gather(*tasks)
                results.extend(batch_results)
                
                # 批次间隔
                if i + batch_size < len(requests):
                    await asyncio.sleep(0.1)
        
        return results
    
    async def single_request(self, session, request_data):
        """单个请求"""
        try:
            async with session.post(
                f"{self.base_url}/chat/completions",
                json=request_data,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {"error": f"HTTP {response.status}"}
        except Exception as e:
            return {"error": str(e)}

# 使用示例
async def main():
    optimizer = BatchInferenceOptimizer()
    
    # 准备测试请求
    requests = [
        {
            "model": "gui-owl-7b",
            "messages": [{"role": "user", "content": f"测试请求 {i}"}],
            "max_tokens": 100
        }
        for i in range(20)
    ]
    
    start_time = time.time()
    results = await optimizer.batch_inference(requests, batch_size=4)
    end_time = time.time()
    
    print(f"批量推理完成: {len(results)}个请求")
    print(f"总耗时: {end_time - start_time:.2f}秒")
    print(f"平均延迟: {(end_time - start_time)/len(results):.2f}秒/请求")

if __name__ == "__main__":
    asyncio.run(main())
```

## 内存优化

### 内存监控和清理
```python
# memory_optimizer.py
import psutil
import gc
import torch

class MemoryOptimizer:
    def __init__(self):
        self.process = psutil.Process()
    
    def get_memory_usage(self):
        """获取内存使用情况"""
        memory_info = self.process.memory_info()
        
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,  # 物理内存
            'vms_mb': memory_info.vms / 1024 / 1024,  # 虚拟内存
            'percent': self.process.memory_percent(),  # 内存占用百分比
            'available_mb': psutil.virtual_memory().available / 1024 / 1024
        }
    
    def get_gpu_memory(self):
        """获取GPU内存使用"""
        if torch.cuda.is_available():
            gpu_memory = {}
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                reserved = torch.cuda.memory_reserved(i) / 1024**3
                total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                
                gpu_memory[f'gpu_{i}'] = {
                    'allocated_gb': allocated,
                    'reserved_gb': reserved,
                    'total_gb': total,
                    'free_gb': total - reserved
                }
            return gpu_memory
        return {}
    
    def cleanup_memory(self):
        """清理内存"""
        # Python垃圾回收
        gc.collect()
        
        # GPU内存清理
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def monitor_memory(self, duration=300, interval=30):
        """监控内存使用"""
        print(f"开始内存监控 {duration}秒...")
        
        samples = []
        start_time = time.time()
        
        while time.time() - start_time < duration:
            memory_usage = self.get_memory_usage()
            gpu_memory = self.get_gpu_memory()
            
            sample = {
                'timestamp': time.time(),
                'memory': memory_usage,
                'gpu': gpu_memory
            }
            samples.append(sample)
            
            print(f"内存: {memory_usage['rss_mb']:.1f}MB ({memory_usage['percent']:.1f}%)")
            
            for gpu_id, gpu_info in gpu_memory.items():
                print(f"{gpu_id}: {gpu_info['allocated_gb']:.1f}GB/{gpu_info['total_gb']:.1f}GB")
            
            time.sleep(interval)
        
        return samples

if __name__ == "__main__":
    optimizer = MemoryOptimizer()
    optimizer.monitor_memory()
```

更多详细的性能优化内容请参考：
- [advanced_optimization.md](advanced_optimization.md) - 高级优化技巧
- [monitoring_tools.md](monitoring_tools.md) - 监控工具和脚本
- [troubleshooting_performance.md](troubleshooting_performance.md) - 性能问题排查