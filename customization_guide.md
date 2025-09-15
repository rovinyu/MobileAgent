# 扩展和定制指南

## 添加新任务类型

### 1. 自定义Mobile-Agent-v3任务

#### 创建自定义任务类
```python
# custom_tasks.py
from mobile_v3.utils.task_base import TaskBase

class CustomShoppingTask(TaskBase):
    """自定义购物任务"""
    
    def __init__(self, instruction, product_name, price_range=None):
        super().__init__(instruction)
        self.product_name = product_name
        self.price_range = price_range
        self.search_results = []
    
    def execute(self, agent):
        """执行购物任务"""
        steps = [
            f"打开购物应用",
            f"搜索商品: {self.product_name}",
            f"筛选价格范围: {self.price_range}" if self.price_range else "查看搜索结果",
            f"选择合适的商品",
            f"查看商品详情"
        ]
        
        for step in steps:
            result = agent.execute_step(step)
            if not result.success:
                return self.handle_failure(step, result.error)
        
        return self.create_success_result()
    
    def handle_failure(self, step, error):
        """处理执行失败"""
        print(f"步骤失败: {step}, 错误: {error}")
        # 实现重试逻辑或替代方案
        return self.create_failure_result(error)

# 使用示例
task = CustomShoppingTask(
    instruction="在淘宝搜索iPhone 15并查看价格",
    product_name="iPhone 15",
    price_range="8000-10000"
)
```

#### 集成自定义任务
```python
# task_manager.py
from custom_tasks import CustomShoppingTask

class TaskManager:
    def __init__(self):
        self.task_registry = {
            'shopping': CustomShoppingTask,
            'navigation': NavigationTask,
            'social': SocialTask
        }
    
    def create_task(self, task_type, **kwargs):
        """创建任务实例"""
        if task_type in self.task_registry:
            return self.task_registry[task_type](**kwargs)
        else:
            raise ValueError(f"未知任务类型: {task_type}")
    
    def register_task(self, task_type, task_class):
        """注册新任务类型"""
        self.task_registry[task_type] = task_class

# 使用示例
manager = TaskManager()
task = manager.create_task('shopping', 
    instruction="购买手机",
    product_name="iPhone 15",
    price_range="8000-10000"
)
```

### 2. 扩展AndroidWorld评测

#### 添加自定义评测任务
```python
# custom_android_tasks.py
from android_world_v3.task_eval import TaskEval

class CustomAppTask(TaskEval):
    """自定义应用任务评测"""
    
    def __init__(self, app_name, target_action):
        super().__init__()
        self.app_name = app_name
        self.target_action = target_action
    
    def setup(self, env):
        """设置测试环境"""
        # 确保应用已安装
        self.ensure_app_installed(env, self.app_name)
        
        # 重置应用状态
        self.reset_app_state(env, self.app_name)
    
    def evaluate(self, env_state):
        """评估任务完成情况"""
        # 检查目标动作是否完成
        if self.check_action_completed(env_state):
            return {
                'success': True,
                'score': 1.0,
                'details': f'{self.target_action} 执行成功'
            }
        else:
            return {
                'success': False,
                'score': 0.0,
                'details': f'{self.target_action} 执行失败'
            }
    
    def check_action_completed(self, env_state):
        """检查动作是否完成"""
        # 实现具体的检查逻辑
        # 例如：检查UI元素、应用状态等
        pass

# 注册自定义任务
def register_custom_tasks():
    """注册自定义任务到AndroidWorld"""
    from android_world_v3 import task_registry
    
    task_registry.register('custom_app_task', CustomAppTask)
    
    print("自定义任务已注册")
```

## 集成新模型

### 1. 添加新的LLM后端

#### 创建模型包装器
```python
# custom_llm_wrapper.py
from mobile_v3.utils.infer import MultimodalLlmWrapper

class CustomLLMWrapper(MultimodalLlmWrapper):
    """自定义LLM包装器"""
    
    def __init__(self, model_name, api_endpoint, api_key):
        super().__init__()
        self.model_name = model_name
        self.api_endpoint = api_endpoint
        self.api_key = api_key
    
    def predict_mm(self, prompt, images=None):
        """多模态预测"""
        try:
            # 构建请求
            request_data = self.build_request(prompt, images)
            
            # 发送请求
            response = self.send_request(request_data)
            
            # 解析响应
            return self.parse_response(response)
            
        except Exception as e:
            print(f"预测失败: {e}")
            return None
    
    def build_request(self, prompt, images):
        """构建API请求"""
        content = [{"type": "text", "text": prompt}]
        
        if images:
            for image in images:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image}"}
                })
        
        return {
            "model": self.model_name,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 1000
        }
    
    def send_request(self, request_data):
        """发送API请求"""
        import requests
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        response = requests.post(
            f"{self.api_endpoint}/chat/completions",
            json=request_data,
            headers=headers,
            timeout=60
        )
        
        response.raise_for_status()
        return response.json()
    
    def parse_response(self, response):
        """解析API响应"""
        if 'choices' in response and len(response['choices']) > 0:
            return response['choices'][0]['message']['content']
        else:
            raise ValueError("无效的API响应")

# 使用示例
custom_llm = CustomLLMWrapper(
    model_name="custom-model-v1",
    api_endpoint="https://api.custom-provider.com/v1",
    api_key="your_api_key"
)
```

#### 集成到Mobile-Agent-v3
```python
# integrate_custom_model.py
from mobile_v3.run_mobileagentv3 import MobileAgentV3
from custom_llm_wrapper import CustomLLMWrapper

def create_agent_with_custom_model():
    """使用自定义模型创建智能体"""
    
    # 创建自定义LLM包装器
    custom_llm = CustomLLMWrapper(
        model_name="custom-gui-model",
        api_endpoint="https://your-api.com/v1",
        api_key="your_api_key"
    )
    
    # 创建Mobile-Agent-v3实例
    agent = MobileAgentV3(
        llm_wrapper=custom_llm,
        adb_path="/path/to/adb",
        coor_type="text"  # 或其他坐标类型
    )
    
    return agent

# 使用自定义模型执行任务
agent = create_agent_with_custom_model()
result = agent.execute_instruction("打开设置并调整亮度")
```

### 2. 添加新的坐标识别方法

#### 自定义坐标识别器
```python
# custom_coordinate_detector.py
from mobile_v3.utils.coordinate_detector import CoordinateDetector

class CustomCoordinateDetector(CoordinateDetector):
    """自定义坐标识别器"""
    
    def __init__(self, model_path=None):
        super().__init__()
        self.model_path = model_path
        self.load_model()
    
    def load_model(self):
        """加载坐标识别模型"""
        if self.model_path:
            # 加载自定义模型
            pass
    
    def detect_coordinates(self, image, instruction):
        """检测坐标"""
        try:
            # 预处理图像
            processed_image = self.preprocess_image(image)
            
            # 执行坐标检测
            coordinates = self.run_detection(processed_image, instruction)
            
            # 后处理结果
            return self.postprocess_coordinates(coordinates)
            
        except Exception as e:
            print(f"坐标检测失败: {e}")
            return None
    
    def preprocess_image(self, image):
        """图像预处理"""
        # 实现图像预处理逻辑
        # 例如：调整大小、归一化等
        return image
    
    def run_detection(self, image, instruction):
        """运行坐标检测"""
        # 实现具体的检测算法
        # 可以使用深度学习模型、传统CV方法等
        pass
    
    def postprocess_coordinates(self, coordinates):
        """坐标后处理"""
        # 验证坐标有效性
        # 应用坐标变换等
        return coordinates

# 注册自定义坐标检测器
def register_custom_detector():
    """注册自定义坐标检测器"""
    from mobile_v3.utils import coordinate_registry
    
    coordinate_registry.register('custom', CustomCoordinateDetector)
    
    print("自定义坐标检测器已注册")
```

## 自定义GUI-Critic评价标准

### 1. 扩展评价维度

#### 自定义评价器
```python
# custom_gui_critic.py
from GUI_Critic_R1.critic_base import CriticBase

class CustomGUICritic(CriticBase):
    """自定义GUI评价器"""
    
    def __init__(self, evaluation_criteria=None):
        super().__init__()
        self.evaluation_criteria = evaluation_criteria or self.default_criteria()
    
    def default_criteria(self):
        """默认评价标准"""
        return {
            'safety': {
                'weight': 0.4,
                'description': '操作安全性'
            },
            'efficiency': {
                'weight': 0.3,
                'description': '操作效率'
            },
            'user_experience': {
                'weight': 0.2,
                'description': '用户体验'
            },
            'accessibility': {
                'weight': 0.1,
                'description': '可访问性'
            }
        }
    
    def evaluate_action(self, image, action, context=None):
        """评价GUI操作"""
        scores = {}
        
        # 评价各个维度
        for criterion, config in self.evaluation_criteria.items():
            score = self.evaluate_criterion(image, action, criterion, context)
            scores[criterion] = {
                'score': score,
                'weight': config['weight']
            }
        
        # 计算综合得分
        total_score = sum(
            scores[criterion]['score'] * scores[criterion]['weight']
            for criterion in scores
        )
        
        return {
            'total_score': total_score,
            'dimension_scores': scores,
            'recommendation': self.generate_recommendation(scores)
        }
    
    def evaluate_criterion(self, image, action, criterion, context):
        """评价单个标准"""
        if criterion == 'safety':
            return self.evaluate_safety(image, action, context)
        elif criterion == 'efficiency':
            return self.evaluate_efficiency(image, action, context)
        elif criterion == 'user_experience':
            return self.evaluate_ux(image, action, context)
        elif criterion == 'accessibility':
            return self.evaluate_accessibility(image, action, context)
        else:
            return 0.5  # 默认中性得分
    
    def evaluate_safety(self, image, action, context):
        """评价安全性"""
        # 检查是否为危险操作
        dangerous_actions = ['删除', '格式化', '重置', '卸载']
        
        if any(danger in action for danger in dangerous_actions):
            return 0.2  # 低安全性
        
        # 检查是否需要权限
        if '权限' in action or '授权' in action:
            return 0.4  # 中等安全性
        
        return 0.8  # 高安全性
    
    def evaluate_efficiency(self, image, action, context):
        """评价效率"""
        # 检查是否为直接操作
        if '点击' in action and '按钮' in action:
            return 0.9  # 高效率
        
        # 检查是否需要多步操作
        if '滑动' in action or '长按' in action:
            return 0.6  # 中等效率
        
        return 0.7  # 默认效率
    
    def generate_recommendation(self, scores):
        """生成建议"""
        low_scores = [
            criterion for criterion, data in scores.items()
            if data['score'] < 0.5
        ]
        
        if low_scores:
            return f"建议改进: {', '.join(low_scores)}"
        else:
            return "操作评价良好"

# 使用示例
critic = CustomGUICritic()
result = critic.evaluate_action(
    image="screenshot.jpg",
    action="点击删除按钮",
    context={"app": "文件管理器", "file_type": "重要文档"}
)
print(f"评价结果: {result}")
```

### 2. 集成业务规则

#### 业务规则引擎
```python
# business_rules_engine.py
class BusinessRulesEngine:
    """业务规则引擎"""
    
    def __init__(self):
        self.rules = []
    
    def add_rule(self, rule):
        """添加业务规则"""
        self.rules.append(rule)
    
    def evaluate(self, action, context):
        """评估业务规则"""
        violations = []
        
        for rule in self.rules:
            if not rule.check(action, context):
                violations.append(rule.description)
        
        return {
            'compliant': len(violations) == 0,
            'violations': violations
        }

class BusinessRule:
    """业务规则基类"""
    
    def __init__(self, description):
        self.description = description
    
    def check(self, action, context):
        """检查规则是否满足"""
        raise NotImplementedError

class PaymentRule(BusinessRule):
    """支付相关规则"""
    
    def __init__(self):
        super().__init__("支付操作需要二次确认")
    
    def check(self, action, context):
        if '支付' in action or '付款' in action:
            return context.get('confirmed', False)
        return True

class DataDeletionRule(BusinessRule):
    """数据删除规则"""
    
    def __init__(self):
        super().__init__("重要数据删除需要管理员权限")
    
    def check(self, action, context):
        if '删除' in action and context.get('data_importance') == 'high':
            return context.get('admin_permission', False)
        return True

# 使用示例
engine = BusinessRulesEngine()
engine.add_rule(PaymentRule())
engine.add_rule(DataDeletionRule())

result = engine.evaluate(
    action="点击支付按钮",
    context={"confirmed": False}
)
print(f"规则检查结果: {result}")
```

## 性能优化定制

### 1. 自定义缓存策略

#### 智能缓存管理器
```python
# cache_manager.py
import time
import hashlib
from typing import Dict, Any, Optional

class IntelligentCacheManager:
    """智能缓存管理器"""
    
    def __init__(self, max_size=1000, ttl=3600):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.max_size = max_size
        self.ttl = ttl  # 生存时间（秒）
    
    def get_cache_key(self, image_data, instruction):
        """生成缓存键"""
        content = f"{instruction}_{len(image_data)}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(self, image_data, instruction) -> Optional[Any]:
        """获取缓存结果"""
        key = self.get_cache_key(image_data, instruction)
        
        if key in self.cache:
            entry = self.cache[key]
            
            # 检查是否过期
            if time.time() - entry['timestamp'] < self.ttl:
                entry['hits'] += 1
                return entry['result']
            else:
                # 删除过期条目
                del self.cache[key]
        
        return None
    
    def set(self, image_data, instruction, result):
        """设置缓存"""
        key = self.get_cache_key(image_data, instruction)
        
        # 如果缓存已满，删除最少使用的条目
        if len(self.cache) >= self.max_size:
            self.evict_lru()
        
        self.cache[key] = {
            'result': result,
            'timestamp': time.time(),
            'hits': 0
        }
    
    def evict_lru(self):
        """删除最少使用的条目"""
        if not self.cache:
            return
        
        # 找到命中次数最少的条目
        lru_key = min(self.cache.keys(), 
                     key=lambda k: self.cache[k]['hits'])
        del self.cache[lru_key]
    
    def clear(self):
        """清空缓存"""
        self.cache.clear()
    
    def stats(self):
        """获取缓存统计"""
        total_hits = sum(entry['hits'] for entry in self.cache.values())
        return {
            'size': len(self.cache),
            'total_hits': total_hits,
            'avg_hits': total_hits / len(self.cache) if self.cache else 0
        }

# 集成到推理流程
class CachedInferenceWrapper:
    """带缓存的推理包装器"""
    
    def __init__(self, base_wrapper, cache_manager=None):
        self.base_wrapper = base_wrapper
        self.cache_manager = cache_manager or IntelligentCacheManager()
    
    def predict_mm(self, prompt, images=None):
        """带缓存的多模态预测"""
        # 尝试从缓存获取结果
        if images:
            cached_result = self.cache_manager.get(images[0], prompt)
            if cached_result:
                return cached_result
        
        # 执行实际推理
        result = self.base_wrapper.predict_mm(prompt, images)
        
        # 缓存结果
        if images and result:
            self.cache_manager.set(images[0], prompt, result)
        
        return result
```

### 2. 自定义批处理策略

#### 动态批处理管理器
```python
# dynamic_batch_manager.py
import asyncio
import time
from typing import List, Tuple, Any

class DynamicBatchManager:
    """动态批处理管理器"""
    
    def __init__(self, max_batch_size=8, max_wait_time=0.1):
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.pending_requests = []
        self.processing = False
    
    async def add_request(self, request_data) -> Any:
        """添加请求到批处理队列"""
        future = asyncio.Future()
        
        self.pending_requests.append({
            'data': request_data,
            'future': future,
            'timestamp': time.time()
        })
        
        # 如果不在处理中，启动批处理
        if not self.processing:
            asyncio.create_task(self.process_batch())
        
        return await future
    
    async def process_batch(self):
        """处理批次"""
        self.processing = True
        
        try:
            while self.pending_requests:
                # 等待更多请求或超时
                await asyncio.sleep(self.max_wait_time)
                
                # 获取当前批次
                batch_size = min(len(self.pending_requests), self.max_batch_size)
                current_batch = self.pending_requests[:batch_size]
                self.pending_requests = self.pending_requests[batch_size:]
                
                if current_batch:
                    await self.execute_batch(current_batch)
        
        finally:
            self.processing = False
    
    async def execute_batch(self, batch):
        """执行批次推理"""
        try:
            # 准备批次数据
            batch_data = [item['data'] for item in batch]
            
            # 执行批量推理（这里需要实现具体的批量推理逻辑）
            results = await self.batch_inference(batch_data)
            
            # 返回结果给对应的Future
            for item, result in zip(batch, results):
                item['future'].set_result(result)
        
        except Exception as e:
            # 如果批处理失败，设置异常
            for item in batch:
                item['future'].set_exception(e)
    
    async def batch_inference(self, batch_data):
        """批量推理实现"""
        # 这里实现具体的批量推理逻辑
        # 可以调用VLLM的批量API或其他批处理方法
        results = []
        for data in batch_data:
            # 模拟推理过程
            await asyncio.sleep(0.1)
            results.append(f"Result for {data}")
        
        return results

# 使用示例
async def main():
    batch_manager = DynamicBatchManager(max_batch_size=4, max_wait_time=0.2)
    
    # 并发发送多个请求
    tasks = [
        batch_manager.add_request(f"Request {i}")
        for i in range(10)
    ]
    
    results = await asyncio.gather(*tasks)
    print(f"批处理结果: {results}")

if __name__ == "__main__":
    asyncio.run(main())
```

这个定制化指南提供了扩展Mobile-Agent-v3系统各个组件的详细方法，包括任务类型、模型集成、评价标准和性能优化等方面的定制化方案。