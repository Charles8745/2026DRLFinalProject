# 系統架構規範 (System Architecture Specification)

**Document Version**: 1.0.0  
**Last Updated**: 2026-04-22  
**Status**: 規劃完成

---

## 架構概觀

```
┌─────────────────────────────────────────────────────────────┐
│                      User Application                        │
└──────────────────┬──────────────────────────────────────────┘
                   │
┌──────────────────┴──────────────────────────────────────────┐
│                    High-Level APIs                          │
│  ┌──────────────┬──────────────┬──────────────────────────┐ │
│  │ Model API    │ Training API │ Inference API            │ │
│  └──────────────┴──────────────┴──────────────────────────┘ │
└──────────────────┬──────────────────────────────────────────┘
                   │
┌──────────────────┴──────────────────────────────────────────┐
│                Core Modules                                 │
│  ┌──────────────┬──────────────┬──────────────────────────┐ │
│  │ Models       │ RL           │ Optimization             │ │
│  │ • Base LLM   │ • PPO        │ • Quantization          │ │
│  │ • LoRA       │ • DQN        │ • KV Cache              │ │
│  │ • Adapter    │ • Policy Net │ • Mixed Precision       │ │
│  └──────────────┴──────────────┴──────────────────────────┘ │
└──────────────────┬──────────────────────────────────────────┘
                   │
┌──────────────────┴──────────────────────────────────────────┐
│                Utility Modules                              │
│  ┌──────────────┬──────────────┬──────────────────────────┐ │
│  │ Data Utils   │ Metrics      │ Logger & Monitor        │ │
│  │ • Loader     │ • BLEU       │ • TensorBoard           │ │
│  │ • Preprocess │ • ROUGE      │ • Weights & Biases      │ │
│  │ • Batch      │ • Perplexity │ • Performance Profile   │ │
│  └──────────────┴──────────────┴──────────────────────────┘ │
└──────────────────┬──────────────────────────────────────────┘
                   │
┌──────────────────┴──────────────────────────────────────────┐
│           External Dependencies                             │
│  PyTorch, Transformers, PEFT, Stable-Baselines3            │
└─────────────────────────────────────────────────────────────┘
```

---

## 模塊設計

### 1. Models Module (`src/models/`)

#### 1.1 Base Model
```python
class BaseModel:
    """Base class for all models"""
    - load_model(model_name)
    - get_config()
    - forward(input_ids, attention_mask)
    - save_model(path)
```

#### 1.2 LoRA Model
```python
class LoRAModel(BaseModel):
    """LLM with LoRA adaptation"""
    - __init__(base_model, lora_rank, lora_alpha)
    - apply_lora()
    - merge_lora()
    - freeze_base()
    - get_trainable_params()
    - forward(input_ids, attention_mask)
```

#### 1.3 Optimized Model
```python
class OptimizedModel(BaseModel):
    """Model with optimization techniques"""
    - quantize(bits: int)
    - apply_kv_cache_optimization()
    - enable_mixed_precision()
    - get_model_size()
    - forward(input_ids, attention_mask)
```

---

### 2. RL Module (`src/rl/`)

#### 2.1 Policy Network
```python
class PolicyNetwork(nn.Module):
    """RL Policy Network for inference optimization"""
    - __init__(input_dim, hidden_dim, output_dim)
    - forward(state) -> action
    - save_checkpoint(path)
    - load_checkpoint(path)
```

#### 2.2 Value Network
```python
class ValueNetwork(nn.Module):
    """Value Network for PPO"""
    - __init__(input_dim, hidden_dim)
    - forward(state) -> value
```

#### 2.3 PPO Trainer
```python
class PPOTrainer:
    """PPO algorithm trainer"""
    - __init__(policy, value_network, learning_rate, device)
    - compute_advantages(rewards, values, gamma, gae_lambda)
    - train_step(states, actions, old_probs, advantages, returns)
    - train_epoch(dataloader, num_epochs)
    - save_checkpoint(path)
```

#### 2.4 DQN Trainer
```python
class DQNTrainer:
    """DQN algorithm trainer"""
    - __init__(q_network, target_network, learning_rate, device)
    - compute_td_target(rewards, next_q_values, dones, gamma)
    - train_step(states, actions, rewards, next_states, dones)
    - update_target_network()
    - save_checkpoint(path)
```

---

### 3. Optimization Module (`src/optimization/`)

#### 3.1 Quantizer
```python
class Quantizer:
    """Model quantization"""
    - quantize_model(model, bits: int) -> QuantizedModel
    - calibrate(dataloader)
    - dequantize()
```

#### 3.2 KV Cache Optimizer
```python
class KVCacheOptimizer:
    """KV cache optimization for inference"""
    - enable_kv_cache()
    - set_cache_size(size: int)
    - clear_cache()
```

#### 3.3 Mixed Precision Manager
```python
class MixedPrecisionManager:
    """Mixed precision inference"""
    - enable_mixed_precision(precision: str)
    - forward_with_precision(model, inputs) -> outputs
```

---

### 4. Utils Module (`src/utils/`)

#### 4.1 Data Utilities
```python
class DataLoader:
    """Data loading and preprocessing"""
    - load_dataset(path)
    - preprocess_data(texts)
    - create_batches(data, batch_size)

class Preprocessor:
    """Text preprocessing"""
    - tokenize(texts, tokenizer)
    - normalize(texts)
    - truncate(texts, max_length)
```

#### 4.2 Evaluation Metrics
```python
class MetricsCalculator:
    """Compute evaluation metrics"""
    - compute_bleu(predictions, references)
    - compute_rouge(predictions, references)
    - compute_perplexity(logits, targets)
    - compute_f1(predictions, targets)
```

#### 4.3 Logger & Monitor
```python
class Logger:
    """Training and inference logging"""
    - log_metric(name, value, step)
    - log_config(config)
    - save_logs(path)

class PerformanceMonitor:
    """Monitor system performance"""
    - track_latency(func)
    - track_memory(func)
    - get_profiling_report()
```

---

## 數據流設計

### Training Flow
```
1. Load Dataset
   └─> Preprocessing (tokenization, normalization)
       └─> Create Data Batches
           └─> Forward Pass
               ├─> Base LLM
               ├─> LoRA Adapter
               └─> Output Logits
                   └─> Compute Loss
                       └─> RL Policy
                           └─> Compute Reward
                               └─> Backward Pass
                                   └─> Update Parameters
                                       └─> Checkpoint Save
                                           └─> Metrics Log
```

### Inference Flow
```
1. Input Text
   └─> Tokenization
       └─> Attention Mask
           └─> Forward Pass
               ├─> Base LLM
               ├─> LoRA Weights
               └─> Output Logits
                   └─> RL Policy Decision
                       └─> Optimization
                           ├─> Quantization (if enabled)
                           ├─> KV Cache (if enabled)
                           └─> Mixed Precision (if enabled)
                               └─> Decoding Strategy
                                   └─> Output Generation
                                       └─> Detokenization
                                           └─> Final Output
```

---

## API 設計

### 1. Model API
```python
# 加載模型
model = load_model('llama-2-7b')
model_with_lora = apply_lora(model, rank=8, alpha=16)

# 保存模型
save_model(model_with_lora, 'checkpoints/model.pt')

# 推理
outputs = model.forward(input_ids, attention_mask)
```

### 2. Training API
```python
# 初始化訓練器
trainer = PPOTrainer(
    model=model,
    learning_rate=1e-4,
    device='cuda'
)

# 訓練
trainer.train(
    train_dataloader=train_loader,
    num_epochs=10,
    save_interval=500
)
```

### 3. Inference API
```python
# 創建推理引擎
engine = InferenceEngine(
    model=model,
    quantize=True,
    bits=8,
    enable_kv_cache=True
)

# 推理
response = engine.generate(
    prompt='Hello world',
    max_length=100,
    temperature=0.7
)
```

### 4. Evaluation API
```python
# 計算指標
metrics = evaluate_model(
    model=model,
    test_dataloader=test_loader,
    metrics=['bleu', 'rouge', 'perplexity']
)

# 性能分析
profile = profile_model(
    model=model,
    input_shape=(1, 256)
)
```

---

## 文件組織

```
src/
├── __init__.py
│
├── models/
│   ├── __init__.py
│   ├── base_model.py         # BaseModel class
│   ├── lora_model.py         # LoRA implementation
│   ├── optimized_model.py    # Optimization wrappers
│   └── utils.py              # Model utilities
│
├── rl/
│   ├── __init__.py
│   ├── policy.py             # Policy network
│   ├── value_net.py          # Value network
│   ├── ppo_trainer.py        # PPO algorithm
│   ├── dqn_trainer.py        # DQN algorithm
│   ├── env.py                # RL environment
│   └── utils.py              # RL utilities
│
├── optimization/
│   ├── __init__.py
│   ├── quantizer.py          # Model quantization
│   ├── kv_cache.py           # KV cache optimization
│   ├── mixed_precision.py    # Mixed precision
│   └── utils.py              # Optimization utilities
│
└── utils/
    ├── __init__.py
    ├── data_utils.py         # Data loading & preprocessing
    ├── metrics.py            # Evaluation metrics
    ├── logger.py             # Logging utilities
    ├── monitor.py            # Performance monitoring
    ├── config.py             # Configuration management
    └── constants.py          # Constants & defaults
```

---

## 配置管理

### Config File (`configs/default.yaml`)
```yaml
model:
  name: "llama-2-7b"
  device: "cuda"
  dtype: "float32"

lora:
  enabled: true
  rank: 8
  alpha: 16
  target_modules: ["q_proj", "v_proj"]

training:
  batch_size: 32
  learning_rate: 1e-4
  num_epochs: 10
  warmup_steps: 500
  save_interval: 500

rl:
  algorithm: "ppo"
  learning_rate: 5e-5
  num_rollout_steps: 2048
  entropy_coeff: 0.01

optimization:
  quantize:
    enabled: false
    bits: 8
  kv_cache: true
  mixed_precision: false

evaluation:
  metrics: ["bleu", "rouge", "perplexity"]
  eval_steps: 500
  eval_batch_size: 32
```

---

## 異常處理與驗證

### Error Handling
```python
# Type checking & validation
def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    if not isinstance(input_ids, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(input_ids)}")
    
    if input_ids.dim() != 2:
        raise ValueError(f"Expected 2D tensor, got shape {input_ids.shape}")
    
    return self._forward_impl(input_ids, attention_mask)
```

### Assertions & Checks
```python
assert model is not None, "Model cannot be None"
assert learning_rate > 0, "Learning rate must be positive"
assert 0 <= gamma <= 1, "Discount factor must be in [0, 1]"
```

---

## 性能優化策略

### 1. 計算優化
- [ ] 使用 Flash Attention
- [ ] 實現 Gradient Checkpointing
- [ ] 應用 Gradient Accumulation
- [ ] 採用 Distributed Training

### 2. 記憶體優化
- [ ] LoRA 降低參數量
- [ ] 量化減少模型大小
- [ ] KV Cache 優化
- [ ] 激活函數重計算

### 3. I/O 優化
- [ ] 預加載數據
- [ ] 非同步數據加載
- [ ] Pin Memory
- [ ] 多進程 DataLoader

---

## 測試策略

### Unit Tests
- 模型前向傳播
- LoRA 層功能
- 優化模塊
- 工具函數

### Integration Tests
- 訓練管道
- 推理管道
- 性能基準

### End-to-End Tests
- 完整工作流
- 模型保存與加載
- 多設備支持

---

## 版本控制與CI/CD

### Git Workflow
```
main (stable)
  ↑
develop (integration)
  ↑
feature/*, bugfix/* (development)
```

### Automated Tests
- pytest on push
- Coverage reporting
- Code quality checks (black, flake8, isort)

---

**Last Updated**: 2026-04-22  
**Next Review**: After Phase 2
