# 🚀 基於 LoRA 與強化學習之 LLM 推理優化

> **2026 深度強化學習期末專題**  
> 一個結合參數高效微調和強化學習的大型語言模型推理優化系統

[![GitHub](https://img.shields.io/badge/GitHub-Charles8745-blue?logo=github)](https://github.com/Charles8745/2026DRLFinalProject)
[![YouTube](https://img.shields.io/badge/YouTube-介紹影片-red?logo=youtube)](https://youtu.be/aoGGqbCDNkg)
[![Status](https://img.shields.io/badge/Status-Active%20Development-brightgreen)]()
[![Version](https://img.shields.io/badge/Version-1.0.0-blue)]()
[![License](https://img.shields.io/badge/License-MIT-green)]()

---

## 📋 項目概述

### 問題陳述

![研究動機 - LLM 推理能力的缺失](./assets/images/research_motivation.svg)

**核心問題**：大型語言模型 (LLM) 在生成任務中表現出色，但其推理過程存在以下挑戰：
- ⚠️ **缺乏真正的推理能力**：GSM-Symbolic 研究揭露，模型只是進行「背誦模式匹配」，而非 System 2 思考
- ⚠️ **計算複雜度高**：推理過程計算量大、延遲高、顯存占用多
- ⚠️ **部署困難**：限制了 LLM 在實時應用和邊緣設備上的部署
- ⚠️ **可靠性問題**：對變數敏感，容易受輸入變化影響

### 核心創新
本專題結合以下技術：
- **LoRA (Low-Rank Adaptation)**: 實現參數高效的模型微調
- **強化學習 (PPO/DQN)**: 優化推理過程和資源分配
- **多維度優化**: 量化、KV 緩存、混合精度推理

### 預期成果
| 指標 | 目標 |
|------|------|
| 推理加速 | **2-5 倍** 速度提升 |
| 記憶體節省 | **40-60%** 顯存降低 |
| 精度保留 | **>95%** 準確度保持 |

---

## ✨ 主要特性

### 🎯 核心功能
- ✅ **LoRA 集成**: 低秩適應實現高效微調，減少 99% 的可訓練參數
- ✅ **強化學習優化**: PPO/DQN 算法優化推理策略和資源分配
- ✅ **多優化技術**: 量化、KV 緩存、混合精度推理
- ✅ **完整評估框架**: BLEU、ROUGE、Perplexity 等多維度評估
- ✅ **性能監控**: 實時性能監控和可視化
- ✅ **易用 API**: 簡潔的 Python API，開箱即用

### 🔧 技術支持
- 🐍 **Python 3.10+**
- 🔥 **PyTorch 2.0+**
- 🤗 **Hugging Face Transformers**
- 🎯 **PEFT (Parameter-Efficient Fine-Tuning)**
- 🎮 **Stable-Baselines3** (強化學習)

---

## 🏗️ 系統架構

### 系統架構 UML 類圖

```mermaid
classDiagram
    class BaseLLMModel {
        -String model_id
        -Dict config
        -Tensor weights
        -String device
        -int max_seq_length
        +load_model(path)
        +forward(input) Tensor
        +generate(prompt) String
        +get_attention() Tensor
    }

    class LoRAAdapter {
        -int rank
        -float alpha
        -Tensor lora_A
        -Tensor lora_B
        -List target_layers
        +init_lora(model)
        +forward(x) Tensor
        +merge_weights()
        +get_trainable_params() int
    }

    class RLPolicyNetwork {
        -int state_dim
        -int action_dim
        -int hidden_dim
        -Network policy_net
        -Network value_net
        +get_action(state) Action
        +get_value(state) float
        +update_policy(loss)
        +compute_advantages() Tensor
    }

    class OptimizationEngine {
        -Quantizer quantizer
        -KVCache kv_cache
        -bool mixed_precision
        -Pruner pruner
        -List techniques
        +apply_quantization()
        +optimize_kv_cache()
        +apply_pruning()
        +benchmark() Dict
    }

    class DataProcessor {
        -Tokenizer tokenizer
        -int batch_size
        +load_data(path) Dataset
        +tokenize(texts) Tensor
        +batch_data() DataLoader
    }

    class EvaluationModule {
        -Dict metrics
        -float threshold
        +compute_bleu() float
        +compute_rouge() float
        +evaluate_all() Dict
    }

    class TrainingTrainer {
        -Optimizer optimizer
        -Scheduler lr_scheduler
        +train_epoch(loader)
        +validate(loader) Dict
        +save_checkpoint(path)
    }

    class MonitoringLogger {
        -String log_dir
        -TensorBoard writer
        +log_metrics(metrics)
        +log_model(model)
        +visualize()
    }

    BaseLLMModel "1" --> "1" LoRAAdapter : enhances
    LoRAAdapter "1" --> "1" RLPolicyNetwork : feeds state
    RLPolicyNetwork "1" --> "1" OptimizationEngine : applies strategy
    TrainingTrainer "1" --> "1" BaseLLMModel : trains
    TrainingTrainer "1" --> "1" LoRAAdapter : fine-tunes
    TrainingTrainer "1" --> "1" RLPolicyNetwork : optimizes
    EvaluationModule "1" --> "1" OptimizationEngine : evaluates
    DataProcessor "1" --> "1" TrainingTrainer : supplies data
    MonitoringLogger "1" ..> "1" TrainingTrainer : observes
    MonitoringLogger "1" ..> "1" RLPolicyNetwork : observes
```

**系統設計說明**：

#### 核心層級 (Core Classes)
1. **BaseLLMModel**: 凍結的基礎 LLM 模型，負責基本文本生成和特徵提取
2. **LoRAAdapter**: 低秩適配層，實現 ΔW = B·A^T，減少 99% 可訓練參數
3. **RLPolicyNetwork**: 強化學習策略網絡，使用 PPO/DQN 優化推理決策
4. **OptimizationEngine**: 應用量化、KV 緩存、混合精度等多重優化技術

#### 支持模塊 (Auxiliary Modules)
- **DataProcessor**: 數據預處理和批次化
- **EvaluationModule**: 多指標評估（BLEU、ROUGE 等）
- **TrainingTrainer**: 統一訓練流程管理
- **MonitoringLogger**: 實時性能監控和日誌記錄

---

## �️ 項目結構

```
2026DRLFinalProject/
│
├── 📄 README.md                          # 本文件
├── 📄 OPENSPEC.md                        # 開發計劃 (12 週)
├── 📄 requirements.txt                   # 依賴庫
│
├── 📊 文檔與演示
│   ├── FinalProject_ppt.pdf              # PPT 演示
│   ├── related_work.pdf                  # 相關文獻
│   └── 系統設計規格書.pdf                # 設計規格
│
├── 📁 openspec/                          # OpenSpec 規範
│   ├── config.yaml                       # 配置規範
│   └── specs/
│       ├── project_specification.md      # 項目規範
│       └── architecture_spec.md          # 架構規範
│
├── 📁 src/                               # 源代碼
│   ├── models/                           # 模型定義
│   │   ├── base_model.py
│   │   ├── lora_model.py
│   │   └── optimized_model.py
│   │
│   ├── rl/                               # 強化學習
│   │   ├── policy.py
│   │   ├── value_net.py
│   │   ├── ppo_trainer.py
│   │   └── dqn_trainer.py
│   │
│   ├── optimization/                     # 優化模塊
│   │   ├── quantizer.py
│   │   ├── kv_cache.py
│   │   └── mixed_precision.py
│   │
│   └── utils/                            # 工具函數
│       ├── data_utils.py
│       ├── metrics.py
│       ├── logger.py
│       └── monitor.py
│
├── 📁 data/                              # 數據目錄
│   ├── raw/                              # 原始數據
│   ├── processed/                        # 處理後數據
│   └── splits/                           # 數據分割
│
├── 📁 experiments/                       # 實驗結果
│   ├── checkpoints/                      # 模型檢查點
│   ├── logs/                             # 訓練日誌
│   └── results/                          # 評估結果
│
├── 📁 tests/                             # 測試代碼
│   ├── test_models.py
│   ├── test_rl.py
│   └── test_optimization.py
│
├── 📁 notebooks/                         # Jupyter 筆記本
│   ├── 01_eda.ipynb
│   ├── 02_baseline.ipynb
│   └── 03_analysis.ipynb
│
├── 📁 configs/                           # 配置文件
│   └── default.yaml
│
└── .gitignore
```

---

## 🚀 快速開始

### 1️⃣ 環境設置

#### 克隆倉庫
```bash
git clone https://github.com/Charles8745/2026DRLFinalProject.git
cd 2026DRLFinalProject
```

#### 創建虛擬環境
```bash
# 使用 Python venv
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# 或
venv\Scripts\activate  # Windows
```

#### 安裝依賴
```bash
pip install -r requirements.txt
```

#### 驗證安裝
```bash
python -c "import torch; print(f'PyTorch {torch.__version__}')"
python -c "import transformers; print(f'Transformers {transformers.__version__}')"
```

### 2️⃣ 了解項目

#### 📹 觀看介紹影片
**快速了解項目概況**: [YouTube - 項目介紹影片](https://youtu.be/aoGGqbCDNkg)

#### 查看項目文檔
```bash
# 查看完整開發計劃
cat OPENSPEC.md

# 查看 PPT 演示
open FinalProject_ppt.pdf  # macOS
# 或使用其他 PDF 閱讀器打開
```

#### 瀏覽文檔
| 文件 | 說明 |
|------|------|
| 📹 [介紹影片](https://youtu.be/aoGGqbCDNkg) | YouTube 上的項目介紹和技術方案講解 |
| `FinalProject_ppt.pdf` | 項目完整演示和技術方案 |
| `related_work.pdf` | 相關研究和技術背景 |
| `系統設計規格書.pdf` | 詳細系統設計規範 |
| `openspec/specs/project_specification.md` | 項目完整規範 |
| `openspec/specs/architecture_spec.md` | 架構詳細設計 |

### 3️⃣ 運行示例 (Coming Soon)

```bash
# 訓練模型
python src/train.py --config configs/default.yaml

# 評估模型
python src/evaluate.py --model_path experiments/checkpoints/best_model.pt

# 運行推理
python src/inference.py --text "Your prompt here"

# 運行性能基準測試
python src/benchmark.py --model_path experiments/checkpoints/best_model.pt
```

---

## � 詳細文檔

### 核心概念

#### LoRA (Low-Rank Adaptation)
LoRA 通過在原始模型旁邊添加小的可訓練矩陣（秩為 r），實現高效微調：

```
Original Weight: W ∈ ℝ^(d_out × d_in)
LoRA Update:    ΔW = B A^T  where A ∈ ℝ^(d_in × r), B ∈ ℝ^(d_out × r)
Training Parameters: r × (d_in + d_out)  << d_in × d_out
```

**優勢**:
- 🎯 參數減少 99%
- ⚡ 訓練速度提升
- 💾 顯存占用降低

#### 強化學習優化
使用 PPO (Proximal Policy Optimization) 優化推理過程：

```
Goal: Maximize Reward = Quality - λ × (Latency + Memory)
Policy: π(action|state) → 決策 token pruning、量化策略等
```

#### 多維度優化
- **量化**: INT8/INT4 量化減少模型大小 4-8 倍
- **KV 緩存**: 優化注意力機制的緩存
- **混合精度**: FP32 + FP16 混合推理

## 🛠️ 技術棧架構

### 技術棧 UML 套件圖

```mermaid
graph TD
    subgraph APP["🖥️ Application Layer"]
        CLI["CLI Tools\npython train.py / evaluate.py"]
        API["Python API\nLoRAModel / PPOTrainer / InferenceEngine"]
        CFG["Config Files\nconfigs/default.yaml"]
    end

    subgraph CORE["⚙️ Core Framework"]
        PT["PyTorch >= 2.0.0\n核心計算框架"]
        HF["Transformers >= 4.30.0\nLLM 模型與分詞器"]
        PEFT["PEFT >= 0.4.0\nLoRA 實現"]
        SB3["Stable-Baselines3 >= 2.0.0\nPPO / DQN 算法"]
        GYM["Gymnasium >= 0.28.0\nRL 環境接口"]
    end

    subgraph MOD["🧩 Core Modules"]
        MODELS["models/\nBaseLLMModel\nLoRAAdapter\nOptimizedModel"]
        RL["rl/\nPolicyNetwork\nPPOTrainer\nDQNTrainer"]
        OPT["optimization/\nQuantizer\nKVCache\nMixedPrecision"]
        UTILS["utils/\nDataUtils\nMetrics\nLogger"]
    end

    subgraph EVAL["📊 Evaluation & Monitoring"]
        BLEU["SacreBLEU >= 2.3.0\n機器翻譯評估"]
        NLTK["NLTK / ROUGE\n文本評估"]
        TB["TensorBoard >= 2.13.0\n訓練視覺化"]
        WB["Weights & Biases\n實驗追蹤"]
    end

    subgraph DEV["🔧 Development Tools"]
        TEST["pytest >= 7.4.0\n單元 / 整合測試"]
        FMT["black >= 23.0.0\n代碼格式化"]
        LINT["flake8 >= 6.0.0\n代碼檢查"]
        ISORT["isort >= 5.12.0\nImport 排序"]
    end

    APP --> CORE
    CORE --> MOD
    MOD --> EVAL
    APP -.-> DEV
```

### 完整技術棧表格

| 層級 | 框架/工具 | 版本 | 說明 |
|------|-----------|------|------|
| **深度學習** | PyTorch | >= 2.0.0 | 核心計算框架 |
| **NLP** | Transformers | >= 4.30.0 | LLM 和分詞 |
| **LoRA** | PEFT | >= 0.4.0 | 參數高效微調 |
| **強化學習** | Stable-Baselines3 | >= 2.0.0 | RL 算法實現 |
| **環境** | Gymnasium | >= 0.28.0 | RL 環境接口 |
| **評估** | SacreBLEU | >= 2.3.0 | 機器翻譯評估 |
| **評估** | NLTK | >= 3.8.0 | NLP 評估工具 |
| **監控** | TensorBoard | >= 2.13.0 | 訓練監控 |
| **測試** | pytest | >= 7.4.0 | 單元測試 |
| **代碼質量** | black | >= 23.0.0 | 代碼格式化 |

---

## 🎯 開發進度

### 5 階段開發計劃 (12 週)

![開發階段](./assets/images/development_phases.svg)

| 階段 | 時間 | 任務 | 狀態 |
|------|------|------|------|
| **Phase 1** | Week 1-2 | 環境設置、文獻研究、原型驗證 | 📋 計劃中 |
| **Phase 2** | Week 3-4 | 架構設計、API 規範、數據管道 | 📋 計劃中 |
| **Phase 3** | Week 5-8 | 核心實現、單元測試、集成測試 | 📋 計劃中 |
| **Phase 4** | Week 9-10 | 模型訓練、參數優化、性能調優 | 📋 計劃中 |
| **Phase 5** | Week 11-12 | 評估驗證、文檔完善、開源發布 | 📋 計劃中 |

**詳細計劃**: 查看 [OPENSPEC.md](./OPENSPEC.md)

---

## 📊 評估指標

### 性能對比

![性能對比](./assets/images/performance_comparison.svg)

### 詳細指標

#### 準確度指標
- **BLEU Score**: 機器翻譯質量 (目標 >0.30)
- **ROUGE Score**: 文本摘要質量 (目標 >0.40)
- **Perplexity**: 語言建模困惑度 (目標 <50)
- **F1 Score**: 分類任務準確率 (目標 >0.75)

#### 性能指標
- **推理延遲**: <50ms (目標)
- **吞吐量**: >200 req/s (目標)
- **峰值記憶體**: <8GB (目標)
- **能耗**: <150W (目標)

#### 優化指標
- **加速比**: >3x (目標)
- **記憶體節省**: >50% (目標)
- **精度保留**: >95% (目標)

---

## 🛠️ API 使用示例

### 加載模型
```python
from src.models import LoRAModel

# 初始化 LoRA 模型
model = LoRAModel(
    base_model='llama-2-7b',
    lora_rank=8,
    lora_alpha=16
)
```

### 訓練模型
```python
from src.rl import PPOTrainer

trainer = PPOTrainer(
    model=model,
    learning_rate=1e-4,
    device='cuda'
)

trainer.train(
    train_dataloader=train_loader,
    num_epochs=10
)
```

### 推理
```python
from src.utils import InferenceEngine

engine = InferenceEngine(
    model=model,
    quantize=True,
    bits=8,
    enable_kv_cache=True
)

response = engine.generate(
    prompt='Hello world',
    max_length=100
)
```

---

## � 推薦閱讀

### 核心論文
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

### 相關資源
- [Hugging Face Transformers 文檔](https://huggingface.co/transformers/)
- [PEFT GitHub](https://github.com/huggingface/peft)
- [Stable-Baselines3 文檔](https://stable-baselines3.readthedocs.io/)

---

## 🤝 貢獻指南

### 報告 Issue
發現問題？請提交 Issue，包含：
- 問題描述
- 復現步驟
- 預期結果
- 實際結果

### 提交 Pull Request
歡迎提交 PR：
1. Fork 倉庫
2. 創建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'feat: Add amazing feature'`)
4. Push 到分支 (`git push origin feature/amazing-feature`)
5. 開啟 Pull Request

### 提交規範
遵循 [Conventional Commits](https://www.conventionalcommits.org/zh-hans/):
```
feat:    新功能
fix:     bug 修複
docs:    文檔更新
refactor: 代碼重構
test:    測試代碼
perf:    性能優化
chore:   工具/配置
```

---

## 📄 許可證

本項目採用 **MIT 許可證**。詳見 [LICENSE](./LICENSE) 文件。

---

## 🙋 聯絡方式

- **GitHub**: [@Charles8745](https://github.com/Charles8745)
- **Email**: [your-email@example.com]
- **Issues**: [GitHub Issues](https://github.com/Charles8745/2026DRLFinalProject/issues)

---

## 🎓 致謝

感謝以下項目和社區的支持：
- Hugging Face Transformers
- PEFT 團隊
- Stable-Baselines3 開發者
- 開源社區

---

## 📈 相關項目與工作

### 本項目相關文檔
| 文檔 | 描述 |
|------|------|
| [OPENSPEC.md](./OPENSPEC.md) | 詳細開發計劃和里程碑 |
| [openspec/specs/project_specification.md](./openspec/specs/project_specification.md) | 完整項目規範 |
| [openspec/specs/architecture_spec.md](./openspec/specs/architecture_spec.md) | 系統架構規範 |
| [FinalProject_ppt.pdf](./FinalProject_ppt.pdf) | PPT 演示 |
| [related_work.pdf](./related_work.pdf) | 相關文獻 |
| [系統設計規格書.pdf](./系統設計規格書.pdf) | 設計規格 |

---

<div align="center">

**⭐ 如果這個項目對你有幫助，請給我們一個 Star！**

Made with ❤️ by Charles | 2026

</div>

---

**最後更新**: 2026-04-22  
**下次更新**: 完成 Phase 1 後
