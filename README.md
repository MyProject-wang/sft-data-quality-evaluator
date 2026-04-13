```markdown
# SFT 数据质量评估与清洗工具

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

对 SFT（Supervised Fine-Tuning）训练数据进行自动化质量评估与清洗。采用 **规则引擎 + 大模型混合评分**，支持 SimHash 近似去重、低质量样本聚类分析，帮助快速筛选高质量训练数据。

## ✨ 核心功能

- **多维度评分**：完整性（0-3）、指令遵循（0-3）、流畅性（0-2）、安全性（0-2），总分 0-10
- **混合评分策略**：规则评分（40%）+ DeepSeek 大模型评分（60%），兼顾效率与语义理解
- **智能去重**：精确去重（MD5）+ SimHash 近似去重（汉明距离阈值可配）
- **低质量样本聚类**：TF-IDF + KMeans 聚类，自动生成可视化 HTML 报告
- **字段自适应**：自动识别 JSON/JSONL 中的常见字段（`instruction`/`question`/`prompt` 等）
- **模块化设计**：配置与逻辑分离，易于扩展和二次开发

## 🚀 快速开始

### 1. 克隆仓库
```bash
git clone https://github.com/wdp-data/sft-data-quality-evaluator.git
cd sft-data-quality-evaluator
```

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

### 3. 准备数据
将 JSON/JSONL 格式的 SFT 数据放入 `./SFT数据集/` 目录（或使用内置 `sample_data/` 样例）。  
数据需包含指令与输出字段，支持常见命名自动映射。

### 4. 运行评估
打开 `run.ipynb` 并依次执行 Cell，或直接在 Python 中调用：

```python
from pipeline import SFTQualityPipeline
from config import Config

config = Config(
    n_samples=500,
    enable_llm_scoring=False,   # 设为 True 启用混合评分
    high_score_threshold=8,
    low_score_threshold=6,
    output_dir="output_sft"
)

pipeline = SFTQualityPipeline(config)
result = pipeline.run(data_dir="./SFT数据集", verbose=True)
```

### 5. 查看结果
- 高质量数据：`output_sft/high_quality_sft.json`
- 低质量数据：`output_sft/low_quality_sft.csv`
- 聚类报告：`output_sft/cluster_report.html`

## ⚙️ 配置说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `n_samples` | `int` | `None` | 采样数量，`None` 为全量 |
| `high_score_threshold` | `int` | `8` | 高质量分数阈值 |
| `low_score_threshold` | `int` | `6` | 低质量分数阈值 |
| `enable_llm_scoring` | `bool` | `False` | 是否启用 LLM 混合评分 |
| `llm_sample_ratio` | `float` | `0.1` | LLM 评分采样比例（控制成本） |
| `llm_score_lower` | `int` | `5` | 触发 LLM 评分的规则总分下限 |
| `llm_score_upper` | `int` | `8` | 触发 LLM 评分的规则总分上限 |
| `llm_rule_weight` | `float` | `0.4` | 融合时规则评分权重 |
| `llm_model_weight` | `float` | `0.6` | 融合时 LLM 评分权重 |
| `llm_api_key` | `str` | `None` | DeepSeek API Key（也可设环境变量 `DEEPSEEK_API_KEY`） |

## 📁 项目结构

```
.
├── config.py               # 配置类
├── simhash.py              # SimHash 算法实现
├── scorer.py               # 评分器（规则 + LLM）
├── dedup.py                # 去重器（精确 + 近似）
├── cluster.py              # 低质量样本聚类器
├── pipeline.py             # 主流水线
├── run.ipynb               # 演示入口 Notebook
├── sample_data/            # 内置样例数据
│   └── sample_sft.jsonl
├── requirements.txt        # 依赖清单
├── .gitignore
├── LICENSE
└── README.md
```

## 📊 效果示例

在 500 条真实 SFT 数据上的运行结果：

- **精确去重**：500 → 500（去重率 0.0%）
- **近似去重**：500 → 494（去重率 1.2%）
- **高质量样本占比**：49.8%（总分 ≥ 8）
- **低质量样本**：5 条，聚类为 3 类（典型问题：截断、跑题、拒绝回答）

## 🛠 依赖清单

```
pandas
numpy
scikit-learn
matplotlib
tqdm
openai
```

## 📄 License

MIT
```