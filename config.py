from dataclasses import dataclass
from typing import Optional

@dataclass
class Config:
    # 数据
    n_samples: Optional[int] = None
    random_seed: int = 42
    # 去重
    exact_dedup: bool = True
    near_dedup: bool = True
    simhash_threshold: float = 0.95
    # 评分与筛选
    low_score_threshold: int = 6
    high_score_threshold: int = 8
    # 聚类
    n_clusters: int = 3
    tfidf_max_features: int = 100
    # 输出
    output_dir: str = "output_sft"
    high_quality_file: str = "high_quality_sft.json"
    low_quality_file: str = "low_quality_sft.csv"
    cluster_report_file: str = "cluster_report.html"
    # 优化选项
    enable_generic_following: bool = True

    # LLM 混合评分配置
    enable_llm_scoring: bool = False
    llm_api_key: Optional[str] = None
    llm_base_url: str = "https://api.deepseek.com"
    llm_model: str = "deepseek-chat"
    llm_sample_ratio: float = 0.1
    llm_score_lower: int = 5
    llm_score_upper: int = 8
    llm_rule_weight: float = 0.4
    llm_model_weight: float = 0.6