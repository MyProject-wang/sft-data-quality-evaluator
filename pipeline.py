import os
import glob
import json
import logging
from typing import Optional, Dict, Any, List
import pandas as pd
import matplotlib.pyplot as plt
from config import Config
from scorer import QualityScorer
from dedup import Deduplicator
from cluster import BadcaseClusterer

logger = logging.getLogger(__name__)

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class SFTQualityPipeline:
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.scorer = QualityScorer(self.config)
        self.deduplicator = Deduplicator(self.config)
        self.clusterer = BadcaseClusterer(self.config)

    def load_real_data_from_dir(self, data_dir: str, n_samples: Optional[int] = None) -> pd.DataFrame:
        files = glob.glob(os.path.join(data_dir, "**", "*.json"), recursive=True) + \
                glob.glob(os.path.join(data_dir, "**", "*.jsonl"), recursive=True)
        if not files:
            raise FileNotFoundError(f"在 {data_dir} 下未找到 .json 或 .jsonl 文件")

        logger.info(f"找到 {len(files)} 个数据文件，开始加载...")
        data = []
        for file_path in files:
            try:
                if file_path.endswith(".jsonl"):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            item = json.loads(line)
                            data.append(item)
                else:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = json.load(f)
                        if isinstance(content, list):
                            data.extend(content)
                        else:
                            data.append(content)
            except Exception as e:
                logger.warning(f"读取文件 {file_path} 出错: {e}")

        def map_field(item: dict, candidates: List[str]) -> Any:
            for cand in candidates:
                if cand in item:
                    return item[cand]
            return ""

        mapped_data = []
        for item in data:
            instruction = map_field(item, ["instruction", "question", "prompt", "query", "text"])
            input_text = map_field(item, ["input", "context"])
            output = map_field(item, ["output", "response", "answer", "completion", "target"])
            task_type = map_field(item, ["task_type", "task", "type"])
            if not output:
                continue
            mapped_data.append({
                "instruction": instruction,
                "input": input_text,
                "output": output,
                "task_type": task_type if task_type else "未知"
            })

        df = pd.DataFrame(mapped_data)
        if n_samples and n_samples < len(df):
            df = df.sample(n=n_samples, random_state=self.config.random_seed)
            logger.info(f"随机采样 {n_samples} 条数据")
        logger.info(f"从真实数据加载了 {len(df)} 条有效样本")
        return df

    def _print_verbose_summary(self, df_scored: pd.DataFrame):
        print("\n" + "="*60)
        print("详细统计信息 (verbose=True)")
        print("="*60)

        print("\n=== 各任务类型平均分 ===")
        if 'task_type' in df_scored.columns:
            type_means = df_scored.groupby('task_type')['total_score'].mean().sort_values()
            for task, score in type_means.items():
                print(f"  {task}: {score:.2f}")
        else:
            print("  无 task_type 字段")

        low = df_scored[df_scored['total_score'] < self.config.low_score_threshold]
        print(f"\n=== 低质量样本（总分<{self.config.low_score_threshold}）===")
        print(f"数量: {len(low)} ({len(low)/len(df_scored)*100:.1f}%)")
        if len(low) >= 3:
            clustered = self.clusterer.cluster(df_scored, verbose=False)
            if clustered is not None:
                for i in range(clustered['cluster'].nunique()):
                    cluster_data = clustered[clustered['cluster'] == i]
                    print(f"\n聚类{i} (数量{len(cluster_data)}):")
                    if len(cluster_data) > 0:
                        typical = cluster_data['instruction'].iloc[0]
                        print(f"  典型指令: {typical[:80]}")
                        avg_score = cluster_data['total_score'].mean()
                        low_comp = (cluster_data['completeness'] < 2).mean()
                        low_follow = (cluster_data['instruction_following'] < 2).mean()
                        low_flu = (cluster_data['fluency'] < 2).mean()
                        print(f"  平均质量分: {avg_score:.1f}")
                        print(f"  主要问题: 完整性不足{low_comp:.0%} 指令遵循不足{low_follow:.0%} 流畅性差{low_flu:.0%}")

        high = df_scored[df_scored['total_score'] >= self.config.high_score_threshold]
        print(f"\n=== 高分样本示例（前3条）===")
        for idx, row in high.head(3).iterrows():
            print(f"\n指令: {row['instruction'][:60]}...")
            print(f"输出: {row['output'][:100]}...")
            print(f"总分: {row['total_score']} (完整性:{row['completeness']}, 遵循:{row['instruction_following']}, 流畅:{row['fluency']})")

        print("\n=== 分数分布 ===")
        print(df_scored['total_score'].value_counts().sort_index())

        plt.figure(figsize=(8,4))
        plt.hist(df_scored['total_score'], bins=range(0, 12), color='skyblue', edgecolor='black')
        plt.xlabel('质量总分 (0-10)')
        plt.ylabel('样本数')
        plt.title('SFT数据质量分布')
        plt.show()
        print("="*60)

    def run(self, data_dir: str, verbose: bool = False, n_samples: Optional[int] = None) -> Dict[str, Any]:
        logger.info("=== SFT质量评估流水线启动 ===")

        if n_samples is not None:
            self.config.n_samples = n_samples

        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"数据目录不存在: {data_dir}")

        df_raw = self.load_real_data_from_dir(data_dir, self.config.n_samples)

        if df_raw.empty:
            raise ValueError("加载的数据为空，请检查数据格式")

        if self.config.exact_dedup:
            df_raw = self.deduplicator.exact_deduplication(df_raw)
        if self.config.near_dedup:
            df_raw = self.deduplicator.near_deduplication(df_raw)

        df_scored = self.scorer.evaluate_dataset(df_raw)

        low_clusters = self.clusterer.cluster(df_scored, verbose=verbose)

        os.makedirs(self.config.output_dir, exist_ok=True)
        high = df_scored[df_scored['total_score'] >= self.config.high_score_threshold].copy()
        low = df_scored[df_scored['total_score'] < self.config.low_score_threshold].copy()

        logger.info(f"高质量数据数量: {len(high)} / {len(df_scored)} ({len(high)/len(df_scored)*100:.1f}%)")
        if len(high) > 0:
            high_path = os.path.join(self.config.output_dir, self.config.high_quality_file)
            high[['instruction', 'input', 'output', 'task_type', 'total_score']].to_json(
                high_path, orient='records', force_ascii=False, indent=2
            )
            logger.info(f"高质量数据已保存至 {high_path}")
        if len(low) > 0:
            low_path = os.path.join(self.config.output_dir, self.config.low_quality_file)
            low[['instruction', 'output', 'total_score']].to_csv(low_path, index=False, encoding='utf-8-sig')
            logger.info(f"低质量数据已保存至 {low_path}")

        if low_clusters is not None:
            report_path = os.path.join(self.config.output_dir, self.config.cluster_report_file)
            self.clusterer.generate_cluster_report(low_clusters, report_path)

        if verbose:
            self._print_verbose_summary(df_scored)

        logger.info("=== 流水线执行完毕 ===")
        return {
            "raw_data": df_raw,
            "scored_data": df_scored,
            "low_clusters": low_clusters,
            "high_quality": high
        }