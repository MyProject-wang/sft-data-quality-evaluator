import pandas as pd
import logging
from typing import Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from config import Config

logger = logging.getLogger(__name__)

class BadcaseClusterer:
    def __init__(self, config: Config):
        self.config = config

    def cluster(self, df: pd.DataFrame, score_threshold: int = None, verbose: bool = False) -> Optional[pd.DataFrame]:
        if score_threshold is None:
            score_threshold = self.config.low_score_threshold
        low = df[df['total_score'] < score_threshold].copy()
        if len(low) < 3:
            if verbose:
                logger.info(f"低质量样本不足3条（实际{len(low)}），跳过聚类")
            return None
        low['instruction'] = low['instruction'].fillna('').astype(str)
        low = low[low['instruction'].str.strip() != '']
        if len(low) < 3:
            if verbose:
                logger.info("有效指令不足3条，跳过聚类")
            return None
        texts = low['instruction'].tolist()
        try:
            vectorizer = TfidfVectorizer(max_features=self.config.tfidf_max_features)
            X = vectorizer.fit_transform(texts)
            n_clusters = min(self.config.n_clusters, len(texts))
            kmeans = KMeans(n_clusters=n_clusters, random_state=self.config.random_seed, n_init=10)
            labels = kmeans.fit_predict(X)
            low['cluster'] = labels
            if verbose:
                logger.info(f"聚类完成，共 {n_clusters} 类")
            return low
        except Exception as e:
            logger.error(f"聚类失败: {e}")
            return None

    def generate_cluster_report(self, df_clustered: pd.DataFrame, output_path: str):
        if df_clustered is None:
            return
        html = """
        <html>
        <head><meta charset="UTF-8"><title>SFT低质量样本聚类报告</title></head>
        <body>
        <h1>低质量样本聚类分析报告</h1>
        """
        for i in range(df_clustered['cluster'].nunique()):
            cluster_data = df_clustered[df_clustered['cluster'] == i]
            html += f"<h2>聚类{i} (数量{len(cluster_data)})</h2>"
            html += "<table border='1'><tr><th>指令</th><th>总分</th><th>完整性</th><th>指令遵循</th><th>流畅性</th></tr>"
            for _, row in cluster_data.head(10).iterrows():
                html += f"<tr><td>{row['instruction'][:100]}</td><td>{row['total_score']}</td><td>{row['completeness']}</td><td>{row['instruction_following']}</td><td>{row['fluency']}</td></tr>"
            html += "</table><br>"
        html += "</body></html>"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        logger.info(f"聚类报告已保存至 {output_path}")