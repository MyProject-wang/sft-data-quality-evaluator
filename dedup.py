import hashlib
import pandas as pd
import logging
from simhash import Simhash
from config import Config

logger = logging.getLogger(__name__)

class Deduplicator:
    def __init__(self, config: Config):
        self.config = config

    def exact_deduplication(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        df['hash'] = df.apply(lambda row: hashlib.md5(
            (str(row['instruction']) + str(row['output'])).encode('utf-8')
        ).hexdigest(), axis=1)
        df = df.drop_duplicates(subset=['hash']).reset_index(drop=True)
        after = len(df)
        logger.info(f"精确去重: {before} -> {after} (去重率 {(1-after/before)*100:.2f}%)")
        df.drop(columns=['hash'], inplace=True)
        return df

    def near_deduplication(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.config.near_dedup:
            return df
        def simhash_func(row):
            text = str(row['instruction']) + " [SEP] " + str(row['output'])
            return Simhash(text) if len(text) > 0 else None

        df['simhash'] = df.apply(simhash_func, axis=1)
        keep = []
        hash_list = []
        max_distance = int(64 * (1 - self.config.simhash_threshold))
        for idx, row in df.iterrows():
            h = row['simhash']
            if h is None:
                keep.append(idx)
                continue
            duplicate = False
            for existing in hash_list:
                if h.distance(existing) <= max_distance:
                    duplicate = True
                    break
            if not duplicate:
                hash_list.append(h)
                keep.append(idx)
        before = len(df)
        df = df.iloc[keep].reset_index(drop=True)
        after = len(df)
        logger.info(f"近似去重: {before} -> {after} (去重率 {(1-after/before)*100:.2f}%)")
        df.drop(columns=['simhash'], inplace=True)
        return df