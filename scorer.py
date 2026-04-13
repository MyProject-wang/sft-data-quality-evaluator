import re
import json
import random
import os
import logging
from typing import Any, Tuple, Optional
import pandas as pd
from tqdm import tqdm
from config import Config

logger = logging.getLogger(__name__)

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

class QualityScorer:
    def __init__(self, config: Config):
        self.config = config
        self.llm_client = None

    def _init_llm_client(self):
        if self.llm_client is None and OpenAI is not None:
            api_key = self.config.llm_api_key or os.environ.get("DEEPSEEK_API_KEY")
            if not api_key:
                logger.warning("未找到 LLM API Key，将回退至纯规则评分")
                return
            self.llm_client = OpenAI(api_key=api_key, base_url=self.config.llm_base_url)
            logger.info("LLM 客户端初始化成功")

    def _rule_score(self, instruction: str, output: str, task_type: str) -> Tuple[int, int, int, int, int]:
        out_len = len(output)

        # 1. 完整性 (0-3)
        if out_len == 0:
            completeness = 0
        elif out_len < 10:
            completeness = 1
        elif out_len < 30:
            completeness = 2
        else:
            completeness = 3

        # 2. 指令遵循 (0-3)
        following = 1
        inst_keywords = set(re.findall(r'[\u4e00-\u9fff]{2,}', instruction))

        if "代码" in task_type:
            if "```" in output or "def " in output:
                following += 2
            elif "return" in output or "print" in output:
                following += 1
            if "while True" in output and "break" not in output:
                following -= 1
        elif "翻译" in task_type:
            if any('\u4e00' <= c <= '\u9fff' for c in output):
                following += 2
            if out_len < len(instruction) * 0.3:
                following -= 1
        elif "问答" in task_type:
            out_keywords = set(re.findall(r'[\u4e00-\u9fff]{2,}', output))
            overlap = len(inst_keywords & out_keywords) / max(1, len(inst_keywords))
            if overlap > 0.3:
                following += 1
            if overlap > 0.6:
                following += 1
        elif "推理" in task_type:
            logic_words = ["因为", "所以", "因此", "如果", "那么", "则"]
            if any(w in output for w in logic_words):
                following += 2
        elif "创意" in task_type:
            if len(set(output)) / max(1, len(output)) < 0.5:
                following -= 2

        if out_len > 20 and inst_keywords:
            out_keywords = set(re.findall(r'[\u4e00-\u9fff]{2,}', output))
            if not (inst_keywords & out_keywords):
                following -= 1

        if any(phrase in output for phrase in ["我不知道", "无法回答", "超出范围"]):
            following = max(0, following - 2)

        if self.config.enable_generic_following and task_type == "未知" and out_len > 20 and inst_keywords:
            out_keywords = set(re.findall(r'[\u4e00-\u9fff]{2,}', output))
            overlap = len(inst_keywords & out_keywords) / max(1, len(inst_keywords))
            if overlap > 0.2:
                following += 1
            if overlap > 0.5:
                following += 1
            following = min(3, following)

        following = min(3, max(0, following))

        # 3. 流畅性 (0-2)
        fluency = 2
        if re.search(r'(.)\1{4,}', output):
            fluency -= 1
        garbage = re.findall(r'[^a-zA-Z\u4e00-\u9fff\s，。！？；：""''《》【】（）]', output)
        if len(garbage) > out_len * 0.3:
            fluency -= 1
        if not re.search(r'[\u4e00-\u9fff]', output) and out_len > 100:
            fluency -= 1
        fluency = max(0, fluency)

        # 4. 安全性 (0-2)
        safety = 2
        if out_len < 5:
            safety = 0
        harmful = ["暴力", "色情", "非法", "诈骗"]
        if any(w in output for w in harmful):
            safety = max(0, safety - 1)

        total = completeness + following + fluency + safety
        return total, completeness, following, fluency, safety

    def _llm_score(self, instruction: str, output: str) -> Optional[Tuple[int, int, int]]:
        if self.llm_client is None:
            return None

        prompt = f"""请对以下指令-输出对进行质量评分，仅返回 JSON 格式。

评分标准：
- instruction_following (0-3分)：输出是否准确、完整地遵循了指令？
- fluency (0-2分)：语言是否流畅自然，无重复、乱码？
- safety (0-2分)：内容是否安全无害？

指令：{instruction[:500]}
输出：{output[:500]}

请返回 JSON：{{"instruction_following": 整数, "fluency": 整数, "safety": 整数}}"""

        try:
            response = self.llm_client.chat.completions.create(
                model=self.config.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=100
            )
            content = response.choices[0].message.content.strip()
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
            else:
                result = json.loads(content)
            follow = min(3, max(0, int(result.get("instruction_following", 2))))
            flu = min(2, max(0, int(result.get("fluency", 2))))
            safe = min(2, max(0, int(result.get("safety", 2))))
            return follow, flu, safe
        except Exception as e:
            logger.debug(f"LLM评分失败: {e}")
            return None

    def score(self, instruction: Any, input_text: Any, output: Any, task_type: str = "") -> Tuple[int, int, int, int, int]:
        inst = str(instruction) if instruction is not None else ""
        out = str(output) if output is not None else ""
        task_type = task_type if task_type is not None else ""

        rule_total, rule_comp, rule_follow, rule_flu, rule_safe = self._rule_score(inst, out, task_type)

        use_llm = False
        if self.config.enable_llm_scoring:
            if self.config.llm_score_lower <= rule_total <= self.config.llm_score_upper:
                if random.random() < self.config.llm_sample_ratio:
                    use_llm = True
                    self._init_llm_client()

        if use_llm:
            llm_result = self._llm_score(inst, out)
            if llm_result is not None:
                llm_follow, llm_flu, llm_safe = llm_result
                w_rule = self.config.llm_rule_weight
                w_llm = self.config.llm_model_weight
                final_follow = int(round(w_rule * rule_follow + w_llm * llm_follow))
                final_flu = int(round(w_rule * rule_flu + w_llm * llm_flu))
                final_safe = int(round(w_rule * rule_safe + w_llm * llm_safe))
                final_total = rule_comp + final_follow + final_flu + final_safe
                return final_total, rule_comp, final_follow, final_flu, final_safe
            else:
                logger.debug("LLM评分失败，回退至规则评分")

        return rule_total, rule_comp, rule_follow, rule_flu, rule_safe

    def evaluate_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("开始批量质量评估...")
        if self.config.enable_llm_scoring:
            logger.info(f"LLM混合评分已启用 (采样比例={self.config.llm_sample_ratio:.0%}, 边界=[{self.config.llm_score_lower}, {self.config.llm_score_upper}])")
        scores = []
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="评估进度"):
            total, comp, follow, flu, safe = self.score(
                row['instruction'], row.get('input', ''), row['output'], row.get('task_type', '')
            )
            scores.append({
                'total_score': total,
                'completeness': comp,
                'instruction_following': follow,
                'fluency': flu,
                'safety': safe
            })
        score_df = pd.DataFrame(scores)
        result = pd.concat([df, score_df], axis=1)
        logger.info("评估完成")
        return result