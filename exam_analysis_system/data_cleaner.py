# data_cleaner.py
import re
import pandas as pd
from typing import Optional, List, Dict
import logging


class DataCleaner:
    @staticmethod
    def extract_exam_data(text: str) -> Optional[pd.DataFrame]:
        logging.info("[階段二] 啟動 Adaptive Data Cleaning...")
        # 【最優版正則】：雙語鍵名兼容 + 強制逗號分隔（保留結構性守門員）+ IGNORECASE
        pattern = re.compile(
            r"(?:题号|Question)[：:]\s*(.*?)\s*[，,]\s*"
            r"(?:知识点|Knowledge)[：:]\s*(.*?)\s*[，,]\s*"
            r"(?:难度|Difficulty)[：:]\s*([0-9.]+)\s*[，,]\s*"
            r"(?:分值|Marks)[：:]\s*([0-9.]+)",
            re.IGNORECASE
        )
        matches = pattern.findall(text)

        if not matches:
            return None

        parsed_rows: List[Dict] = []
        for match in matches:
            try:
                difficulty = float(match[2].strip('[] '))
                marks = float(match[3].strip('[] '))
                # 防伪匹配：難度必在 1-5，分值必 > 0
                if not (1.0 <= difficulty <= 5.0) or marks <= 0:
                    logging.warning(f"[數據校驗] 丟棄異常行: 難度={difficulty}, 分值={marks}")
                    continue
                parsed_rows.append({
                    'Question_ID': match[0].strip('[] '),
                    'Knowledge_Point': match[1].strip('[] '),
                    'Difficulty_Level': difficulty,
                    'Marks': marks
                })
            except ValueError:
                continue

        return pd.DataFrame(parsed_rows) if parsed_rows else None
