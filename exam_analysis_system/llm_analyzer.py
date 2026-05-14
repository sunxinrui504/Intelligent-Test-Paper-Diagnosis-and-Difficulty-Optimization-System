# llm_analyzer.py
import requests
import logging
import pandas as pd
from typing import Optional
from config import API_KEY, BASE_URL, MODEL_NAME

class LLMAnalyzer:
    @staticmethod
    def _call_api(system_prompt: str, user_prompt: str, temp: float) -> Optional[str]:
        # (保持原有的 API Call 邏輯不變)
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"}
        payload = {"model": MODEL_NAME, "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], "temperature": temp}
        try:
            return requests.post(BASE_URL, headers=headers, json=payload).json()['choices'][0]['message']['content']
        except Exception as e:
            return None

    @staticmethod
    def analyze_paper(content, attempt=1):
        from data_cleaner import DataCleaner
        from config import MAX_RETRIES, BASE_TEMPERATURE

        logging.info(f"[階段二] AI 知識點提取與難度量化 (第 {attempt} 次嘗試)...")
        system_prompt = "你是教育輔助專家。負責精確提取試卷資訊，只輸出規定格式。"

        # 【強化版 Prompt】：具體範例 + 格式禁令
        user_prompt = (
            f"分析以下試卷內容，提取每道題的題號、知識點、難度(1-5)、分值。\n"
            f"嚴格按照以下格式輸出每一道題，每題佔一行，用逗號分隔：\n"
            f"题号：1a，知识点：压力计算，难度：2.5，分值：2\n"
            f"\n"
            f"【禁令】不要換行到下一行內容、不要加粗、不要輸出任何解釋性文字、不要加編號列表！\n"
            f"\n"
            f"試卷內容：\n{content[:4000]}"
        )

        if attempt > 1:
            user_prompt = "[最高警告：你上次的輸出格式完全錯誤！請強制使用上述單行逗號分隔格式！]\n" + user_prompt

        result_text = LLMAnalyzer._call_api(system_prompt, user_prompt, BASE_TEMPERATURE + (attempt * 0.1))

        df = DataCleaner.extract_exam_data(result_text) if result_text else None
        if df is None and attempt < MAX_RETRIES:
            return LLMAnalyzer.analyze_paper(content, attempt + 1)
        return df

    @staticmethod
    def generate_teaching_advice(parsed_df, target_score):
        """生成教學診斷與建議"""
        logging.info("🧠 正在生成 AI 教學診斷建議...")
        kps = parsed_df.groupby('Knowledge_Point')['Marks'].sum().to_dict()
        sys_p = "你是資深教育學家。請根據試卷知識點佔比與目標分數，給出 3 點具體的教學優化建議。"
        usr_p = f"試卷知識點與分值：{kps}\n期望下次考試平均分：{target_score}\n請分析學生可能的知識缺口，並給出教學建議。"
        return LLMAnalyzer._call_api(sys_p, usr_p, temp=0.5)

    @staticmethod
    def generate_smart_exam(knowledge_points, target_dcw, exam_type, style_reference):
        """根據企劃書精確定義 GOOD / BAD，並模仿上傳試卷的風格"""
        sys_p = "你是頂級出題專家。請直接輸出試卷題目內容，不需要前言。"
        
        type_desc = {
            "BAD": "【Novice 基礎版】：側重基礎公式套用、單一知識點測試、題型直白簡單。適合資源匱乏地區或弱勢學生打基礎。",
            "GOOD": "【Expert 高階版】：側重多概念綜合分析、實驗設計、高階邏輯推導。情境複雜，適合拔高與測驗資優生。"
        }
        
        usr_p = f"""
        請設計一份包含 4-5 題的試卷。
        - 知識點範圍: {', '.join(knowledge_points[:6])}
        - 整體難度 (1-5): {target_dcw:.2f}
        - 試卷定位: {type_desc[exam_type]}
        
        【極度重要】：請模仿以下原始試卷的排版風格、用語習慣（如題號標示法 1(a)(i)、括號分值 [2] 等）：
        --- 原始風格參考 ---
        {style_reference[:1000]}
        ---
        
        如果有需要圖片的地方，請使用 [圖片佔位符：描述該圖片內容] 來代替。
        """
        return LLMAnalyzer._call_api(sys_p, usr_p, temp=0.7)
