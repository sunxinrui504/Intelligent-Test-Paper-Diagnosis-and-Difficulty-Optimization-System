# main.py
import tkinter as tk
from tkinter import filedialog, simpledialog
import logging
import os
import pandas as pd
import matplotlib.pyplot as plt
from config import SYNTHETIC_PAPER_COUNT, STUDENTS_PER_CLASS, OUTPUT_DIR
from tkinter import messagebox
from document_parser import DocumentParser
from llm_analyzer import LLMAnalyzer
from ml_engine import MLEngine
from exporter import ReportExporter

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def run_enterprise_pipeline():
    root = tk.Tk()
    root.withdraw()
    
    print("\n" + "="*80)
    print(" 🚀 啟動：基於 AI 與機器學習的試卷分析與評估智慧輔助系統 (終極完全體)")
    print("="*80 + "\n")
    
    # =========================================================
    # 步驟 1：上傳歷史試卷 (PDF/Word/TXT) — 原有功能保留
    # =========================================================
    root.call('wm', 'attributes', '.', '-topmost', True)
    messagebox.showinfo("步驟 1/3", "請選擇您要分析的【原始歷史試卷】(可多選)")
    file_paths = filedialog.askopenfilenames(
        title="[階段一] 批量上傳海量試卷", 
        filetypes=[("Documents", "*.txt;*.pdf;*.docx")]
    )
    if not file_paths: return

    # =========================================================
    # 步驟 2：輸入學生人數 — 由教師在界面輸入，不再硬編碼
    # =========================================================
    num_students = simpledialog.askinteger(
        "步驟 2/3：學生人數",
        f"請輸入班級學生人數（預設 {STUDENTS_PER_CLASS}）：",
        initialvalue=STUDENTS_PER_CLASS, minvalue=1, maxvalue=999
    )
    if not num_students: num_students = STUDENTS_PER_CLASS
    print(f"👨‍🎓 班級學生人數設定為: {num_students}")

    # =========================================================
    # 步驟 3：上傳真實歷史成績 (Excel/CSV) — 多 Sheet 對應不同試卷
    # =========================================================
    messagebox.showinfo("步驟 3/3", "請選擇該班級的【歷史成績表】\n\n"
        "Excel 格式：每個工作表(Sheet)對應一份試卷，Sheet名稱需包含試卷檔名關鍵字\n"
        "例如：試卷檔名 '1_Density_3_QP.pdf' → Sheet名 '1_Density' 或 'Density'\n"
        "CSV 格式：單一工作表，對應所有試卷\n\n"
        "如無成績表請按取消，系統將使用 LLM 預估難度。")
    grades_path = filedialog.askopenfilename(
        title="上傳歷史成績（可跳過）",
        filetypes=[("Excel/CSV", "*.xlsx;*.csv")]
    )

    # 解析成績表：每份試卷對應一個 DataFrame，欄位=題號，行=學生各題得分
    real_scores_matrix_list = []
    has_real_data = False
    if grades_path:
        print(f"\n📥 正在讀取真實成績數據: {os.path.basename(grades_path)}")
        try:
            if grades_path.endswith('.csv'):
                # CSV 只有單一工作表，分配給所有試卷
                df_grades = pd.read_csv(grades_path)
                numeric_cols = df_grades.select_dtypes(include='number').columns.tolist()
                if numeric_cols:
                    scores_df = df_grades[numeric_cols]
                    for _ in file_paths:
                        real_scores_matrix_list.append(scores_df)
                    has_real_data = True
                    print(f"  ✅ CSV 模式：{len(scores_df)} 名學生 × {len(scores_df.columns)} 題 → 分配給所有試卷")
            else:
                # Excel：讀取所有工作表，智能匹配對應試卷
                all_sheets = pd.read_excel(grades_path, sheet_name=None)
                print(f"  偵測到 {len(all_sheets)} 個工作表: {list(all_sheets.keys())}")

                # 為每份試卷尋找最匹配的 Sheet
                paper_filenames = [os.path.basename(fp).rsplit('.', 1)[0] for fp in file_paths]
                sheet_matched = {}  # sheet_name -> paper_index

                for paper_idx, paper_name in enumerate(paper_filenames):
                    best_sheet = None
                    best_overlap = 0

                    for sheet_name in all_sheets.keys():
                        # 計算試卷檔名與 Sheet 名的最大公共子串重疊度
                        p_lower = paper_name.lower().replace('_', ' ')
                        s_lower = sheet_name.lower().replace('_', ' ')
                        # 逐詞匹配：試卷檔名的每個詞是否出現在 Sheet 名中
                        p_words = [w for w in p_lower.split() if len(w) > 1]
                        s_words = [w for w in s_lower.split() if len(w) > 1]
                        overlap = sum(1 for w in p_words if any(w in sw or sw in w for sw in s_words))

                        if overlap > best_overlap:
                            best_overlap = overlap
                            best_sheet = sheet_name

                    if best_sheet and best_overlap > 0:
                        sheet_matched[best_sheet] = paper_idx
                        print(f"  📎 試卷 '{paper_name}' ↔ Sheet '{best_sheet}' (匹配度={best_overlap})")
                    else:
                        print(f"  ⚠️ 試卷 '{paper_name}' 未找到匹配的 Sheet")

                # 構建每份試卷對應的成績 DataFrame
                for paper_idx in range(len(file_paths)):
                    matched_sheet = None
                    for s_name, p_idx in sheet_matched.items():
                        if p_idx == paper_idx:
                            matched_sheet = s_name
                            break

                    if matched_sheet:
                        df_sheet = all_sheets[matched_sheet]
                        numeric_cols = df_sheet.select_dtypes(include='number').columns.tolist()
                        if numeric_cols:
                            real_scores_matrix_list.append(df_sheet[numeric_cols])
                            has_real_data = True
                        else:
                            real_scores_matrix_list.append(None)
                    else:
                        real_scores_matrix_list.append(None)

                # 如果有未匹配的試卷但有多餘的 Sheet，按順序補位
                unmatched_sheets = [s for s in all_sheets.keys() if s not in sheet_matched]
                for i in range(len(real_scores_matrix_list)):
                    if real_scores_matrix_list[i] is None and unmatched_sheets:
                        fallback_sheet = unmatched_sheets.pop(0)
                        df_sheet = all_sheets[fallback_sheet]
                        numeric_cols = df_sheet.select_dtypes(include='number').columns.tolist()
                        if numeric_cols:
                            real_scores_matrix_list[i] = df_sheet[numeric_cols]
                            has_real_data = True
                            print(f"  📎 試卷 '{paper_filenames[i]}' ↔ Sheet '{fallback_sheet}' (順序補位)")

                if has_real_data:
                    matched_count = sum(1 for x in real_scores_matrix_list if x is not None)
                    print(f"  ✅ Excel 模式：成功匹配 {matched_count}/{len(file_paths)} 份試卷的成績數據！")
                else:
                    print("  ⚠️ 所有工作表中均未找到數值欄位。")

        except Exception as e:
            print(f"❌ 成績表讀取失敗 ({e})，將使用 LLM 預估難度。")

    if not has_real_data:
        print("\n⚠️ 未上傳成績表，系統將使用 LLM 預估難度 + 虛擬數據進行訓練。")
        real_scores_matrix_list = [None] * len(file_paths)

    ml_system = MLEngine()
    all_parsed_data = []
    parsed_data_list = []  # 記錄每份試卷獨立的 DataFrame 以便與成績對接
    style_reference = ""

    # 迴圈解析所有試卷 (保留您喜歡的詳細進度面板)
    for i, filename_path in enumerate(file_paths):
        filename = os.path.basename(filename_path)
        print(f"\n▶ 正在批次處理入庫: {filename}")
        content = DocumentParser.read_file(filename_path)
        if not content: continue
        
        # 提取第一份試卷作為 AI 出題的排版風格參考
        if i == 0: style_reference = content[:1500] 
        
        parsed_df = LLMAnalyzer.analyze_paper(content)
        if parsed_df is not None:
            all_parsed_data.append(parsed_df)
            parsed_data_list.append(parsed_df)  # 同時記錄到獨立列表，供真實成績對接
            print(f"  └─ 成功提取 {len(parsed_df)} 個考點數據。")
            
    if not all_parsed_data:
        print("❌ 所有文件解析失敗。")
        return

    # 【企業級聚合】將所有試卷數據融合成一個大數據表
    print("\n🔗 正在聚合所有試卷數據建立知識圖譜 (Data Lake)...")
    master_df = pd.concat(all_parsed_data, ignore_index=True)
    master_df.to_csv(os.path.join(OUTPUT_DIR, "Master_Database.csv"), index=False, encoding='utf-8-sig')
    
    # =========================================================
    # 階段三：根據是否有真實數據，選擇不同的訓練路徑
    # =========================================================
    if has_real_data:
        print("\n🔬 偵測到真實成績數據！啟動【實證難度校正 + 數據增強】路徑...")
        hist_df = ml_system.process_real_and_augment_data(parsed_data_list, real_scores_matrix_list)
        # 用校正後的數據重新聚合 master_df（實證難度已覆蓋 LLM 預估）
        corrected_dfs = []
        for i, df in enumerate(parsed_data_list):
            if i < len(real_scores_matrix_list) and real_scores_matrix_list[i] is not None:
                corrected_dfs.append(ml_system.calculate_empirical_difficulty(df, real_scores_matrix_list[i]))
            else:
                corrected_dfs.append(df)
        master_df = pd.concat(corrected_dfs, ignore_index=True)
        master_df.to_csv(os.path.join(OUTPUT_DIR, "Master_Database.csv"), index=False, encoding='utf-8-sig')
    else:
        print("\n⚙️ 未提供真實成績，啟動【LLM 預估 + 虛擬數據生成】路徑（原有流程）...")
        hist_df = ml_system.generate_synthetic_data(master_df, num_students, SYNTHETIC_PAPER_COUNT)
    
    # 【新技術】訓練多個模型 (LR, Random Forest, XGBoost) 並選出最佳
    print("\n⚙️ 正在執行 Ensemble Learning (集成學習) 尋找最佳模型...")
    ml_system.train_and_select_best_model(hist_df)
    
    # 【舊精華】生成並彈出機器學習診斷儀表板 (Matplotlib 四宮格)
    print("📊 正在生成機器學習診斷儀表板...")
    ml_system.train_and_visualize(hist_df, master_df)
    
    # 彈出視窗詢問期望平均分
    target_score = simpledialog.askfloat(
        "階段五：智慧生成 (Target-Oriented)", 
        f"🏆 最佳預測模型: {ml_system.best_model_name}\n\n已成功融合 {len(file_paths)} 份試卷。\n請輸入您期望的『下一次班級平均分』(0-100):",
        minvalue=0, maxvalue=100
    )
    
    if target_score:
        print(f"\n🎯 [階段五] 目標設定：期望班級平均分 {target_score} 分")
        target_dcw = ml_system.predict_target_dcw(target_score)
        print(f"🧠 ML 逆向回歸算法推導 -> 所需試卷難度係數 (DCW) 應為: {target_dcw:.2f}")
        
        # 【新技術】生成教學診斷建議
        advice = LLMAnalyzer.generate_teaching_advice(master_df, target_score)
        print(f"\n💡 【AI 教學診斷與優化建議】\n{advice}")
        
        # 獲取全局知識點
        knowledge_points = master_df['Knowledge_Point'].unique().tolist()
        
        print("\n⏳ 正在指揮大模型生成 【BAD 型試卷】 (基礎套用型)...")
        bad_exam = LLMAnalyzer.generate_smart_exam(knowledge_points, target_dcw, "BAD", style_reference)
        
        print("⏳ 正在指揮大模型生成 【GOOD 型試卷】 (綜合推導型)...")
        good_exam = LLMAnalyzer.generate_smart_exam(knowledge_points, target_dcw, "GOOD", style_reference)
        
        # 【新技術】導出為企業級 Word 檔案 (.docx) 與 ECharts (HTML)
        ReportExporter.export_exam_to_word(bad_exam, "Generated_BAD", OUTPUT_DIR)
        ReportExporter.export_exam_to_word(good_exam, "Generated_GOOD", OUTPUT_DIR)
        echarts_path = ReportExporter.generate_echarts_html(hist_df, ml_system.best_model_name, target_dcw, OUTPUT_DIR)
            
        # 【舊精華】直接在終端機列印出來讓您預覽！
        print("\n" + "="*35 + " 產出預覽：BAD 型試卷 " + "="*35)
        print(bad_exam)
        print("="*92)
        
        print("\n" + "="*35 + " 產出預覽：GOOD 型試卷 " + "="*35)
        print(good_exam)
        print("="*93)
        
        print(f"\n✅ 任務完美完成！")
        print(f"📄 試卷已導出為 Word (.docx) 格式，存於 {OUTPUT_DIR}")
        print(f"📈 ECharts 可視化報告已生成：{echarts_path} (請用瀏覽器打開)")
        print("💡 請查看彈出的資料圖表（Dashboard），按右上角 X 即可退出程式。")
        
        # 保持圖表打開直到使用者關閉
        plt.show()

if __name__ == "__main__":
    run_enterprise_pipeline()
