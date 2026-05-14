# ml_engine.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import logging

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'PingFang SC', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False 

class MLEngine:
    def __init__(self):
        # 集成學習模型庫
        self.models = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
            "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        }
        self.best_model_name = ""
        self.best_model = None
        self.is_trained = False
        
    def calculate_empirical_difficulty(self, parsed_df: pd.DataFrame, student_scores_df: pd.DataFrame) -> pd.DataFrame:
        """
        【實證難度校正】：根據每個學生在每一題的真實得分，計算實證難度
        公式：難度係數 (1-5) = 1 + 4 * (1 - 該題平均得分 / 該題滿分)
        得分率越低 → 難度越高；全班滿分 → 難度=1；全班零分 → 難度=5
        """
        logging.info("[數據校正] 正在利用真實學生作答數據計算實證難度...")
        corrected_df = parsed_df.copy()

        for index, row in corrected_df.iterrows():
            q_id = str(row['Question_ID'])
            max_marks = float(row['Marks'])

            # 在成績表中尋找匹配的題號欄位
            matching_cols = [col for col in student_scores_df.columns if q_id in str(col)]

            if matching_cols:
                col = matching_cols[0]
                avg_score = student_scores_df[col].mean()
                scoring_rate = min(max(avg_score / max_marks, 0.0), 1.0)
                real_difficulty = 1 + 4 * (1 - scoring_rate)
                old_diff = row['Difficulty_Level']
                corrected_df.at[index, 'Difficulty_Level'] = round(real_difficulty, 2)
                logging.info(f"  題目 {q_id}: LLM預估={old_diff} → 實證難度={round(real_difficulty, 2)} (得分率={scoring_rate:.1%})")
            else:
                logging.warning(f"  成績表中缺少題目 {q_id} 的數據，保留 LLM 預估難度。")

        return corrected_df

    def process_real_and_augment_data(self, parsed_data_list, real_scores_matrix_list) -> pd.DataFrame:
        """
        【真實數據融合 + 數據增強】：結合真實作答矩陣與 DCW 計算，進行高斯噪聲擴增
        若無真實數據，退回 LLM 預估 + 基準公式換算（向後兼容）
        """
        logging.info("[階段三] 啟動真實數據融合與 Gaussian 數據增強...")

        real_training_data = []
        for i, df in enumerate(parsed_data_list):
            if df.empty: continue

            # 如果有提供該份試卷的真實作答矩陣，則進行實證難度校正
            if i < len(real_scores_matrix_list) and real_scores_matrix_list[i] is not None:
                scores_df = real_scores_matrix_list[i]
                # 1. 計算並覆蓋實證難度
                corrected_df = self.calculate_empirical_difficulty(df, scores_df)
                # 2. 計算這份試卷整體的真實 DCW（加權難度）
                dcw = np.average(corrected_df['Difficulty_Level'], weights=corrected_df['Marks'])
                # 3. 計算這份試卷的真實全班總平均分
                real_avg_score = scores_df.sum(axis=1).mean()
                logging.info(f"  Paper_{i+1}: DCW={dcw:.2f}, 真實平均分={real_avg_score:.2f}")
            else:
                # 若無真實數據，使用 LLM 預估數據（向後兼容）
                dcw = np.average(df['Difficulty_Level'], weights=df['Marks'])
                real_avg_score = 100 * (1 - (dcw * 0.12))
                logging.info(f"  Paper_{i+1}: DCW={dcw:.2f}, 預估平均分={real_avg_score:.2f} (無真實數據)")

            real_training_data.append({'DCW': dcw, 'Average_Score': real_avg_score})

        base_df = pd.DataFrame(real_training_data)
        logging.info(f"成功處理 {len(base_df)} 份歷史試卷的特徵萃取！")

        # Data Augmentation: 將少量的真實數據擴充 20 倍，供集成模型學習避免過擬合
        logging.info("[階段三] 啟動 Data Augmentation (Gaussian Noise)...")
        augmented_data = []
        for _, row in base_df.iterrows():
            for _ in range(20):
                noise_dcw = np.clip(np.random.normal(row['DCW'], 0.15), 1.0, 5.0)
                noise_score = np.clip(np.random.normal(row['Average_Score'], 4.5), 0, 100)
                augmented_data.append({'DCW': round(noise_dcw, 2), 'Average_Score': round(noise_score, 2)})

        return pd.concat([base_df, pd.DataFrame(augmented_data)], ignore_index=True)

    def generate_synthetic_data(self, parsed_df: pd.DataFrame, num_students: int, num_papers: int) -> pd.DataFrame:
        logging.info("[階段三] 根據聚合數據計算整體 DCW 並生成歷史成績...")
        base_dcw = np.average(parsed_df['Difficulty_Level'], weights=parsed_df['Marks'])
        
        history = []
        for i in range(num_papers):
            paper_dcw = np.clip(np.random.normal(base_dcw, 0.8), 1.0, 5.0) 
            expected_mean = 100 * (1 - (paper_dcw * 0.12)) # 100分制
            scores = np.clip(np.random.normal(loc=expected_mean, scale=12, size=num_students), 0, 100)
            history.append({
                'Paper_ID': f'Paper_Hist_{i+1}',
                'DCW': round(paper_dcw, 2),
                'Average_Score': round(np.mean(scores), 2)
            })
        return pd.DataFrame(history)

    def train_and_select_best_model(self, hist_df: pd.DataFrame):
        X = hist_df[['DCW']].values
        y = hist_df['Average_Score'].values
        best_r2 = -float('inf')
        
        for name, model in self.models.items():
            model.fit(X, y)
            y_pred = model.predict(X)
            r2 = r2_score(y, y_pred)
            mse = mean_squared_error(y, y_pred)
            logging.info(f"  ├─ {name} 模型 -> R²: {r2:.3f}, MSE: {mse:.2f}")
            
            if r2 > best_r2:
                best_r2 = r2
                self.best_model_name = name
                self.best_model = model
                
        self.is_trained = True
        logging.info(f"  └─ 🏆 獲勝模型: {self.best_model_name}")

    def train_and_visualize(self, hist_df: pd.DataFrame, parsed_df: pd.DataFrame):
        """保留您最喜歡的四宮格彈出視窗"""
        X = hist_df[['DCW']].values
        y = hist_df['Average_Score'].values
        y_pred = self.best_model.predict(X)
        
        # 為了畫出平滑的預測線，將 X 排序
        sorted_zip = sorted(zip(X.flatten(), y_pred))
        X_sorted, y_pred_sorted = zip(*sorted_zip)
        
        plt.style.use('ggplot')
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle(f"Enterprise Analytics Dashboard (Powered by {self.best_model_name})", fontsize=18, fontweight='bold')
        
        # 1. 迴歸分析
        axes[0, 0].scatter(X, y, color='#1f77b4', s=50, alpha=0.7)
        axes[0, 0].plot(X_sorted, y_pred_sorted, color='#d62728', linewidth=2.5)
        axes[0, 0].set_title(f"Prediction Model: DCW vs Score")
        axes[0, 0].set_xlabel("DCW (Difficulty)"); axes[0, 0].set_ylabel("Average Score")
        
        # 2. 殘差分析
        axes[0, 1].scatter(y_pred, y - y_pred, color='#9467bd', s=50, alpha=0.7)
        axes[0, 1].axhline(y=0, color='black', linestyle='--')
        axes[0, 1].set_title("Residual Analysis (Model Validation)")
        axes[0, 1].set_xlabel("Predicted Score"); axes[0, 1].set_ylabel("Residuals")
        
        # 3. 聚合難度分佈圓餅圖
        diff_counts = parsed_df['Difficulty_Level'].value_counts()
        axes[1, 0].pie(diff_counts, labels=[f"Level {float(k):.1f}" for k in diff_counts.index], autopct='%1.1f%%', colors=plt.cm.Paired.colors)
        axes[1, 0].set_title("Aggregated Difficulty Distribution")
        
        # 4. 知識點覆蓋長條圖
        kp_marks = parsed_df.groupby('Knowledge_Point')['Marks'].sum().sort_values(ascending=False).head(10).sort_values()
        axes[1, 1].barh(kp_marks.index, kp_marks.values, color='#2ca02c')
        axes[1, 1].set_title("Top 10 Knowledge Point Coverage")
        
        plt.tight_layout()
        # 非常重要：設定 block=False，讓終端機繼續打印生成的試卷，而不是卡在這裡
        plt.show(block=False)
        plt.pause(0.1)

    def predict_target_dcw(self, target_score: float) -> float:
        """非線性模型的網格搜索逆向推導 (Grid Search Inverse Prediction)"""
        if not self.is_trained: return 3.0
        candidate_dcws = np.linspace(1.0, 5.0, 400).reshape(-1, 1)
        predicted_scores = self.best_model.predict(candidate_dcws)
        closest_index = np.argmin(np.abs(predicted_scores - target_score))
        return float(np.clip(candidate_dcws[closest_index][0], 1.0, 5.0))
