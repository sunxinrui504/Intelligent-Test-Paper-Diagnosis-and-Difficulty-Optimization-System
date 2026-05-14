# config.py
import os

# API Configuration
API_KEY = os.getenv("KIMI_API_KEY", "sk-cKhe6E9FDt2vTAvtmQVoo3ShkjK61LKxOY4mGiG0pL0r8rEO")
BASE_URL = 'https://api.moonshot.cn/v1/chat/completions'
MODEL_NAME = 'moonshot-v1-8k'

# System Parameters
MAX_RETRIES = 3
BASE_TEMPERATURE = 0.2
SYNTHETIC_PAPER_COUNT = 40  # 模擬更多歷史數據以提高 ML 精準度
STUDENTS_PER_CLASS = 50

# Output Directories
OUTPUT_DIR = "system_outputs"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
