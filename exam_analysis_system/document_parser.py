# document_parser.py
import PyPDF2
from docx import Document
import logging
import os
from typing import Optional
from config import OUTPUT_DIR

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


class DocumentParser:
    @staticmethod
    def read_file(file_path: str) -> Optional[str]:
        logging.info(f"[階段一] 正在解析非結構化文件: {file_path}")
        try:
            content = ""
            if file_path.lower().endswith('.txt'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            elif file_path.lower().endswith('.pdf'):
                with open(file_path, 'rb') as f:
                    pdf = PyPDF2.PdfReader(f)
                    content = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
            elif file_path.lower().endswith('.docx'):
                doc = Document(file_path)
                content = "\n".join([para.text for para in doc.paragraphs])
            else:
                logging.error("不支持的文件格式。")
                return None

            # 儲存統一化的純文字文件
            base_name = os.path.basename(file_path).split('.')[0]
            output_path = os.path.join(OUTPUT_DIR, f"{base_name}_parsed_raw.txt")
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
            logging.info(f"[階段一] 成功輸出統一化文字檔: {output_path}")

            return content
        except Exception as e:
            logging.error(f"文件讀取失敗 {file_path}: {e}")
            return None
