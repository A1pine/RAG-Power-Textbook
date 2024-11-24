import fitz  # PyMuPDF
import pdfplumber
import pytesseract
from PIL import Image
import pandas as pd

class PDFParser:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.documents = []
        self.text_data = []
        self.table_data = []
        self.formula_texts = []

    def parse_text(self):
        """使用 PyMuPDF 提取 PDF 中的纯文本内容"""
        doc = fitz.open(self.pdf_path)
        text_data = []
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()
            if text.strip():  # 跳过空白页面
                text_data.append(text.strip())
        self.text_data = text_data
        self.documents.extend(self.text_data)
        print(f"共解析 {len(text_data)} 页纯文本。")

    def parse_tables(self):
        """使用 pdfplumber 提取 PDF 中的表格内容"""
        with pdfplumber.open(self.pdf_path) as pdf:
            table_data = []
            for page_num, page in enumerate(pdf.pages):
                tables = page.extract_tables()
                for table in tables:
                    df = pd.DataFrame(table)  # 转换为 DataFrame
                    table_data.append((page_num + 1, df))
            self.table_data = table_data
        self.documents.extend(self.table_data)
        print(f"共解析 {len(table_data)} 个表格。")

    def parse_formulas(self):
        """提取公式区域的图片并使用 OCR 转换为文本"""
        doc = fitz.open(self.pdf_path)
        formula_texts = []
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            ocr_result = pytesseract.image_to_string(img)
            formula_texts.append((page_num + 1, ocr_result))
        self.formula_texts = formula_texts
        self.documents.extend(self.formula_texts)
        print(f"共解析 {len(formula_texts)} 页公式文本。")
