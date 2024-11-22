from pdf_parser import PDFParser
from text_cleaner import TextCleaner

class KnowledgeBase:
    def __init__(self):
        self.data = []

    def add_entry(self, entry):
        """添加新条目到知识库"""
        self.data.append(entry)

    def save_to_file(self, file_path):
        """将知识库保存为文本文件"""
        with open(file_path, "w", encoding="utf-8") as f:
            for entry in self.data:
                f.write(entry + "\n")

    def load_from_file(self, file_path):
        """从文件加载知识库"""
        with open(file_path, "r", encoding="utf-8") as f:
            self.data = [line.strip() for line in f.readlines()]

class KnowledgeBuilder:
    def __init__(self):
        self.parser = PDFParser()
        self.cleaner = TextCleaner()

    def build_from_pdf(self, pdf_path):
        """从 PDF 文件构建知识库"""
        raw_text = self.parser.parse_pdf(pdf_path)
        cleaned_text = self.cleaner.clean_text(raw_text)
        return cleaned_text
