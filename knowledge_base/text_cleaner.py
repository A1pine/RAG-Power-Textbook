import re

class TextCleaner:
    def clean_text(self, text):
        """清理文本：去除空行、无关符号等"""
        text = re.sub(r"\s+", " ", text)  # 替换多余空格
        text = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fa5，。！？：；（）《》“”‘’]", "", text)  # 清理无关符号
        return text.strip()
