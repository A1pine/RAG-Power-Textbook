from sentence_transformers import SentenceTransformer
import numpy as np

class TextVectorizer:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(model_name)
        print(f"嵌入模型 {model_name} 加载完成。")

    def vectorize_text(self, text_data):
        """将文本数据转化为嵌入向量"""
        if not text_data:
            print("没有文本数据可供向量化。")
            return []
        embeddings = self.embedding_model.encode(text_data, convert_to_tensor=True)
        print(f"已为 {len(text_data)} 段文本生成嵌入向量。")
        return embeddings

    def vectorize_tables(self, table_data):
        """将表格数据转化为嵌入向量"""
        if not table_data:
            print("没有表格数据可供向量化。")
            return []
        table_texts = []
        for page_num, table in table_data:
            table_text = table.apply(lambda row: " | ".join(row.astype(str)), axis=1).str.cat(sep="\n")
            table_texts.append(table_text)
        embeddings = self.embedding_model.encode(table_texts, convert_to_tensor=True)
        print(f"已为 {len(table_data)} 个表格生成嵌入向量。")
        return embeddings
