from knowledge_base.pdf_parser import PDFParser
from knowledge_base.text_vectorizer import TextVectorizer
from knowledge_base.vector_store import VectorStore

class RAGPowerEdu:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.parser = PDFParser(pdf_path)
        self.vectorizer = TextVectorizer()
        self.vector_store = VectorStore()
        self.embeddings = []

    def parse_pdf(self):
        """解析 PDF 内容"""
        self.parser.parse_text()
        self.parser.parse_tables()
        self.parser.parse_formulas()

    def build_knowledge_base(self):
        """构建知识库"""
        text_embeddings = self.vectorizer.vectorize_text(self.parser.text_data)
        table_embeddings = self.vectorizer.vectorize_tables(self.parser.table_data)
        self.embeddings.extend(text_embeddings)
        self.embeddings.extend(table_embeddings)
        self.vector_store.build_vector_store(self.embeddings)

    def save_knowledge_base(self, save_path):
        """保存知识库"""
        self.vector_store.save_vector_store(save_path)

if __name__ == "__main__":
    pdf_path = "examples/power_textbook.pdf"
    save_path = "examples/knowledge_base.index"

    rag_system = RAGPowerEdu(pdf_path)
    rag_system.parse_pdf()
    rag_system.build_knowledge_base()
    rag_system.save_knowledge_base(save_path)
