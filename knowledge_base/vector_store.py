import faiss
import numpy as np

class VectorStore:
    def __init__(self):
        self.vector_store = None

    def build_vector_store(self, embeddings):
        """构建向量数据库"""
        if not embeddings:
            print("没有嵌入向量可供构建向量数据库。")
            return
        dimension = embeddings[0].shape[-1]
        index = faiss.IndexFlatL2(dimension)
        self.vector_store = faiss.IndexIDMap(index)
        vectors = np.vstack([embedding.cpu().numpy() for embedding in embeddings])
        ids = np.arange(len(vectors))
        self.vector_store.add_with_ids(vectors, ids)
        print("向量数据库构建完成。")

    def save_vector_store(self, save_path):
        """保存向量数据库到文件"""
        if not self.vector_store:
            print("没有向量数据库可以保存。")
            return
        faiss.write_index(self.vector_store, save_path)
        print(f"向量数据库已保存至 {save_path}。")
