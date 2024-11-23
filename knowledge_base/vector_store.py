from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os

class VectorStore:
    def __init__(self):
        self.vector_store = None
        self.text_data = []  # 用于存储与向量相关的文档或文本信息

    def build_vector_store(self, embeddings):
        """构建向量存储"""
        if embeddings is None or len(embeddings) == 0:
            print("没有嵌入向量可供构建向量数据库。")
            return

        # 将张量转移到 CPU 并转换为 numpy 数组
        embeddings = [embedding.cpu().numpy() for embedding in embeddings]

        dimension = embeddings[0].shape[-1]
        index = faiss.IndexFlatL2(dimension)
        self.vector_store = faiss.IndexIDMap(index)
        vectors = np.vstack(embeddings)
        ids = np.arange(len(vectors))
        self.vector_store.add_with_ids(vectors, ids)
        print(f"Number of vectors: {len(vectors)}, Number of IDs: {len(ids)}")
        print("向量数据库构建完成。")


    def save_vector_store(self, save_path):
        """保存向量数据库到文件"""
        if not self.vector_store:
            print("没有向量数据库可以保存。")
            return
        faiss.write_index(self.vector_store, save_path)
        print(f"向量数据库已保存至 {save_path}。")

    def load_vector_store(self, load_path):
        """从文件中加载向量数据库"""
        if not os.path.exists(load_path):
            print(f"向量数据库文件 {load_path} 不存在。")
            return
        self.vector_store = faiss.read_index(load_path)
        print(f"向量数据库已从 {load_path} 加载。")
        return self.vector_store
    
    def search_vector_store(self, query_embedding, top_k=5):
        """在向量数据库中搜索与查询向量最相似的向量"""
        if not self.vector_store:
            print("没有向量数据库可供搜索。")
            return []
        
        query_embedding = query_embedding.reshape(1, -1)
        print(f"Query embedding shape: {query_embedding.shape}, dtype: {query_embedding.dtype}")
        print(type(self.vector_store))
        try:
            distances, indices = self.vector_store.search(query_embedding, k=top_k)
        except Exception as e:
            print(f"Error during search: {e}")
            return []
        return distances, indices

# 使用示例
if __name__ == "__main__":
    # 单进程运行代码
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    sentences = ["这是一个句子。", "这是另一个句子。"]
    embeddings = model.encode(sentences)

    store = VectorStore()
    store.build_vector_store(embeddings)

    query_sentence = "这是一个查询句子。"
    query_embedding = model.encode([query_sentence])[0]
    indices = store.search_vector_store(query_embedding)
    print(f"最相似的向量索引: {indices}")