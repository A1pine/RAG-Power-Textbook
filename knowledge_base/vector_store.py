from sentence_transformers import SentenceTransformer
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np
from uuid import uuid4
import torch
import os

class VectorStore:
    def __init__(self):
        self.vector_store = None
        self.text_data = []  # 用于存储与向量相关的文档或文本信息
        self.embedding_model=FastEmbedEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    def build_vector_store(self, documents):
        """构建向量存储"""
        if documents is None or len(documents) == 0:
            print("没有嵌入向量可供构建向量数据库。")
            return

        # 将张量转移到 CPU 并转换为 numpy 数组
        # 确保所有张量都在 CPU 上
        # documents = torch.tensor(documents)
        documents = [str(doc) for doc in documents]
        # documents = np.asarray(documents) 


        # 将文档数据拆分为 chunk 的大小
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        documents = text_splitter.create_documents(documents)
        all_splits = text_splitter.split_documents(documents)

        # dimension = documents[0].shape[-1]
        index = faiss.IndexFlatL2(len(documents))
        self.vector_store =  FAISS.from_documents(
            documents=all_splits,
            embedding=self.embedding_model,
            normalize_L2=True,
        )
                

        uuids = [str(uuid4()) for _ in range(len(documents))]

        self.vector_store.add_documents(documents=documents, ids=uuids)
        print("向量数据库构建完成。")


    def save_vector_store(self, save_path):
        """保存向量数据库到文件"""
        if not self.vector_store:
            print("没有向量数据库可以保存。")
            return
        self.vector_store.save_local(save_path)
        print(f"向量数据库已保存至 {save_path}。")

    def load_vector_store(self, load_path):
        """从文件中加载向量数据库"""
        if not os.path.exists(load_path):
            print(f"向量数据库文件 {load_path} 不存在。")
            return
        self.vector_store = FAISS.load_local(load_path, self.embedding_model, allow_dangerous_deserialization=True)
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