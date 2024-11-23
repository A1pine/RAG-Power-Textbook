from rag_evaluator.rag_evaluator import RAGEvaluator
from rag_evaluator.test_case_generator import TestCaseGenerator
from knowledge_base.pdf_parser import PDFParser
from knowledge_base.text_vectorizer import TextVectorizer
from knowledge_base.vector_store import VectorStore
import os
import numpy as np

class RAGSystem:
    def __init__(self, pdf_path, index_path="examples/knowledge_base.index", embedding_model="sentence-transformers/all-MiniLM-L6-v2"):
        self.pdf_path = pdf_path
        self.index_path = index_path
        self.parser = PDFParser(pdf_path)
        self.vectorizer = TextVectorizer(model_name=embedding_model)
        self.vector_store = VectorStore()
        self.embeddings = []
    def build_knowledge_base(self):
        """解析 PDF 或从本地文件加载知识库"""
        print("正在构建知识库...")
        self.parser.parse_text()
        self.parser.parse_tables()
        # self.parser.parse_formulas()
            
        text_embeddings = self.vectorizer.vectorize_text(self.parser.text_data)
        table_embeddings = self.vectorizer.vectorize_tables(self.parser.table_data)
            
        self.embeddings.extend(text_embeddings)
        self.embeddings.extend(table_embeddings)
            
        self.vector_store.build_vector_store(self.embeddings)
        self.vector_store.save_vector_store(self.index_path)
        print(f"知识库已保存到 {self.index_path}")

    def retrieve(self, query, top_k=1):
        """从向量数据库检索相关文档"""
        if not self.embeddings:
            raise ValueError("知识库未初始化，请先调用 build_knowledge_base")
        
        # 检查text_data是否为空
        if not self.parser.text_data:
            raise ValueError("文本数据为空，请确保正确加载了文档数据")
        
        print("Text data length:", len(self.parser.text_data))  # 调试信息
        
        query_embedding = self.vectorizer.embedding_model.encode([query])
        indices, distances = self.vector_store.search_vector_store(query_embedding, top_k=top_k)
        
        print("Retrieved indices:", indices)
        print("Retrieved distances:", distances)
        
        # 确保索引是整数
        indices = [int(i) for i in indices[0]]
        distances = [float(d) for d in distances[0]]
        
        # 检查索引是否有效
        valid_indices = [i for i in indices if 0 <= i < len(self.parser.text_data)]
        if not valid_indices:
            raise ValueError("没有找到有效的检索结果")
        
        retrieved_docs = [(self.parser.text_data[i], dist) 
                        for i, dist in zip(valid_indices, distances[:len(valid_indices)])]
        return retrieved_docs

if __name__ == "__main__":
    # 示例 API 配置
    api_key = "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"

    # 初始化系统
    pdf_path = "examples/power_textbook.pdf"
    index_path = "examples/knowledge_base.index"
    rag_system = RAGSystem(pdf_path, index_path)
    rag_system.build_knowledge_base()

    # 测试集生成
    testcase_generator = TestCaseGenerator(api_key)
    sample_context = """
        The different voltage levels in a power system are due to the presence of transformers.
        Therefore, the procedure for selecting base voltage is as follows: 
        A voltage corresponding to any part of the system could be taken as a base and the 
        base voltages in other parts of the circuit, separated from the original part by transformers 
        is related through the turns ratio of the transformers. 
       """
    generated_testcases = testcase_generator.generate_test_cases(sample_context, 3)
    # generated_testcases = [{'query': '变压器的工作原理是什么？', 'answer': '变压器通过电磁感应原理，将交流电的电压从一个值变换到另一个值。当初级绕组通电时，在铁芯中产生的磁场会感应次级绕组中的电压，从而改变输出电压的大小。'}]
    print("生成的测试集:", generated_testcases)

    # 初始化评估工具
    evaluator = RAGEvaluator(api_key)

    # 使用 RAGAs 评估
    test_cases = [
        {"query": case["query"], "ideal_answer": case["answer"]}
        for case in generated_testcases
    ]
    ragas_metrics = evaluator.evaluate_rag_with_ragas(test_cases, rag_system)

    # 使用 GPT 评估每条测试案例
    gpt_scores = []
    for case in test_cases:
        retrieved_answer = rag_system.retrieve(case["query"], top_k=1)[0][0]
        gpt_score = evaluator.judge_with_gpt(case["query"], retrieved_answer, case["ideal_answer"])
        gpt_scores.append(gpt_score)

    # 输出评估报告
    output_file = os.path.join("examples", "rag_evaluation_report.html")
    evaluator.generate_full_report(ragas_metrics, gpt_scores, output_file)
    print(f"评估报告已生成: {output_file}")