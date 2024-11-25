from rag_evaluator.rag_evaluator import RAGEvaluator
from rag_evaluator.test_case_generator import TestCaseGenerator
from knowledge_base.pdf_parser import PDFParser
from knowledge_base.text_vectorizer import TextVectorizer
from knowledge_base.vector_store import VectorStore

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.prompts import ChatPromptTemplate
# Rerank
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain_groq import ChatGroq

import os
import numpy as np

class RAGSystem:
    def __init__(self, pdf_path, index_path="examples/knowledge_base.index", embedding_model="sentence-transformers/all-MiniLM-L6-v2"):
        self.pdf_path = pdf_path
        self.index_path = index_path
        self.parser = PDFParser(pdf_path)
        self.embedding_model = embedding_model
        self.vectorizer = TextVectorizer(model_name=embedding_model)
        self.vector_store = VectorStore()
        self.documents = []

    def build_knowledge_base(self):
        """解析 PDF 或从本地文件加载知识库"""
        # 如果文件存在则跳过
        if os.path.exists(self.index_path):
            return
        print("正在构建知识库...")
        self.parser.parse_text()
        self.parser.parse_tables()
        # self.parser.parse_formulas()
            
        self.documents = self.parser.documents
            
        self.vector_store.build_vector_store(self.documents)
        self.vector_store.save_vector_store(self.index_path)
        print(f"知识库已保存到 {self.index_path}")

    def retrieve(self, query, top_k=1):
        """从向量数据库检索相关文档"""
        template = """以context为基础, 使用英语回答问题:
          {context}
        
          Question: {question}
          """
        
        prompt = ChatPromptTemplate.from_template(template)
 
        llm =  ChatGroq(api_key = api_key)
        compressor = FlashrankRerank()
        compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever, model=llm
        )
 
        chain = (
                RunnableParallel({"context": compression_retriever, "question": RunnablePassthrough()})
                | prompt
                | llm
                | StrOutputParser()
        )
 
        vector_answer = chain.invoke(query)
 
        return vector_answer


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
        The grading theory is more of theoretical interest than practical for the following reasons. Capacitance grading is difficult of non-availability of materials with widely varying permittivities and secondly with time the permittivities of the materials may change as a result this may completely change the potential gradient distribution and may even lead to complete rupture of the cable dielectric material at normal working voltage.
       """
    generated_testcases = testcase_generator.generate_test_cases(sample_context, 5)
    # generated_testcases = [{'query': '变压器的工作原理是什么？', 'answer': '变压器通过电磁感应原理，将交流电的电压从一个值变换到另一个值。当初级绕组通电时，在铁芯中产生的磁场会感应次级绕组中的电压，从而改变输出电压的大小。'}]
    print("生成的测试集:", generated_testcases)

    # 初始化评估工具
    evaluator = RAGEvaluator(api_key)

    # 使用 RAGAs 评估
    test_cases = [
        {"query": case["query"], "ideal_answer": case["answer"]}
        for case in generated_testcases
    ]

    # if not retriever: 
    vector_store = rag_system.vector_store.load_vector_store(index_path)
    retriever = vector_store.as_retriever()
    
    
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