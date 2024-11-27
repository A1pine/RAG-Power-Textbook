import time
from langchain_groq import ChatGroq
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from datasets import Dataset
import os
import json
import numpy as np
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings


class RAGEvaluator:
    def __init__(self, groq_api_key):
        # 初始化 LangChain ChatGroq 客户端
        self.client = ChatGroq(api_key=groq_api_key)
        self.embeddings = FastEmbedEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        os.environ["GROQ_API_KEY"] = groq_api_key

    def judge_with_gpt(self, query, retrieved, reference):
        """使用 GPT 对检索结果进行评分"""
        prompt = f"""
        用户的问题是：{query}
        RAG 系统的回答是：{retrieved}
        参考答案是：{reference}
        
        请评分（0-10），并解释理由，用英语回复
        格式：{{"score": score, "reason": "explanation"}}
        """
        messages = [
            {"role": "system", "content": "你是一个评估助手。"},
            {"role": "user", "content": prompt}
        ]
        print(query, retrieved, reference)
        response = self.client.invoke(messages)
        time.sleep(1)  # 延时 1 秒
        return response.content

    def evaluate_rag_with_ragas(self, test_cases, rag_system, retry_limit=3):
        """使用 RAGAs 评估 RAG 系统，自动处理 NaN 重试"""
        results = []
        for case in test_cases:
            query = case["query"]
            ideal_answer = case["ideal_answer"]
            retrieved = rag_system.retrieve(query, top_k=1)[0][0]  # 假设取第一个结果

            results.append({
                "query": query,                  # 用户输入
                "retrieved": retrieved,          # 检索到的内容
                "ground_truth": ideal_answer,    # 理想答案
                "user_input": query,             # 对应于 context_precision 所需的字段
                "retrieved_contexts": [retrieved],  # 对应于 context_precision 所需的字段
                "response": retrieved            # 对应于 faithfulness 所需的字段
            })

        eval_dataset = Dataset.from_list(results)

        for attempt in range(retry_limit):
            ragas_metrics = evaluate(
                eval_dataset,
                llm=self.client,
                embeddings=self.embeddings,
                raise_exceptions=False
            )
            time.sleep(1)  # 延时 1 秒

            # 检查是否存在 NaN 值
            contains_nan = False
            for metric in ragas_metrics.scores:
                for value in metric.values():
                    if np.isnan(value):
                        contains_nan = True
                        break  # 找到 NaN 后跳出当前循环

                if contains_nan:
                    break  # 如果在任一 metric 中找到 NaN，跳出外部循环

            if not contains_nan:
                return ragas_metrics  # 如果没有 NaN，返回评估结果
            else:
                print(f"评估中存在 NaN 值，重试第 {attempt + 1} 次...")


        raise ValueError("评估多次尝试后仍存在 NaN 值，请检查输入数据或 API 配置。")

    def generate_full_report(self, ragas_metrics, gpt_scores, output_file="rag_evaluation_report.html"):
        """生成综合评估报告"""
        with open(output_file, "w") as report:
            report.write("<h1>RAG Evaluation Report</h1>")

            # 添加 RAGAS 指标
            report.write("<h2>RAGAs Metrics</h2>")
            report.write(f"<p>{ragas_metrics.scores}</p>")

            # 添加 GPT 评分
            report.write("<h2>GPT Scores</h2>")
            for idx, score in enumerate(gpt_scores):
                report.write(f"<p><b>Query {idx + 1}:</b> {score}</p>")


# 示例用法
if __name__ == "__main__":
    # 假设 `test_cases` 和 `rag_system` 已定义
    test_cases = [
        {"query": "什么是机器学习？", "ideal_answer": "机器学习是一种让计算机自动学习的技术。"}
        # 更多测试用例
    ]

    class MockRAGSystem:
        def retrieve(self, query, top_k=1):
            return [("机器学习是一种让机器从数据中学习的技术。",)]

    rag_system = MockRAGSystem()

    evaluator = RAGEvaluator(groq_api_key="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")
    ragas_metrics = evaluator.evaluate_rag_with_ragas(test_cases, rag_system)
    gpt_scores = [
        evaluator.judge_with_gpt(
            case["query"],
            rag_system.retrieve(case["query"], top_k=1)[0][0],
            case["ideal_answer"]
        )
        for case in test_cases
    ]
    evaluator.generate_full_report(ragas_metrics, gpt_scores)
