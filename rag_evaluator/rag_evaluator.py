from ragas import evaluate
from openai import OpenAI

class RAGEvaluator:
    def __init__(self, api_key, api_base):
        self.client = OpenAI(
            api_key=api_key,
            base_url=api_base
        )
    
    def judge_with_gpt(self, query, retrieved, reference):
        """使用 GPT 对检索结果进行评分"""
        prompt = f"""
        用户的问题是：{query}
        RAG 系统的回答是：{retrieved}
        参考答案是：{reference}
         
        请评分（0-10），并解释理由。
        格式：{{"score": 分数, "reason": "解释"}}
        """
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "你是一个评估助手。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        return response.choices[0].message.content

    def evaluate_rag_with_ragas(self, test_cases, rag_system):
        """使用 RAGAs 评估 RAG 系统"""
        results = []
        for case in test_cases:
            query = case["query"]
            ideal_answer = case["ideal_answer"]
            retrieved = rag_system.retrieve(query, top_k=1)[0][0]
            results.append({
                "query": query,
                "retrieved": retrieved,
                "reference": ideal_answer
            })
        return evaluate(results)

    def generate_full_report(self, ragas_metrics, gpt_scores, output_file="rag_evaluation_report.html"):
        """生成综合评估报告"""
        with open(output_file, "w") as report:
            report.write("<h1>RAG Evaluation Report</h1>")
            report.write("<h2>RAGAs Metrics</h2>")
            report.write(f"<pre>{ragas_metrics}</pre>")
            report.write("<h2>GPT Scores</h2>")
            for idx, score in enumerate(gpt_scores):
                report.write(f"<p>Query {idx+1}: {score}</p>")
