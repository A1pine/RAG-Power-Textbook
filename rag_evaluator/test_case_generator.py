from langchain_groq import ChatGroq
import json
import re

class TestCaseGenerator:
    def __init__(self, groq_api_key):
        # 初始化 ChatGroq 客户端
        self.client = ChatGroq(
            api_key=groq_api_key,
        )

    # 提取 JSON 部分
    def extract_json_from_response(self, content):
        # 使用正则表达式查找 JSON 块
        json_pattern = r"```\s*(\{.*?\})\s*```|(\{.*?\})"
        match = re.search(json_pattern, content, re.DOTALL)
        
        if match:
            json_str = match.group(1) or match.group(2)
            try:
                # 尝试解析 JSON
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                print(f"JSON 解码错误: {e}")
                print(f"无法解析的 JSON 内容: {json_str}")
                return None
        else:
            # 如果找不到 JSON 块，尝试手动分割并清理
            try:
                json_start = content.find("{")
                json_end = content.rfind("}")
                if json_start != -1 and json_end != -1:
                    json_str = content[json_start:json_end + 1]
                    return json.loads(json_str)
            except json.JSONDecodeError as e:
                print(f"JSON 解码错误: {e}")
            print("未找到 JSON 格式的内容。")
            return None

    def generate_test_cases(self, context, num_cases=5):
        """使用 ChatGroq 生成测试集"""
        prompt = f"""
        给定以下内容：
        {context}

        请生成 {num_cases} 个关于内容的用户查询，以及理想答案（以 JSON 格式输出）， 用英语回复。
        格式例如:
        [
          {{"query": "什么是变压器的主要应用场景",
            "answer": "电力传输, 家庭电器, 工业设备等"}},
          {{"query": "什么是变压器的组成",
            "answer": "初级绕组, 次级绕组, 铁芯"}}
        ]
        """

        messages = [
            {"role": "system", "content": "你是智能辅导助手。"},
            {"role": "user", "content": prompt}
        ]

        response = self.client.invoke(messages)
        content = response.content
        # print(content)

        # 提取 JSON 测试集
        test_cases = json.loads(content)
        if test_cases:
            return test_cases
        else:
            raise ValueError(f"生成测试集失败，响应内容: {content}")

if __name__ == "__main__":
    sample_context = """
    变压器是一种静止的电气装置，用于改变交流电的电压。
    它通常由初级绕组、次级绕组和铁芯组成。
    """
    # 替换为你的 Groq API Key
    groq_api_key = ""
    testcase_generator = TestCaseGenerator(groq_api_key)
    test_cases = testcase_generator.generate_test_cases(sample_context, num_cases=2)
    print(test_cases)
