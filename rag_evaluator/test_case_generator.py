from openai import OpenAI

class TestCaseGenerator:
    def __init__(self, api_key, api_base):
        self.client = OpenAI(
            api_key= api_key,  
            base_url = api_base
        )

    def generate_test_cases(self, context, num_cases=5):
        """使用 GPT 生成测试集"""
        prompt = f"""
        给定以下内容：
        {context}

        请生成 {num_cases} 个关于内容的用户查询，以及理想答案（以 JSON 格式输出）。
        """
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "你是智能辅导助手。"},
                      {"role": "user", "content": prompt}],
            temperature=0.7
        )
        return response.choices[0].message.content

if __name__ == "__main__":
    sample_context = """
    变压器是一种静止的电气装置，用于改变交流电的电压。
    它通常由初级绕组、次级绕组和铁芯组成。
    """
    testcase_generator = TestCaseGenerator("", "") # 替换为实际的API密钥和 API BASE
    testcase_generator.generate_test_cases(sample_context, 1)