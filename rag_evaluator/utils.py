import json

def load_test_data(file_path):
    """加载测试数据"""
    with open(file_path, "r") as f:
        return json.load(f)
