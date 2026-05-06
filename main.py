import os
import pandas as pd
from omegaconf import OmegaConf
from transformers import AutoTokenizer

MODEL_PATH = "/data/models/Qwen3.5-0.8B"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

# 读取一条数据并调用 apply_chat_template
import json

df = pd.read_parquet("/data/datasets/td-mobile-data/train.parquet")
messages = df.iloc[0]["messages"]
print("type of messages:", type(messages))
print()

# 递归将 numpy 类型转为纯 Python 类型，并修复 arguments
def convert_numpy(obj):
    if hasattr(obj, "tolist"):
        return convert_numpy(obj.tolist())
    elif isinstance(obj, dict):
        new = {}
        for k, v in obj.items():
            if v is not None:
                # 将 arguments 从 JSON 字符串解析为 dict
                if k == "arguments" and isinstance(v, str):
                    v = json.loads(v)
                new[k] = convert_numpy(v)
        return new
    elif isinstance(obj, list):
        return [convert_numpy(v) for v in obj]
    return obj

messages = convert_numpy(messages)
print("messages:", messages)
print()

result = tokenizer.apply_chat_template(messages, tokenize=False)
print("apply_chat_template result:")
print(result)

