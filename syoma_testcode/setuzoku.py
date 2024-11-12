import requests
from requests.exceptions import ConnectionError, Timeout, RequestException
import torch

# 接続先のURLを指定
url = "http://133.67.33.50"

def print_gpu_memory_usage():
    print(f"Allocated: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")
    print(f"Cached: {torch.cuda.memory_reserved() / 1024 ** 3:.2f} GB")

print_gpu_memory_usage()

try:
    response = requests.get(url)
    response.raise_for_status()
except ConnectionError:
    print("ホストへの接続を確立できません。ホストへのルートがありません。")
except Timeout:
    print("リクエストがタイムアウトしました。")
except RequestException as e:
    print(f"リクエスト中にエラーが発生しました: {e}")
else:
    print("接続成功")