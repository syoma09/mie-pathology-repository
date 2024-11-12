import GPUtil

# GPUの使用状況を取得
gpus = GPUtil.getGPUs()

# 各GPUの情報を表示
for gpu in gpus:
    print(f"GPU {gpu.id}:")
    print(f"  Name: {gpu.name}")
    print(f"  Load: {gpu.load * 100:.2f}%")
    print(f"  Free Memory: {gpu.memoryFree / 1024:.2f} MB")
    print(f"  Used Memory: {gpu.memoryUsed / 1024:.2f} MB")
    print(f"  Total Memory: {gpu.memoryTotal / 1024:.2f} MB")
    print(f"  Temperature: {gpu.temperature} °C")