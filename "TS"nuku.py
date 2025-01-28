import pandas as pd

# 元のCSVファイルを読み込む
input_file = "input.csv"  # 入力ファイル名を指定
output_file = "output.csv"  # 出力ファイル名を指定

# CSVをデータフレームとして読み込む
df = pd.read_csv(input_file)

# "number"列に"TS"を含む行を削除
df_filtered = df[~df['number'].str.contains("TS", na=False)]

# 新しいCSVファイルとして保存
df_filtered.to_csv(output_file, index=False)

print(f"Filtered data saved to {output_file}")
