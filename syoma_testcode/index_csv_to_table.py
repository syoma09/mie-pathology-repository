import pandas as pd
import matplotlib.pyplot as plt

# CSVファイルのパス
csv_file_path1 = '/net/nfs3/export/home/sakakibara/root/workspace/mie-pathology-repository/syoma_testcode/log_csv/020240925_045226 (1).csv'
csv_file_path2 = '/net/nfs3/export/home/sakakibara/root/workspace/mie-pathology-repository/syoma_testcode/log_csv/20240708_151952 (1).csv'
csv_file_path3 = '/net/nfs3/export/home/sakakibara/root/workspace/mie-pathology-repository/syoma_testcode/log_csv/020240925_045226 (2).csv'
csv_file_path4 = '/net/nfs3/export/home/sakakibara/root/workspace/mie-pathology-repository/syoma_testcode/log_csv/20240708_151952 (3).csv'

# CSVファイルの読み込み
df1 = pd.read_csv(csv_file_path1)
df2 = pd.read_csv(csv_file_path2)
df3 = pd.read_csv(csv_file_path3)
df4 = pd.read_csv(csv_file_path4)

# 最低値の計算
min_value1 = df1['Value'].min()
min_step1 = df1.loc[df1['Value'].idxmin(), 'Step']

min_value2 = df2['Value'].min()
min_step2 = df2.loc[df2['Value'].idxmin(), 'Step']

min_value3 = df3['Value'].min()
min_step3 = df3.loc[df3['Value'].idxmin(), 'Step']

min_value4 = df4['Value'].min()
min_step4 = df4.loc[df4['Value'].idxmin(), 'Step']

# プロットの作成
plt.figure(figsize=(10, 6))

# データセット1のプロット
plt.plot(df1['Step'], df1['Value'], marker='o', linestyle='-', markersize=1, label='Dataset 1')
# データセット2のプロット
plt.plot(df2['Step'], df2['Value'], marker='x', linestyle='-', markersize=1, label='Dataset 2')
# データセット3のプロット
plt.plot(df3['Step'], df3['Value'], marker='s', linestyle='-', markersize=1, label='Dataset 3')
# データセット4のプロット
plt.plot(df4['Step'], df4['Value'], marker='d', linestyle='-', markersize=1, label='Dataset 4')

plt.xlabel('Step', fontsize=14)
plt.ylabel('Value', fontsize=14)
plt.title('MAE', fontsize=16)
plt.grid(True)

# Stepを100ごとに目盛りを表示
plt.xticks(ticks=range(0, max(df1['Step'].max(), df2['Step'].max(), df3['Step'].max(), df4['Step'].max()) + 1, 100))

# 最低値をプロットの右上に表示（小数第3位まで）
plt.text(df1['Step'].max(), min_value1, f'Min (Dataset 1): {min_value1:.3f}', fontsize=14, ha='right', va='bottom')
plt.text(df2['Step'].max(), min_value2, f'Min (Dataset 2): {min_value2:.3f}', fontsize=14, ha='right', va='top')
plt.text(df3['Step'].max(), min_value3, f'Min (Dataset 3): {min_value3:.3f}', fontsize=14, ha='right', va='bottom')
plt.text(df4['Step'].max(), min_value4, f'Min (Dataset 4): {min_value4:.3f}', fontsize=14, ha='right', va='top')

# 凡例を追加
plt.legend()

plt.show()