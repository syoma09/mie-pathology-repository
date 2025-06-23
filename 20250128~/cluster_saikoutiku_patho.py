import pandas as pd
import cv2
import numpy as np

# CSVファイルの読み込み
csv_file = '/net/nfs3/export/home/sakakibara/data/_out/log_root/uniencoder3_clustering_syuusei_20250131_120005/57-10_clustered_patches.csv'
df = pd.read_csv(csv_file)

# クラスタごとの色を設定
cluster_colors = {
    0.0: (255, 0, 0),    # 赤
    1.0: (0, 255, 0),    # 緑
    2.0: (0, 0, 255),    # 青
    3.0: (255, 255, 0),  # 黄色
    4.0: (0, 255, 255),  # シアン
    5.0: (255, 0, 255),  # マゼンタ
    6.0: (128, 0, 128),  # 紫
    7.0: (255, 165, 0),  # オレンジ
    8.0: (0, 128, 128),  # ティール
    9.0: (128, 128, 0)   # オリーブ
}

# 透過度の設定
alpha = 0.2  # 透過度を調整してパッチ画像が見えるようにする

# パッチ画像のサイズ
original_patch_size = 256
target_patch_size = 512

# 元画像のサイズを計算
max_x = int(df['x'].max() + target_patch_size)
max_y = int(df['y'].max() + target_patch_size)

# 元画像を作成
composite_image = np.zeros((max_y, max_x + 300, 3), dtype=np.uint8)  # 左側にスペースを追加

# クラスタごとの色と番号の対応を表示
legend_height = 30 * len(cluster_colors)
legend_image = np.zeros((legend_height, 300, 3), dtype=np.uint8)
for i, (cluster, color) in enumerate(cluster_colors.items()):
    cv2.rectangle(legend_image, (10, 10 + i * 30), (30, 30 + i * 30), color, -1)
    cv2.putText(legend_image, f'Cluster {int(cluster)}', (40, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

# レジェンドを元画像に追加
composite_image[:legend_height, :300] = legend_image

# パッチ画像を元画像に重ね合わせる
for index, row in df.iterrows():
    x, y, width, height, path, kmeans_cluster, dbscan_cluster = row
    if pd.isna(path):
        continue
    path = str(path)  # パスを文字列に変換
    patch_image = cv2.imread(path)

    if patch_image is None:
        continue

    # パッチ画像を512x512に拡大
    patch_image = cv2.resize(patch_image, (target_patch_size, target_patch_size))

    # クラスタごとの色を取得
    color = cluster_colors[kmeans_cluster]

    # パッチ画像に色を重ねる
    overlay = np.full(patch_image.shape, color, dtype=np.uint8)
    cv2.addWeighted(overlay, alpha, patch_image, 1 - alpha, 0, patch_image)

    # クラスタリング結果を表示
    cv2.putText(patch_image, f'Cluster {int(kmeans_cluster)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # 元画像にパッチ画像を重ねる
    x, y = int(x), int(y)
    composite_image[y:y+target_patch_size, x+300:x+target_patch_size+300] = patch_image  # 左側のスペースを考慮して配置

# 画像全体を縮小
scale_factor = 0.1  # 縮小率を設定
resized_image = cv2.resize(composite_image, (int((max_x + 300) * scale_factor), int(max_y * scale_factor)))

# 結果を保存
output_path = '/net/nfs3/export/home/sakakibara/root/workspace/mie-pathology-repository/20250128~/clustered_image.png'
cv2.imwrite(output_path, resized_image)
print(f"Clustered image saved to: {output_path}")