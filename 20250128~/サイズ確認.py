import pandas as pd
from pathlib import Path
from PIL import Image

# CSVファイルのパス
csv_path = "/net/nfs3/export/home/sakakibara/root/workspace/mie-pathology-repository/_data/survival_time_cls/20220413_aut2.csv"
# パッチ画像のルートディレクトリ
patch_root = Path("/net/nfs3/export/home/sakakibara/data/_out/mie-pathology/")

# CSV読み込み
df = pd.read_csv(csv_path)

# tvtが3以外のnumberを抽出
slide_ids = df[df['tvt'] != 3]['number'].astype(str).tolist()

for slide_id in slide_ids:
    slide_dir = patch_root / slide_id
    if not slide_dir.exists():
        print(f"{slide_id}: ディレクトリが存在しません")
        continue
    # ディレクトリ内のpngファイルを1枚取得
    patch_files = list(slide_dir.glob("*.png"))
    if not patch_files:
        print(f"{slide_id}: パッチ画像が見つかりません")
        continue
    patch_path = patch_files[0]
    try:
        img = Image.open(patch_path)
        print(f"{slide_id}: {patch_path.name} サイズ={img.size}")
    except Exception as e:
        print(f"{slide_id}: 画像読み込みエラー {e}")