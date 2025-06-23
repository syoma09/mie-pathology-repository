import os
import pandas as pd
from pathlib import Path

def update_patchlist_paths(patchlist_path, updated_patchlist_path):
    df = pd.read_csv(patchlist_path)
    df['path'] = df['path'].str.replace('/net/nfs2/', '/net/nfs3/')
    df.to_csv(updated_patchlist_path, index=False)
    print(f"Updated patchlist saved to: {updated_patchlist_path}")

def process_patchlists(src_dir):
    # src_dir内のすべてのpatchlist.csvファイルを取得
    patchlist_files = list(Path(src_dir).rglob("patchlist.csv"))

    for patchlist_file in patchlist_files:
        # 保存先のパスを設定
        updated_patchlist_path = patchlist_file.parent / "patchlist_updated.csv"
        
        # パッチリストの更新
        update_patchlist_paths(patchlist_file, updated_patchlist_path)

def main():
    src_dir = "/net/nfs3/export/home/sakakibara/data/_out/mie-pathology/"
    process_patchlists(src_dir)

if __name__ == '__main__':
    main()