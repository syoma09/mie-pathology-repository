import os
import pandas as pd

def update_patchlist_paths(patient_number):
    patchlist_path = f"/net/nfs3/export/home/sakakibara/data/_out/mie-pathology/{patient_number}/patchlist/patchlist.csv"
    updated_patchlist_path = f"/net/nfs3/export/home/sakakibara/data/_out/mie-pathology/{patient_number}/patchlist/patchlist_updated.csv"
    
    if os.path.exists(patchlist_path):
        # CSVファイルを読み込む
        df = pd.read_csv(patchlist_path)
        
        # path列のnfs2をnfs3に置換
        df['path'] = df['path'].str.replace('/net/nfs2/', '/net/nfs3/')
        
        # 更新されたCSVファイルを保存
        df.to_csv(updated_patchlist_path, index=False)
        print(f"Updated patchlist saved for patient {patient_number}")
    else:
        print(f"Patchlist not found for patient {patient_number}")

def main():
    # 患者番号のリストを取得
    patients_df = pd.read_csv("_data/survival_time_cls/20220726_cls.csv")
    patient_numbers = patients_df['number'].astype(str).tolist()

    # 各患者のpatchlistを更新
    for patient_number in patient_numbers:
        update_patchlist_paths(patient_number)

if __name__ == '__main__':
    main()