import os
import shutil

def move_log_file(log_file_path):
    # ログファイル名を取得
    log_file_name = os.path.basename(log_file_path)
    
    # ディレクトリ名をログファイル名から抽出
    dir_name = log_file_name.split('.')[-1]
    
    # ディレクトリのパスを作成
    target_dir = os.path.join(os.path.dirname(log_file_path), dir_name)
    
    # ディレクトリが存在しない場合は作成
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    # ログファイルをディレクトリに移動
    shutil.move(log_file_path, os.path.join(target_dir, log_file_name))
    print(f"Moved {log_file_name} to {target_dir}")

def move_all_logs(logs_dir):
    # 指定されたディレクトリ内のすべてのファイルを検索
    for file_name in os.listdir(logs_dir):
        file_path = os.path.join(logs_dir, file_name)
        # ファイルかどうかを確認
        if os.path.isfile(file_path):
            move_log_file(file_path)

if __name__ == "__main__":
    # ログファイルが保存されているディレクトリを指定
    logs_dir = "/net/nfs3/export/home/sakakibara/root/workspace/mie-pathology-repository/logs"
    
    # すべてのログファイルを移動
    move_all_logs(logs_dir)