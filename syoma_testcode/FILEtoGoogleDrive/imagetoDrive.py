from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import os

# Google Drive APIのスコープ
SCOPES = ['https://www.googleapis.com/auth/drive.file']

def authenticate():
    # 認証情報を格納する変数
    creds = None
    # 既存のトークンファイルが存在する場合、それを読み込む
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    # 認証情報が無効または存在しない場合、新たに認証を行う
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            # トークンが期限切れの場合、リフレッシュする
            creds.refresh(Request())
        else:
            # 新たに認証を行う
            flow = InstalledAppFlow.from_client_secrets_file(
                'syoma_testcode/client_secret_605530777392-69iospj4t87ucsbda2hj0e30e4f29v6e.apps.googleusercontent.com.json', SCOPES)
            creds = flow.run_local_server(port=8080)
        # 新しいトークンをファイルに保存する
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    return creds

def upload_file_to_drive(service, filename, folder_id):
    # アップロードするファイルのメタデータを設定する
    file_metadata = {'name': os.path.basename(filename)}
    if folder_id:
        file_metadata['parents'] = [folder_id]

    # ファイルをアップロードするためのMediaFileUploadオブジェクトを作成する
    media = MediaFileUpload(filename, resumable=True)
    # ファイルをGoogle Driveにアップロードする
    file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
    # アップロードしたファイルのIDを出力する
    print(f'File ID: {file.get("id")}')

def create_folder_in_drive(service, folder_name, parent_folder_id=None):
    # フォルダのメタデータを設定する
    folder_metadata = {
        'name': folder_name,
        'mimeType': 'application/vnd.google-apps.folder'
    }
    if parent_folder_id:
        folder_metadata['parents'] = [parent_folder_id]

    # フォルダを作成する
    folder = service.files().create(body=folder_metadata, fields='id').execute()
    return folder.get('id')

def upload_folder_to_drive(service, folder_path, parent_folder_id=None):
    # フォルダを作成する
    folder_id = create_folder_in_drive(service, os.path.basename(folder_path), parent_folder_id)
    
    # フォルダ内のすべてのファイルとサブフォルダを再帰的にアップロードする
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            upload_file_to_drive(service, file_path, folder_id)
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            upload_folder_to_drive(service, dir_path, folder_id)

if __name__ == '__main__':
    # アップロードするフォルダパス
    folder_path = '/net/nfs3/export/home/sakakibara/data/_out/mie-pathology/1-2'
    # フォルダID
    folder_id = '1BOcjBkJ7sBlywboQYgRrYGbUQSW8Welw'
    
    # 認証を行い、Google Drive APIサービスを構築する
    creds = authenticate()
    service = build('drive', 'v3', credentials=creds)
    
    # フォルダをGoogle Driveにアップロードする
    upload_folder_to_drive(service, folder_path, folder_id)
