#xmlのアノテーション情報をsvs画像に描画。→Annotation_svsへ保存 layer12のxmlファイルを対象,外が腫瘍全体、内が高悪性度
import openslide
import xml.etree.ElementTree as ET
import numpy as np
import cv2
from PIL import Image
import torch
import os
from pathlib import Path

# 入力ディレクトリと出力ディレクトリのパス
input_dir = Path('/net/nfs3/export/dataset/morita/mie-u/orthopedic/AIPatho/layer12/')
output_dir = Path('/net/nfs3/export/home/sakakibara/root/workspace/mie-pathology-repository/Annotation_svs/')
os.makedirs(output_dir, exist_ok=True)

# デバイスの設定
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# アノテーション情報を取得する関数
def get_annotations(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    annotations = []
    for region in root.findall('.//Region'):
        vertices = []
        for vertex in region.findall('.//Vertex'):
            x = float(vertex.get('X'))
            y = float(vertex.get('Y'))
            vertices.append((x, y))
        annotations.append(vertices)
    return annotations

# 対応するSVSファイルとXMLファイルを処理する
for xml_file in input_dir.glob('*.xml'):
    svs_file = input_dir / (xml_file.stem + '.svs')
    if not svs_file.exists():
        continue

    # SVSファイルを読み込む
    slide = openslide.OpenSlide(svs_file)

    # アノテーション情報を取得する
    annotations = get_annotations(xml_file)

    # 画像のダウンサンプリング倍率
    downsample_factor = 32

    # ダウンサンプリングした画像を取得する
    level = slide.get_best_level_for_downsample(downsample_factor)
    downsampled_image = slide.read_region((0, 0), level, slide.level_dimensions[level])
    downsampled_image = np.array(downsampled_image)[:, :, :3]

    # ダウンサンプリング倍率を計算
    actual_downsample = slide.level_downsamples[level]

    # 画像をCUDAに転送
    downsampled_image_gpu = torch.tensor(downsampled_image).to(device)

    # アノテーションを描画する
    for vertices in annotations:
        vertices = [(int(x / actual_downsample), int(y / actual_downsample)) for x, y in vertices]
        vertices_np = np.array(vertices, dtype=np.int32)
        vertices_gpu = torch.tensor(vertices_np).to(device)
        # OpenCVのpolylines関数はCUDAをサポートしていないため、CPUで処理します
        downsampled_image_cpu = downsampled_image_gpu.cpu().numpy()
        cv2.polylines(downsampled_image_cpu, [vertices_np], isClosed=True, color=(0, 0, 255), thickness=5)  # 青色で太めに描画
        downsampled_image_gpu = torch.tensor(downsampled_image_cpu).to(device)

    # 画像をCPUに戻してPNG形式で保存する
    downsampled_image = downsampled_image_gpu.cpu().numpy()
    output_path = output_dir / f'{xml_file.stem}.png'
    Image.fromarray(downsampled_image).save(output_path)

    print(f'アノテーション付き画像を {output_path} に保存しました。')