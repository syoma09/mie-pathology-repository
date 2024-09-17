import openslide
from PIL import Image, ImageDraw
import xml.etree.ElementTree as ET

# ファイルパス
xml_file_path = '/net/nfs2/export/dataset/morita/mie-u/orthopedic/AIPatho/xml/61-6.xml'
svs_file_path = '/net/nfs2/export/dataset/morita/mie-u/orthopedic/AIPatho/svs/61-6.svs'
output_image_path = '/net/nfs2/export/home/sakakibara/root/workspace/mie-pathology-repository/syoma_testcode/low_quality_annotated_image_61-6.png'

# XMLファイルの読み込みとアノテーションの抽出
def parse_annotations(xml_file):
    print(f"Parsing annotations from XML file: {xml_file}")
    tree = ET.parse(xml_file)
    root = tree.getroot()
    annotations = {}

    for annotation in root.findall('Annotation'):
        annotation_id = annotation.get('Id')
        color = annotation.get('LineColor', '0')  # デフォルトは黒
        color = '#' + format(int(color), '06x')  # 色コードをRGB形式に変換

        print(f"Found annotation with ID: {annotation_id}, color: {color}")

        vertices = []
        for region in annotation.findall('Regions/Region'):
            for vertex in region.findall('Vertices/Vertex'):
                x = float(vertex.get('X'))
                y = float(vertex.get('Y'))
                vertices.append((x, y))
        
        annotations[annotation_id] = {
            'type': 'polygon',
            'vertices': vertices,
            'color': color
        }
    
    print(f"Parsed {len(annotations)} annotations")
    return annotations

# アノテーション情報の解析
annotations = parse_annotations(xml_file_path)

# SVSファイルの読み込み
try:
    print(f"Opening SVS file: {svs_file_path}")
    slide = openslide.OpenSlide(svs_file_path)
except Exception as e:
    print(f"Error opening slide file: {e}")
    raise

# 画像レベルを設定（通常は0が最も高解像度）
level = 0

# スライド画像のサイズを取得
try:
    width, height = slide.dimensions
    print(f"Original slide dimensions: width={width}, height={height}")
except Exception as e:
    print(f"Error getting slide dimensions: {e}")
    raise

# 解像度を大幅に下げるためのスケーリングファクター
scale_factor = 0.01  # 1%に縮小
print(f"Scaling factor: {scale_factor}")

# スライド画像のダウンサンプリングを行い、アノテーションを描画する
def process_image(level, scale_factor, annotations):
    print(f"Processing image with scale factor: {scale_factor}")
    
    # スライド全体をダウンサンプリングして読み込み
    try:
        img = slide.read_region((0, 0), level, (width, height))
        img = img.convert("RGB")  # 画像をRGB形式に変換
        print(f"Read region from slide, size: {img.size}")
    except Exception as e:
        print(f"Error reading region from slide: {e}")
        raise

    # 画像サイズを縮小
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    img = img.resize((new_width, new_height), Image.Resampling.NEAREST)
    print(f"Resized image to: {new_width}x{new_height}")

    # アノテーションを描画
    draw = ImageDraw.Draw(img)
    for annotation_id, annotation in annotations.items():
        if annotation['type'] == 'polygon':
            # 解像度に合わせてアノテーションの座標を変換
            scaled_vertices = [(int(x * scale_factor), int(y * scale_factor)) for (x, y) in annotation['vertices']]
            draw.polygon(scaled_vertices, outline=annotation['color'])
            print(f"Drew annotation ID {annotation_id} with {len(scaled_vertices)} vertices")

    return img

# 低画質の画像を作成して保存
low_quality_img = process_image(level, scale_factor, annotations)
try:
    low_quality_img.save(output_image_path, format='PNG')
    print(f"Low quality annotated image saved to {output_image_path}")
except Exception as e:
    print(f"Error saving image: {e}")
    raise
