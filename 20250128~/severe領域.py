#高悪性度領域にsevere=1としてラベル付けするスクリプト3/17
import os
import pandas as pd
import xml.etree.ElementTree as ET
from pathlib import Path
from tqdm import tqdm
from shapely.geometry import Polygon, box

def parse_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    regions = {"1": [], "2": []}
    for annotation in root.findall(".//Annotation"):
        annotation_id = annotation.get("Id")
        if annotation_id in regions:
            for region in annotation.findall(".//Region"):
                vertices = []
                for vertex in region.findall(".//Vertex"):
                    x = float(vertex.get("X"))
                    y = float(vertex.get("Y"))
                    vertices.append((x, y))
                regions[annotation_id].append(Polygon(vertices))
    return regions

def calculate_intersection_area(patch_box, regions):
    intersection_area = 0
    for region in regions:
        if patch_box.intersects(region):
            intersection_area += patch_box.intersection(region).area
    return intersection_area

def is_within_region(x, y, width, height, regions, threshold=0.5):
    patch_box = box(x, y, x + width, y + height)
    patch_area = patch_box.area
    intersection_area = calculate_intersection_area(patch_box, regions)
    return (intersection_area / patch_area) >= threshold

def process_patient(patient_number, src_dir, dst_dir):
    xml_path = Path(src_dir) / f"{patient_number}.xml"
    patchlist_path = Path(dst_dir) / patient_number / "patchlist" / "patchlist_updated.csv"
    output_path = Path(dst_dir) / patient_number / "patchlist" / "patchlist_severe.csv"

    if not xml_path.exists() or not patchlist_path.exists():
        print(f"Skipping {patient_number}: XML or patchlist file not found.")
        return

    regions = parse_xml(xml_path)
    if not regions["2"]:
        print(f"No severe regions found for {patient_number}.")
        return

    print(f"Severe regions for {patient_number}: {regions['2']}")

    patchlist = pd.read_csv(patchlist_path)
    patchlist["severe"] = patchlist.apply(
        lambda row: 1 if is_within_region(row["x"], row["y"], row["width"], row["height"], regions["2"]) else 0,
        axis=1
    )
    patchlist.to_csv(output_path, index=False)
    
    # 1となったパッチの数と全パッチ枚数の比率を表示
    severe_count = patchlist["severe"].sum()
    total_patches = len(patchlist)
    ratio = severe_count / total_patches
    print(f"Processed {patient_number}: saved to {output_path}, severe patches: {severe_count}/{total_patches} ({ratio:.2%})")

    # annotation_id=2の面積がannotation_id=1の面積の何%になっているかを計算
    area_1 = sum(region.area for region in regions["1"])
    area_2 = sum(region.area for region in regions["2"])

    if area_1 == 0:
        print(f"Skipping {patient_number}: Area of annotation_id=1 is zero.")
        return

    percentage = (area_2 / area_1) * 100
    print(f"Patient {patient_number}: annotation_id=2 is {percentage:.2f}% of annotation_id=1")

def main():
    annotation_dir = "/net/nfs3/export/dataset/morita/mie-u/orthopedic/AIPatho/layer12"
    patchlist_dir = "/net/nfs3/export/home/sakakibara/data/_out/mie-pathology"
    patients_df = pd.read_csv("_data/survival_time_cls/20220726_cls.csv")
    patient_numbers = patients_df["number"].astype(str).tolist()

    for patient_number in tqdm(patient_numbers, desc="Processing patients"):
        process_patient(patient_number, annotation_dir, patchlist_dir)

if __name__ == "__main__":
    main()