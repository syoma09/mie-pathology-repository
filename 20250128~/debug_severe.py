import os
import pandas as pd
import xml.etree.ElementTree as ET
from pathlib import Path
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
    return (intersection_area / patch_area) >= threshold, intersection_area, patch_area

def debug_patient(patient_number, src_dir, dst_dir):
    xml_path = Path(src_dir) / f"{patient_number}.xml"
    patchlist_path = Path(dst_dir) / patient_number / "patchlist" / "patchlist_updated.csv"
    output_path = Path(dst_dir) / patient_number / "patchlist" / "patchlist_severe_debug.csv"

    if not xml_path.exists() or not patchlist_path.exists():
        print(f"Skipping {patient_number}: XML or patchlist file not found.")
        return

    regions = parse_xml(xml_path)
    if not regions["2"]:
        print(f"No severe regions found for {patient_number}.")
        return

    print(f"Severe regions for {patient_number}: {regions['2']}")

    patchlist = pd.read_csv(patchlist_path)
    debug_info = []

    for _, row in patchlist.iterrows():
        is_severe, intersection_area, patch_area = is_within_region(row["x"], row["y"], row["width"], row["height"], regions["2"])
        debug_info.append({
            "x": row["x"],
            "y": row["y"],
            "width": row["width"],
            "height": row["height"],
            "severe": 1 if is_severe else 0,
            "intersection_area": intersection_area,
            "patch_area": patch_area,
            "intersection_ratio": intersection_area / patch_area
        })

    debug_df = pd.DataFrame(debug_info)
    debug_df.to_csv(output_path, index=False)
    print(f"Debug information saved to {output_path}")

def main():
    annotation_dir = "/net/nfs3/export/dataset/morita/mie-u/orthopedic/AIPatho/layer12"
    patchlist_dir = "/net/nfs3/export/home/sakakibara/data/_out/mie-pathology"
    patient_number = "60-8"
    debug_patient(patient_number, annotation_dir, patchlist_dir)

if __name__ == "__main__":
    main()