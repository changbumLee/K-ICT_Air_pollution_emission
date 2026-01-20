import json
import os
import glob
from tqdm import tqdm
from PIL import Image

def process_labels_unified(original_labels_path, images_dir, output_root, split_name, mode):
    # 변환된 통합 라벨 파일 저장 경로
    unified_output_path = os.path.join(output_root, f'{split_name}/unified_labels/{mode}_unified.json')
    os.makedirs(os.path.dirname(unified_output_path), exist_ok=True)
    
    unified_data = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 0, "name": "chimney", "supercategory": "None"}]
    }
    annotation_id_counter = 1
    
    json_files = sorted(glob.glob(os.path.join(original_labels_path, '*.json')))
    print(f"'{split_name}' 데이터 변환 중 ({len(json_files)}개 파일)...")

    for json_path in tqdm(json_files):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        for _, info in data.items():
            file_name = info['filename']
            img_path = os.path.join(images_dir, file_name)
            
            if not os.path.exists(img_path):
                continue
            
            # 이미지 크기 불러오기
            try:
                with Image.open(img_path) as img:
                    img_width = img.width
                    img_height = img.height
            except Exception as e:
                print(f"경고: 이미지 파일 '{file_name}' 열기 실패: {e}. 건너뜁니다.")
                continue

            # 이미지 고유 ID (파일명 해시)
            image_id = hash(file_name)

            # 이미지 정보 저장
            unified_data["images"].append({
                "id": image_id,
                "file_name": file_name,
                "width": img_width,
                "height": img_height
            })
            
            # 라벨 정보 변환
            for region in info.get('regions', []):
                if region['shape_attributes'].get('name') == 'polyline':
                    points_x = region['shape_attributes']['all_points_x']
                    points_y = region['shape_attributes']['all_points_y']
                    
                    x_min = min(points_x)
                    y_min = min(points_y)
                    x_max = max(points_x)
                    y_max = max(points_y)
                    
                    bbox_w = x_max - x_min
                    bbox_h = y_max - y_min
                    x_center = x_min + bbox_w / 2
                    y_center = y_min + bbox_h / 2
                    
                    # 바운딩 박스를 정규화 [x_center, y_center, width, height]
                    bbox = [x_center / img_width, y_center / img_height, bbox_w / img_width, bbox_h / img_height]
                    
                    height = float(region['region_attributes']['chi_height_m'])
                    
                    unified_data["annotations"].append({
                        "id": annotation_id_counter,
                        "image_id": image_id,
                        "category_id": 0,
                        "bbox": bbox,
                        "height_m": height,
                        "area": bbox[2] * bbox[3],
                        "iscrowd": 0
                    })
                    annotation_id_counter += 1

    # JSON 파일 저장
    with open(unified_output_path, 'w', encoding='utf-8') as f:
        json.dump(unified_data, f, indent=4)
    print(f"통합 JSON 변환 완료: {unified_output_path}")


if __name__ == '__main__':
    base_data_path = '../data/Mission2'
    process_labels_unified(
        os.path.join(base_data_path, 'Training/labels'),
        os.path.join(base_data_path, 'Training/images'),
        base_data_path, 'Training', mode='train'
    )
    print("-" * 30)
    process_labels_unified(
        os.path.join(base_data_path, 'Validation/labels'),
        os.path.join(base_data_path, 'Validation/images'),
        base_data_path, 'Validation', mode='val'
    )