import json
import os
from glob import glob
from tqdm import tqdm

def process_and_convert_to_coco(json_folder_path, split_name):
    categories = [
        {"id": 0, "name": "smokestack", "supercategory": "factory"}
    ]
    
    images_info = []
    annotations_info = []
    
    annotation_id_counter = 1
    processed_image_ids = set()
    
    json_files = sorted(glob(os.path.join(json_folder_path, '*.json')))
    
    if not json_files:
        print(f"경고: '{json_folder_path}' 폴더에 JSON 파일이 없습니다.")
        return None

    print(f"'{split_name}' 데이터셋 변환 중 ({len(json_files)}개 파일)...")
    for json_path in tqdm(json_files):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for key in data:
            image_data = data[key]
            
            try:
                file_name = image_data['filename']
                img_width = int(image_data['file_attributes']['img_width'])
                img_height = int(image_data['file_attributes']['img_height'])
                original_img_id_str = image_data["file_attributes"]["img_id"]


                # "K3_CHN_20130308050237_8" -> 201303080502378
                numeric_part = ''.join(filter(str.isdigit, original_img_id_str))
                current_image_id = int(numeric_part)
                
            except (KeyError, ValueError, TypeError):
                print(f"경고: '{file_name}' 파일의 필수 정보가 유효하지 않아 건너뜁니다.")
                continue


            if current_image_id not in processed_image_ids:
                images_info.append({
                    "id": current_image_id,
                    "file_name": file_name,
                    "width": img_width,
                    "height": img_height
                })
                processed_image_ids.add(current_image_id)
            
 
            for region in image_data.get('regions', []):
                shape = region['shape_attributes']
                if shape.get('name') == 'rect':
                    x, y, w, h = shape['x'], shape['y'], shape['width'], shape['height']
                    
                    annotations_info.append({
                        "id": annotation_id_counter,
                        "image_id": current_image_id,
                        "category_id": 0, # 모든 객체는 'smokestack' (id: 0)
                        "bbox": [x, y, w, h],
                        "area": w * h,
                        "iscrowd": 0,
                        "segmentation": []
                    })
                    annotation_id_counter += 1
            

    return {
        "info": {
            "description": f"Mission1 {split_name} Dataset",
            "version": "1.0",
        },
        "licenses": [],
        "categories": categories,
        "images": images_info,
        "annotations": annotations_info
    }


if __name__ == '__main__':
    #  사용자 설정: 이 부분의 경로를 자신의 환경에 맞게 수정하세요.
    base_path = '../data/Mission1'
    output_dir = os.path.join(base_path, 'annotations')
    os.makedirs(output_dir, exist_ok=True)
    
    # --- 훈련 데이터 변환 및 저장 ---
    train_labels_folder = os.path.join(base_path, 'Training', 'labels')
    train_coco_data = process_and_convert_to_coco(train_labels_folder, 'Training')
    
    if train_coco_data:
        train_output_path = os.path.join(output_dir, 'train.json')
        with open(train_output_path, 'w', encoding='utf-8') as f:
            json.dump(train_coco_data, f, indent=4)
        print(f"훈련 데이터 COCO 변환 완료: {train_output_path}")

    print("-" * 30)

    # --- 검증 데이터 변환 및 저장 ---
    val_labels_folder = os.path.join(base_path, 'Validation', 'labels')
    val_coco_data = process_and_convert_to_coco(val_labels_folder, 'Validation')
    
    if val_coco_data:
        val_output_path = os.path.join(output_dir, 'val.json')
        with open(val_output_path, 'w', encoding='utf-8') as f:
            json.dump(val_coco_data, f, indent=4)
        print(f"검증 데이터 COCO 변환 완료: {val_output_path}")

