import os
import numpy as np
from PIL import Image
from tqdm import tqdm

def remap_pixel_values(input_dir, output_dir, mapping):
    os.makedirs(output_dir, exist_ok=True)
    file_list = [f for f in os.listdir(input_dir) if f.endswith('.tif')]

    if not file_list:
        print(f"'{input_dir}'에 .tif 파일이 없습니다.")
        return

    print(f"'{input_dir}' 폴더의 파일을 변환합니다...")

    for filename in tqdm(file_list):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        try:
            with Image.open(input_path) as img:
                arr = np.array(img).astype(np.int64)

                remap_func = np.vectorize(mapping.get)
                new_arr = remap_func(arr)

                new_img = Image.fromarray(new_arr.astype(np.uint8), mode='L')
                new_img.save(output_path)
        except Exception as e:
            print(f"경고: '{filename}' 처리 중 오류: {e}")

# 픽셀 값 매핑 정의: 10 -> 0, 90 -> 1
pixel_mapping = {10: 1, 90: 0}

data_root = "../data/Mission3"

# 학습 데이터 라벨 변환
remap_pixel_values(
    os.path.join(data_root, 'Training/labels'),
    os.path.join(data_root, 'Training/labels_remapped'),
    pixel_mapping
)

# 검증 데이터 라벨 변환
remap_pixel_values(
    os.path.join(data_root, 'Validation/labels'),
    os.path.join(data_root, 'Validation/labels_remapped'),
    pixel_mapping
)
