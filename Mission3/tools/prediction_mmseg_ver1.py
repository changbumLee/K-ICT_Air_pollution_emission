# predict_mmseg.py (통합 버전)

import argparse
import os
import mmcv
from mmengine.model.utils import revert_sync_batchnorm
from mmengine.registry import init_default_scope
from mmseg.apis import inference_model, init_model
from tqdm import tqdm

def find_all_images(img_dir):
    """
    지정된 디렉토리와 모든 하위 디렉토리에서 이미지 파일을 찾습니다.
    (기존 prediction.py의 glob 로직과 동일한 기능)
    """
    image_paths = []
    for root, _, files in os.walk(img_dir):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                image_paths.append(os.path.join(root, file))
    return sorted(image_paths)

def main():
    parser = argparse.ArgumentParser(description='MMSegmentation Inference Script')
    parser.add_argument('config', help='Config file path')
    parser.add_argument('checkpoint', help='Checkpoint file path')
    parser.add_argument('img_dir', help='Directory where images are saved')
    parser.add_argument('out_dir', help='Directory where results are saved')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()

    init_default_scope('mmseg')

    print("Initializing model...")
    model = init_model(args.config, args.checkpoint, device=args.device)
    if args.device == 'cpu':
        model = revert_sync_batchnorm(model)
    print("Model loaded successfully.")

    # [수정] 하위 폴더의 모든 이미지를 재귀적으로 탐색
    image_paths_to_process = find_all_images(args.img_dir)
    
    print(f"Found {len(image_paths_to_process)} images in {args.img_dir} and its subdirectories.")

    for img_path in tqdm(image_paths_to_process, desc="Running inference"):
        result = inference_model(model, img_path)
        
        # [수정] 입력 폴더 구조를 출력 폴더에 그대로 복제
        # (기존 prediction.py의 rel_path 로직과 동일한 기능)
        relative_path = os.path.relpath(img_path, args.img_dir)
        output_path = os.path.join(args.out_dir, relative_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        pred_mask = result.pred_sem_seg.data.squeeze().cpu().numpy()
        mmcv.imwrite(pred_mask, output_path)

    print(f"\nInference complete. Results are saved in {args.out_dir}")

if __name__ == '__main__':
    main()
