# mmseg/datasets/cityscapes_drive2.py

import os.path as osp
import os
from glob import glob
from typing import List
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class Drive2Dataset(BaseSegDataset):
    METAINFO = dict(
        # classes=('주행가능영역', '인도', '로드마크', '차선', '연석', 
        #          '벽,울타리', '승용차', '트럭', '버스', '바이크, 자전거', 
        #          '기타 차량', '보행자', '라이더', '교통용 콘 및 봉', '기타 수직 물체', 
        #          '건물', '교통 표지', '교통 신호', '기타'),
        classes=('Drivable Area', 'Sidewalk', 'Road Mark', 'Lane', 'Curb', 
                'Wall/Fence', 'Car', 'Truck', 'Bus', 'Bike/Bicycle', 
                'Other Vehicle', 'Pedestrian', 'Rider', 'Traffic Cone/Pole', 'Other Vertical Object', 
                'Building', 'Traffic Sign', 'Traffic Light', 'Other'),
        palette=[[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156], 
                 [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0], 
                 [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60], 
                 [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100], 
                 [0, 80, 100], [0, 0, 230], [119, 11, 32]]
    )

    def __init__(self, **kwargs) -> None:
        # 이 코드는 config 파일의 img_suffix와 seg_map_suffix를 그대로 사용합니다.
        # 또한, BaseSegDataset가 data_prefix를 단일 문자열로 처리한다고 가정합니다.
        super().__init__(**kwargs)

    def _img_to_seg_map(self, img_path: str) -> str:
        """이미지 경로를 세그멘테이션 맵 경로로 변환합니다."""
        
        # BaseSegDataset의 self.data_prefix는 dict 형태입니다.
        img_prefix = self.data_prefix['img_path']
        ann_prefix = self.data_prefix['seg_map_path']
        
        # 이미지 경로에서 이미지 파일이 속한 폴더의 상대 경로를 추출
        relative_folder = osp.relpath(osp.dirname(img_path), osp.join(self.data_root, img_prefix))
        
        file_name, ext = osp.splitext(osp.basename(img_path))
        
        # 파일명 규칙에 따라 라벨 파일명 생성
        if file_name.endswith('_leftImg8bit'):
            # Daeduk_..._leftImg8bit.png -> Daeduk_..._gtFine_CategoryId.png
            # 번호가 1 감소하는 규칙을 반영
            parts = file_name.split('_')
            num_str = parts[1]
            try:
                num = int(num_str)
                seg_name = f'{parts[0]}_{num:06d}_gtFine_CategoryId.png'
            except (ValueError, IndexError):
                # 번호 형식이 잘못된 경우를 대비한 대체
                seg_name = file_name.replace('_leftImg8bit', '_gtFine_CategoryId') + '.png'
            
        elif ext == '.jpg':
            # round(...)_camera(...).jpg -> round(...)_CategoryId.png
            # .jpg 확장자를 제거하고 _CategoryId.png를 추가
            seg_name = file_name + '_CategoryId.png'
        else:
            seg_name = file_name + '_CategoryId.png'

        # 최종 라벨 경로를 구성 (절대 경로)
        # 이미지 폴더의 상대 경로를 라벨 폴더 경로에 결합
        return osp.join(self.data_root, ann_prefix, relative_folder, seg_name)

    def load_data_list(self) -> List[dict]:
        """Loads all image and segmentation map pairs from the specified data_prefix."""
        img_dir = osp.join(self.data_root, self.data_prefix['img_path'])
        
        data_list = []
        
        # glob을 사용하여 하위 디렉토리를 재귀적으로 탐색
        all_img_files = []
        for suffix in self.img_suffix:
            # **를 사용하여 모든 하위 폴더의 파일을 찾습니다.
            all_img_files.extend(glob(osp.join(img_dir, '**', f'*{suffix}'), recursive=True))

        for img_path in all_img_files:
            seg_map_path = self._img_to_seg_map(img_path)
            
            # 라벨 파일의 존재 여부 확인
            if osp.exists(seg_map_path):
                data_info = dict(
                    img_path=img_path,
                    seg_map_path=seg_map_path,
                    seg_fields=[], 
                    reduce_zero_label=self.reduce_zero_label)
            
            data_list.append(data_info)
        
        print(f"Loaded {len(data_list)} image-label pairs.")
        return data_list
