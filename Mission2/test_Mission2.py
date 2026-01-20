import os
import json
import random
import numpy as np
from PIL import Image
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system') 

import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import box_iou
from torchvision import transforms
import timm

# =========================
# 설정
# =========================
# RESULT_DIR = "results/hrnet_results_5"
RESULT_DIR = "./results/"
os.makedirs(RESULT_DIR, exist_ok=True)

def set_seed(seed: int = 500):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# =========================
# HRNet-W48 Height Regressor (학습 스크립트와 동일)
# =========================
class HeightRegressorHRNet(nn.Module):
    def __init__(self):
        super().__init__()
        # hrnet_w48: 분류헤드 제거 + GAP으로 (N, C) 1D 벡터 반환
        self.backbone = timm.create_model(
            'hrnet_w48',
            pretrained=True,
            num_classes=0,
            global_pool='avg'
        )
        in_features = getattr(self.backbone, "num_features", None)
        if in_features is None:
            in_features = self.backbone.feature_info[-1]['num_chs']

        self.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )

    def forward(self, crops):
        features = self.backbone(crops)
        return self.fc(features).squeeze(1)

# =========================
# Dataset 
# =========================
class ChimneyDataset(Dataset):
    def __init__(self, images_dir: str, unified_json_path: str, resize_to: int = 512):
        with open(unified_json_path, "r") as f: data = json.load(f)
        self.images = {img["id"]: img for img in data["images"]}
        self.annotations = {}
        for ann in data["annotations"]:
            if ann["bbox"][2] > 0 and ann["bbox"][3] > 0:
                self.annotations.setdefault(ann["image_id"], []).append(ann)
        self.ids = list(self.images.keys())
        self.images_dir = images_dir
        self.resize_to = resize_to
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    @staticmethod
    def cxcywh_norm_to_xyxy(bbox_norm, w, h):
        cx, cy, bw, bh = bbox_norm
        cx *= w; cy *= h; bw *= w; bh *= h
        return [cx - 0.5 * bw, cy - 0.5 * bh, cx + 0.5 * bw, cy + 0.5 * bh]

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        info = self.images[img_id]
        file_name = info["file_name"]
        img_path = os.path.join(self.images_dir, file_name)
        # pil = Image.open(img_path).convert("RGB")
        with Image.open(img_path) as im:
            pil = im.convert("RGB").copy()
        W, H = pil.size
        anns = self.annotations.get(img_id, [])
        boxes, heights = [], []
        for ann in anns:
            xyxy = self.cxcywh_norm_to_xyxy(ann["bbox"], W, H)
            if xyxy[2] > xyxy[0] and xyxy[3] > xyxy[1]:
                boxes.append(xyxy); heights.append(float(ann["height_m"]))
        boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4))
        heights = torch.tensor(heights, dtype=torch.float32)
        pil_resized = pil.resize((self.resize_to, self.resize_to))
        scale_x, scale_y = self.resize_to / W, self.resize_to / H
        boxes_resized = boxes.clone()
        if boxes_resized.shape[0] > 0:
            boxes_resized[:, [0, 2]] *= scale_x
            boxes_resized[:, [1, 3]] *= scale_y
        img_tensor = self.to_tensor(pil_resized)
        target = {"boxes": boxes_resized, "heights": heights, "image_id": torch.tensor([img_id]),
                  "file_name": file_name, "orig_size": (W, H)}
        return img_tensor, target, pil

    def __len__(self):
        return len(self.ids)

def collate_fn(batch):
    return [b[0] for b in batch], [b[1] for b in batch], [b[2] for b in batch]

# =========================
# 시각화 함수 
# =========================
def draw_results_1x2(pil_img, gt_boxes, gt_heights, pred_boxes, pred_heights, miou, mae, rmse, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    fig.suptitle(f"mIoU: {miou:.4f} | MAE: {mae:.2f}m | RMSE: {rmse:.2f}m", fontsize=16)
    # Ground Truth
    ax = axes[0]; ax.imshow(pil_img); ax.set_title("Ground Truth")
    for box, h in zip(gt_boxes, gt_heights):
        x1, y1, x2, y2 = box
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor="g", facecolor="none")
        ax.add_patch(rect)
        ax.text(x1, y1 - 10, f"{h:.1f}m", color="white", backgroundcolor='g')
    ax.axis("off")
    # Prediction
    ax = axes[1]; ax.imshow(pil_img); ax.set_title("Prediction")
    for box, h in zip(pred_boxes, pred_heights):
        x1, y1, x2, y2 = box
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor="r", facecolor="none")
        ax.add_patch(rect)
        ax.text(x1, y1 - 10, f"{h:.1f}m", color="white", backgroundcolor='r')
    ax.axis("off")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=150)
    plt.close(fig)

# =========================
# 테스트 실행 함수 
# =========================
def test_with_hrnet(weights_path: str, data_dir: str, device: str = "cuda", resize_to: int = 512, conf_thres: float = 0.5, iou_thres: float = 0.3, max_vis: int = 5):
    set_seed(500)
    ds = ChimneyDataset(os.path.join(data_dir, "images"), os.path.join(data_dir, "unified_labels/val_unified.json"), resize_to=resize_to)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn, pin_memory=False)

    # Detector
    detector = fasterrcnn_resnet50_fpn(weights=None)
    in_feat = detector.roi_heads.box_predictor.cls_score.in_features
    # [수정] 정의되지 않은 클래스명 버그 수정
    detector.roi_heads.box_predictor = FastRCNNPredictor(in_feat, 2)
    reg = HeightRegressorHRNet()

    ckpt = torch.load(weights_path, map_location=device)
    detector.load_state_dict(ckpt["detector"])
    reg.load_state_dict(ckpt["reg"])
    detector.to(device).eval()
    reg.to(device).eval()

    samples_to_save = []
    
    all_ious, all_abs_errors, all_sq_errors = [], [], []

    with torch.no_grad():
        for imgs, targets, pils in tqdm(loader, desc="Testing (HRNet)"):
            img, tgt, pil = imgs[0].to(device), targets[0], pils[0]
            file_name, (W, H) = tgt["file_name"], tgt["orig_size"]
            scale_x, scale_y = W / resize_to, H / resize_to

            pred = detector([img])[0]
            keep = pred["scores"] > conf_thres
            pred_boxes_rs = pred["boxes"][keep]

            pred_heights = []
            if pred_boxes_rs.shape[0] > 0:
                crops = torchvision.ops.roi_align(img.unsqueeze(0), [pred_boxes_rs], (224, 224))
                pred_heights = reg(crops).cpu().numpy()

            gt_boxes_rs = tgt["boxes"].to(device)
            gt_heights = tgt["heights"].cpu().numpy()

            img_ious, img_abs_errs, img_sq_errs = [], [], []
            if gt_boxes_rs.shape[0] > 0 and pred_boxes_rs.shape[0] > 0:
                iou_mat = box_iou(pred_boxes_rs, gt_boxes_rs)
                best_iou, best_gt_idx = iou_mat.max(dim=1)
                
                for i, iou_val in enumerate(best_iou):
                    if iou_val >= iou_thres:
                        gt_h = gt_heights[best_gt_idx[i]]
                        pr_h = pred_heights[i]
                        img_ious.append(iou_val.item())
                        img_abs_errs.append(abs(pr_h - gt_h))
                        img_sq_errs.append((pr_h - gt_h) ** 2)
            
            # 전체 성능 리스트에 현재 이미지 결과 추가
            all_ious.extend(img_ious)
            all_abs_errors.extend(img_abs_errs)
            all_sq_errors.extend(img_sq_errs)

            # 시각화용 샘플 저장
            gt_boxes_orig = tgt["boxes"].clone()
            if gt_boxes_orig.shape[0] > 0: gt_boxes_orig[:, [0, 2]] *= scale_x; gt_boxes_orig[:, [1, 3]] *= scale_y
            pred_boxes_orig = pred_boxes_rs.clone().cpu()
            if pred_boxes_orig.shape[0] > 0: pred_boxes_orig[:, [0, 2]] *= scale_x; pred_boxes_orig[:, [1, 3]] *= scale_y
            
            samples_to_save.append({
                "pil": pil.copy(), "file_name": file_name,
                "gt_boxes": gt_boxes_orig.numpy(), "gt_heights": gt_heights,
                "pred_boxes": pred_boxes_orig.numpy(), "pred_heights": pred_heights,
                "miou": np.mean(img_ious) if img_ious else 0.0,
                "mae": np.mean(img_abs_errs) if img_abs_errs else 0.0,
                "rmse": np.sqrt(np.mean(img_sq_errs)) if img_sq_errs else 0.0,
            })

    # 전체 테스트셋에 대한 최종 성능 계산 및 출력
    if all_ious:
        total_miou = np.mean(all_ious)
        total_mae = np.mean(all_abs_errors)
        total_rmse = np.sqrt(np.mean(all_sq_errors))
        print("\n" + "="*50)
        print("최종 성능 평가 결과 (전체 테스트셋)")
        print(f"   - 총 {len(all_ious)}개의 매칭된 객체에 대한 평가")
        print(f"   - Bounding Box 평균 IoU (mIoU) : {total_miou:.4f}")
        print(f"   - 평균 높이 예측 오차 (MAE)      : {total_mae:.2f}m")
        print(f"   - 높이 예측 오차 제곱근 (RMSE)   : {total_rmse:.2f}m")
        print("="*50 + "\n")
    else:
        print("\n IoU 임계값을 만족하는 객체를 찾지 못해 전체 점수를 계산할 수 없습니다.\n")

    # 시각화 결과 저장
    random.shuffle(samples_to_save)
    for i, s in enumerate(samples_to_save[:max_vis]):
        save_path = os.path.join(RESULT_DIR, f"{i+1:02d}_{s['file_name']}")
        draw_results_1x2(s["pil"], s["gt_boxes"], s["gt_heights"], s["pred_boxes"], s["pred_heights"], s["miou"], s["mae"], s["rmse"], save_path)
    print(f"시각화 결과 저장 완료: {RESULT_DIR} (총 {min(max_vis, len(samples_to_save))}장)")


if __name__ == "__main__":
    # WEIGHTS = "./runs/faster_hrnet_5/best_epoch.pth"
    WEIGHTS = "./best_epoch.pth"
    DATA_DIR = "../data/Mission2/Validation"
    DEV = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.exists(WEIGHTS):
        print(f"[!] 가중치 파일을 찾을 수 없습니다: {WEIGHTS}")
    else:
        test_with_hrnet(
            weights_path=WEIGHTS, data_dir=DATA_DIR, device=DEV,
            resize_to=512, conf_thres=0.5, iou_thres=0.3, max_vis=10
        )