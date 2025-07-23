
import os
import cv2
import torch
import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore
from PIL import Image
from torchvision import transforms
from segment_anything_hq import sam_model_registry, SamPredictor
from open_clip import create_model_and_transforms, get_tokenizer
import torchvision.ops as ops

IMG_SIZE = 640

clip_model, _, clip_preprocess = create_model_and_transforms(
    'ViT-L-14', pretrained='openai', device='cuda' if torch.cuda.is_available() else 'cpu'
)
clip_tokenizer = get_tokenizer('ViT-L-14')

def load_sam_hq_tiny(model_path="sam_hq_vit_tiny.pth"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry["vit_tiny"](checkpoint=model_path).to(device)
    predictor = SamPredictor(sam)
    return predictor

sam_predictor = load_sam_hq_tiny()

def remove_group_boxes(dino_boxes, existing_boxes, iou_threshold=0.3, max_allowed_overlap=1):
    filtered = []
    for dino_box in dino_boxes:
        overlaps = sum(compute_iou(dino_box, manual_box) > iou_threshold for manual_box in existing_boxes)
        if overlaps <= max_allowed_overlap:
            filtered.append(dino_box)
    return filtered

def remove_large_boxes(boxes, scale_factor=1.5, min_rel_area=0.2):
    """
    Removes boxes that are either:
    - too large (area > median_area * scale_factor)
    - too small (area < median_area * min_rel_area)
    """
    if not boxes:
        return []

    areas = [(x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in boxes]
    median_area = np.median(areas)

    filtered = [
        box for box, area in zip(boxes, areas)
        if median_area * min_rel_area <= area <= median_area * scale_factor
    ]
    return filtered

def box_centroid(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)

def centroid_distance(box1, box2):
    cx1, cy1 = box_centroid(box1)
    cx2, cy2 = box_centroid(box2)
    return ((cx1 - cx2)**2 + (cy1 - cy2)**2)**0.5

def compute_iou(boxA, boxB):
    xA1, yA1, xA2, yA2 = boxA
    xB1, yB1, xB2, yB2 = boxB

    x1 = max(xA1, xB1)
    y1 = max(yA1, yB1)
    x2 = min(xA2, xB2)
    y2 = min(yA2, yB2)

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    if inter_area == 0:
        return 0.0

    areaA = (xA2 - xA1) * (yA2 - yA1)
    areaB = (xB2 - xB1) * (yB2 - yB1)
    union_area = areaA + areaB - inter_area

    return inter_area / union_area




def get_clip_features(image_pil):
    device = next(clip_model.parameters()).device
    image_input = clip_preprocess(image_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = clip_model.encode_image(image_input)
    return image_features / image_features.norm(dim=-1, keepdim=True)

class Annotator(QtWidgets.QLabel):
    def __init__(self, image_path):
        super().__init__()
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        self.orig_image = self.image.copy()
        self.image = cv2.resize(self.image, (IMG_SIZE, IMG_SIZE))
        self.setPixmap(QtGui.QPixmap.fromImage(self.convert_cv_qt(self.image)))
        self.drawing = False
        self.start_point = None
        self.end_point = None
        self.auto_drawn_boxes = [] 
        self.boxes = []
        self.setMouseTracking(True)
        self.setFixedSize(IMG_SIZE, IMG_SIZE)

    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        return convert_to_Qt_format

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.drawing = True
            self.start_point = event.pos()
        elif event.button() == QtCore.Qt.RightButton:
            x, y = event.x(), event.y()
            for box_list in [self.boxes, self.auto_drawn_boxes]:
                for box in box_list:
                    x1, y1, x2, y2 = box
                    if x1 <= x <= x2 and y1 <= y <= y2:
                        box_list.remove(box)
                        self.redraw_all_boxes()
                        return

    def mouseMoveEvent(self, event):
        if self.drawing:
            self.end_point = event.pos()
            temp_img = self.image.copy()
            cv2.rectangle(temp_img, (self.start_point.x(), self.start_point.y()),
                          (self.end_point.x(), self.end_point.y()), (0, 255, 0), 2)
            self.setPixmap(QtGui.QPixmap.fromImage(self.convert_cv_qt(temp_img)))

    def mouseReleaseEvent(self, event):
        if self.drawing and self.start_point and self.end_point:
            self.drawing = False
            box = [self.start_point.x(), self.start_point.y(), self.end_point.x(), self.end_point.y()]
            self.boxes.append(box)
            cv2.rectangle(self.image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            self.setPixmap(QtGui.QPixmap.fromImage(self.convert_cv_qt(self.image)))

    def get_manual_boxes(self):
        return self.boxes

    def get_image(self):
        return self.image

    def redraw_all_boxes(self):
        self.image = cv2.resize(self.orig_image.copy(), (IMG_SIZE, IMG_SIZE))
        for box in self.boxes:
            cv2.rectangle(self.image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        for box in self.auto_drawn_boxes:
            cv2.rectangle(self.image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        self.setPixmap(QtGui.QPixmap.fromImage(self.convert_cv_qt(self.image)))

def predict_masks_from_boxes(image, boxes):
    sam_predictor.set_image(image)
    masks = []
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        box_w, box_h = x2 - x1, y2 - y1
        offset = min(box_w, box_h) // 8
        points = np.array([
            [cx, cy],
            [cx - offset, cy - offset],
            [cx + offset, cy - offset],
            [cx - offset, cy + offset],
            [cx + offset, cy + offset],
        ])
        labels = np.ones(len(points), dtype=np.int32)
        masks_i, _, _ = sam_predictor.predict(point_coords=points, point_labels=labels, multimask_output=False)
        masks.append(masks_i[0])
    return masks

# def get_similar_regions(image, ref_boxes, thresholds=[0.99, 0.80]):
#     if not ref_boxes:
#         return []
#     ref_crops = [Image.fromarray(image[y1:y2, x1:x2]) for x1, y1, x2, y2 in ref_boxes]
#     ref_feats = torch.cat([get_clip_features(crop) for crop in ref_crops])
#     similar_boxes = []
#     h, w = image.shape[:2]
#     step = 64
#     for thr in thresholds:
#         for y in range(0, h - step, step // 2):
#             for x in range(0, w - step, step // 2):
#                 patch = Image.fromarray(image[y:y + step, x:x + step])
#                 patch_feat = get_clip_features(patch)
#                 sims = torch.mm(patch_feat, ref_feats.T)
                
#                 if sims.max().item() > thr:
#                     similar_boxes.append([x, y, x + step, y + step])
#     # Deduplicate boxes here if needed
#     return similar_boxes

def get_similar_regions(image, ref_boxes, threshold=0.8):
    """Finds image regions similar to reference boxes using CLIP embeddings.
    
    Args:
        image: Input image array (H,W,3)
        ref_boxes: List of reference boxes [[x1,y1,x2,y2],...]
        threshold: Similarity score cutoff (default: 0.8)
    
    Returns:
        List of similar bounding boxes
    """
    if not ref_boxes:
        return []

    # Extract CLIP features for all reference crops
    ref_crops = [Image.fromarray(image[y1:y2, x1:x2]) for x1,y1,x2,y2 in ref_boxes]
    ref_feats = torch.cat([get_clip_features(crop) for crop in ref_crops])
    
    similar_boxes = []
    h, w = image.shape[:2]
    step = 64  # Search window size
    
    # Slide across image with 50% overlap
    for y in range(0, h-step, step//2):
        for x in range(0, w-step, step//2):
            patch = Image.fromarray(image[y:y+step, x:x+step])
            patch_feat = get_clip_features(patch)
            
            # Compare against all reference features
            similarity_scores = torch.mm(patch_feat, ref_feats.T)
            
            if similarity_scores.max().item() > threshold:
                similar_boxes.append([x, y, x+step, y+step])
    
    return similar_boxes


class MainWindow(QtWidgets.QWidget):
    def __init__(self, image_path):
        super().__init__()
        self.setWindowTitle("SAM-HQ Box Prompt Annotator")
        self.annotator = Annotator(image_path)
        self.btn_auto = QtWidgets.QPushButton("Auto BBox")
        self.btn_auto.clicked.connect(self.run_clip_sam)
        self.reset_button = QtWidgets.QPushButton("Reset Auto BBox")
        self.reset_button.clicked.connect(self.reset_all_bboxes)
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.annotator)
        layout.addWidget(self.btn_auto)
        layout.addWidget(self.reset_button)
        self.setLayout(layout)

    def run_clip_sam(self):
        manual_boxes = self.annotator.get_manual_boxes()
        image = self.annotator.get_image()
        drawn_boxes = [list(map(int, box)) for box in manual_boxes]

        # Use previously drawn auto boxes + manual boxes as existing boxes
        existing_boxes = drawn_boxes + self.annotator.auto_drawn_boxes

        # Get new candidate boxes from CLIP similarity
        new_boxes = get_similar_regions(image, manual_boxes)
        print(f"📦 Found {len(new_boxes)} similar boxes.")

        # Predict masks for all (existing + new) boxes
        all_candidate_boxes = existing_boxes + new_boxes
        masks = predict_masks_from_boxes(image, all_candidate_boxes)

        accepted_boxes = []

        for i, mask in enumerate(masks):
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            x, y, w, h = cv2.boundingRect(contours[0])
            area = w * h
            if area < 200:  # Minimum area threshold
                continue

            new_box = [x, y, x + w, y + h]

            # Skip if overlaps manual boxes too much (IoU > 0.3)
            if any(compute_iou(new_box, mb) > 0.3 for mb in drawn_boxes):
                continue

            # Skip if centroid too close (<15 px) to any existing box centroid
            if any(centroid_distance(new_box, eb) < 15 for eb in existing_boxes):
                continue

            # Skip if box overlaps multiple existing boxes (likely group box)
            overlap_count = sum(compute_iou(new_box, eb) > 0.3 for eb in existing_boxes)
            if overlap_count > 1:
                continue

            accepted_boxes.append(new_box)
        
        #  Apply group box and large box filters here:
        filtered_new_boxes = remove_group_boxes(accepted_boxes, existing_boxes=all_candidate_boxes, iou_threshold=0.3, max_allowed_overlap=1)
        accepted_boxes = remove_large_boxes(filtered_new_boxes, scale_factor=2)
        # Run NMS on combined accepted + existing boxes (IoU threshold 0.5)
        combined_boxes = existing_boxes + accepted_boxes
        boxes_tensor = torch.tensor(combined_boxes, dtype=torch.float32)
        scores = torch.ones(len(combined_boxes))  # dummy scores
        keep_indices = ops.nms(boxes_tensor, scores, iou_threshold=0.4)
        final_boxes = [combined_boxes[i] for i in keep_indices]

        # Update annotator's auto drawn boxes: only keep boxes not in manual_boxes
        self.annotator.auto_drawn_boxes = [box for box in final_boxes if box not in drawn_boxes]

        # Redraw all boxes (manual + auto)
        self.annotator.redraw_all_boxes()

 


    def reset_all_bboxes(self):
        self.annotator.boxes = []
        self.annotator.auto_drawn_boxes = []
        self.annotator.image = cv2.resize(self.annotator.orig_image.copy(), (IMG_SIZE, IMG_SIZE))
        self.annotator.setPixmap(QtGui.QPixmap.fromImage(self.annotator.convert_cv_qt(self.annotator.image)))
        print("🔁 All bounding boxes cleared and image reset.")

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    img_path = r"C:\Naman\projects\maize_data\maize_data_top\images\3.jpg"
    window = MainWindow(img_path)
    window.show()
    sys.exit(app.exec_())
