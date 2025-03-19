import os
import cv2
import argparse
from ultralytics import YOLO

def find_and_snip_ppl(img_path, crop_dir, ppl_model):
    """Finds people and saves cropped regions"""
    img = cv2.imread(img_path)
    results = ppl_model(img)
    ppl_boxes = []
    
    for idx, res in enumerate(results):
        for box in res.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            ppl_boxes.append((x1, y1, x2, y2))
            
            crop = img[y1:y2, x1:x2]
            cv2.imwrite(f"{crop_dir}/{os.path.basename(img_path)[:-4]}_{idx}.jpg", crop)
    
    return ppl_boxes

def check_ppe_on_crops(ppl_boxes, orig_img, crop_dir, out_dir, ppe_model):
    """Checks safety gear on cropped people"""
    img = cv2.imread(orig_img)
    detections = ppe_model.predict(crop_dir, save=False)
    
    for idx, (x1, y1, x2, y2) in enumerate(ppl_boxes):
        for gear in detections[idx].boxes:
            cls_id = int(gear.cls[0])
            conf = float(gear.conf[0])
            gx1, gy1, gx2, gy2 = map(int, gear.xyxy[0])
            
            # Map to original image
            abs_x1 = x1 + gx1
            abs_y1 = y1 + gy1
            abs_x2 = x1 + gx2
            abs_y2 = y1 + gy2
            
            # Draw results
            cv2.rectangle(img, (abs_x1, abs_y1), (abs_x2, abs_y2), (0,255,0), 2)
            cv2.putText(img, f"PPE-{cls_id} {conf:.2f}", 
                       (abs_x1, abs_y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    
    out_path = os.path.join(out_dir, os.path.basename(orig_img))
    cv2.imwrite(out_path, img)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True, dest='img_dir')
    parser.add_argument("--cropped", required=True, dest='crop_dir')
    parser.add_argument("--output", required=True, dest='out_dir')
    parser.add_argument("--person_model", required=True, dest='ppl_model')
    parser.add_argument("--ppe_model", required=True)
    args = parser.parse_args()
    
    ppl_detector = YOLO(args.ppl_model)
    ppe_checker = YOLO(args.ppe_model)
    
    os.makedirs(args.crop_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    
    for img_file in os.listdir(args.img_dir):
        if img_file.endswith((".jpg", ".png")):
            img_path = os.path.join(args.img_dir, img_file)
            boxes = find_and_snip_ppl(img_path, args.crop_dir, ppl_detector)
            check_ppe_on_crops(boxes, img_path, args.crop_dir, args.out_dir, ppe_checker)