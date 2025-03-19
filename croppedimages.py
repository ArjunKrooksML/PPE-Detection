import os
import cv2
import argparse
from ultralytics import YOLO

def crop_ppl_imgs(ppl_model, in_dir, min_conf=0.5):
    """Crops person regions from input images using detection model"""
    script_dir = os.path.abspath(os.path.dirname(__file__))
    out_dir = os.path.join(script_dir, "cropped_ppl")
    os.makedirs(out_dir, exist_ok=True)
    
    detector = YOLO(os.path.abspath(ppl_model))
    
    for img_file in os.listdir(in_dir):
        if not img_file.lower().endswith(('.jpg', '.png')):
            continue
            
        img_path = os.path.join(in_dir, img_file)
        img = cv2.imread(img_path)
        
        # Detect people
        detections = detector(img)[0]
        
        for idx, box in enumerate(detections.boxes.xyxy):
            x1, y1, x2, y2 = map(int, box)
            conf = detections.boxes.conf[idx].item()
            
            if conf > min_conf:
                ppl_crop = img[y1:y2, x1:x2]
                crop_path = os.path.join(out_dir, f"{os.path.splitext(img_file)[0]}_{idx}.jpg")
                cv2.imwrite(crop_path, ppl_crop)
    
    print("Person cropping complete")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--person_model", type=str, required=True, dest='ppl_model')
    parser.add_argument("--input_dir", type=str, required=True, dest='in_dir')
    args = parser.parse_args()
    
    crop_ppl_imgs(args.ppl_model, args.in_dir)