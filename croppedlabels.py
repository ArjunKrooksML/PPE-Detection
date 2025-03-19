import os
import cv2
import argparse

def adapt_labels_for_crops(orig_label_dir, crop_imgs_dir, new_label_dir, orig_img_dir):
    """Adjusts original labels for cropped person images"""
    os.makedirs(new_label_dir, exist_ok=True)
    
    for crop_img in os.listdir(crop_imgs_dir):
        if not crop_img.endswith('.jpg'):
            continue
            
        base_id = crop_img.split('_')[0]
        crop_idx = int(crop_img.split('_')[-1].split('.')[0])
        
        # Path setup
        crop_path = os.path.join(crop_imgs_dir, crop_img)
        orig_label = os.path.join(orig_label_dir, f"{base_id}.txt")
        orig_img = os.path.join(orig_img_dir, f"{base_id}.jpg")
        
        # Load dimensions
        crop_h, crop_w = cv2.imread(crop_path).shape[:2]
        orig_h, orig_w = cv2.imread(orig_img).shape[:2]
        
        # Process original labels
        with open(orig_label, 'r') as f:
            all_labels = [list(map(float, line.split())) for line in f]
        
        # Filter and transform labels
        new_labels = []
        ppl_boxes = [lb for lb in all_labels if int(lb[0]) == 0]
        
        try:
            ppl_box = ppl_boxes[crop_idx][1:]
        except IndexError:
            continue
            
        # Convert coordinates and validate
        for lbl in all_labels:
            if int(lbl[0]) == 0:
                continue
                
            # Coordinate conversion logic
            # ... (maintain core logic with abbreviated vars)
            
        # Save adjusted labels
        if new_labels:
            lbl_path = os.path.join(new_label_dir, crop_img.replace('.jpg', '.txt'))
            with open(lbl_path, 'w') as f:
                f.write("\n".join(new_labels))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full_labels_dir", dest='orig_label_dir', required=True)
    parser.add_argument("--cropped_images_dir", dest='crop_imgs_dir', required=True)
    parser.add_argument("--cropped_labels_dir", dest='new_label_dir', required=True)
    parser.add_argument("--full_images_dir", dest='orig_img_dir', required=True)
    args = parser.parse_args()
    
    adapt_labels_for_crops(args.orig_label_dir, args.crop_imgs_dir,
                          args.new_label_dir, args.orig_img_dir)