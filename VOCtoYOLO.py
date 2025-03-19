import os
import argparse
import xml.etree.ElementTree as ET

def parse_xml(xml_p, cls_map):
    try:
        tree = ET.parse(xml_p)
        root = tree.getroot()
        
        img_w = int(root.find("size/width").text)
        img_h = int(root.find("size/height").text)
        
        annos = []
        for obj in root.findall("object"):
            cls_name = obj.find("name").text.strip()
            if cls_name not in cls_map:
                print(f"Class '{cls_name}' missing")
                continue
            
            cls_id = cls_map[cls_name]
            bbox = obj.find("bndbox")
            
            xmin = int(bbox.find("xmin").text)
            ymin = int(bbox.find("ymin").text)
            xmax = int(bbox.find("xmax").text)
            ymax = int(bbox.find("ymax").text)
            
            cx = ((xmin + xmax) / 2) / img_w
            cy = ((ymin + ymax) / 2) / img_h
            w = (xmax - xmin) / img_w
            h = (ymax - ymin) / img_h
            
            annos.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        
        return annos
    
    except Exception as e:
        print(f"Error processing {xml_p}")
        return []

def voc2yolo(in_dir, out_dir, cls_file):
    os.makedirs(out_dir, exist_ok=True)
    
    if not os.path.exists(cls_file):
        print(f"Class file {cls_file} missing")
        return
    
    with open(cls_file, "r") as f:
        cls_map = {n.strip(): i for i, n in enumerate(f.readlines())}
    
    print(f"Class mapping: {cls_map}")
    
    if not os.path.exists(in_dir):
        print(f"Input dir {in_dir} not found")
        return
    
    for xml_f in os.listdir(in_dir):
        if not xml_f.endswith(".xml"):
            continue
        
        xml_p = os.path.join(in_dir, xml_f)
        yolo_annos = parse_xml(xml_p, cls_map)
        
        if yolo_annos:
            txt_p = os.path.join(out_dir, xml_f.replace(".xml", ".txt"))
            with open(txt_p, "w") as f:
                f.write("\n".join(yolo_annos))
    
    print("Conversion complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert VOC XML to YOLO format")
    parser.add_argument("in_dir", help="Input dir with XML files")
    parser.add_argument("out_dir", help="Output dir for YOLO txt files")
    parser.add_argument("cls_file", help="File containing class list")
    
    args = parser.parse_args()
    voc2yolo(args.in_dir, args.out_dir, args.cls_file)