import os
import argparse
from ultralytics import YOLO

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml", type=str, required=True, dest='data_cfg')
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()
    
    ppl_model = YOLO("yolov8n.pt")
    ppl_model.train(data=os.path.abspath(args.data_cfg), 
                   epochs=args.epochs,
                   imgsz=args.imgsz,
                   batch=args.batch)
    ppl_model.export(format="onnx")