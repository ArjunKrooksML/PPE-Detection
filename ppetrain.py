import os
import argparse
from ultralytics import YOLO

def train_ppe_model(data_cfg, init_weights, epochs=20, img_size=640, batch=16):
    """Trains PPE detection model"""
    model = YOLO(os.path.abspath(init_weights))
    model.train(data=os.path.abspath(data_cfg),
               epochs=epochs,
               imgsz=img_size,
               batch=batch)
    model.export(format="onnx")
    print("PPE model training complete")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_yaml", dest='data_cfg', required=True)
    parser.add_argument("--weights", dest='init_weights', required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()
    
    train_ppe_model(args.data_cfg, args.init_weights, args.epochs, args.imgsz, args.batch)