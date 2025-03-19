# PPE-Detection

PPE Detection System Readme
Project Overview

Dataset: https://drive.google.com/file/d/1myGjrJZSWPT6LYOshF9gfikyXaTCBUWb/view

Toolkit for detecting Personal Protective Equipment (PPE) using YOLO models.

    Detects people in images
    Crops person regions
    Checks PPE compliance
    Includes training scripts and data converters

Files & Usage

    croppedimages.py
    Crops detected people from images
    Usage:
    python croppedimages.py --person_model [MODEL_PATH] --input_dir [IMAGE_DIR]

    croppedlabels.py
    Adjusts labels for cropped person images
    Usage:
    python croppedlabels.py --full_labels_dir [LABEL_DIR] --cropped_images_dir [CROP_DIR] --cropped_labels_dir [NEW_LABEL_DIR] --full_images_dir [ORIG_IMG_DIR]

    inference.py
    Full PPE detection pipeline
    Usage:
    python inference.py --images [IMAGE_DIR] --cropped [CROP_DIR] --output [OUT_DIR] --person_model [PERSON_MODEL] --ppe_model [PPE_MODEL]

    persontrain.py
    Trains person detection model
    Usage:
    python persontrain.py --yaml [DATA_YAML] --epochs [NUM] --imgsz [SIZE] --batch [SIZE]

    ppetrain.py
    Trains PPE detection model
    Usage:
    python ppetrain.py --data_yaml [DATA_YAML] --weights [INIT_WEIGHTS] --epochs [NUM] --imgsz [SIZE] --batch [SIZE]

    VOCtoYOLO.py
    Converts VOC XML to YOLO format
    Usage:
    python VOCtoYOLO.py [XML_DIR] [OUTPUT_DIR] [CLASS_FILE]

Installation

pip install ultralytics opencv-python argparse  

Workflow

    Prepare data using VOCtoYOLO.py
    Train models:

python persontrain.py --yaml person_data.yaml  

python ppetrain.py --data_yaml ppe_data.yaml --weights yolov8n.pt  

Run inference:

    python inference.py --images input/ --cropped crops/ --output results/ --person_model person.onnx --ppe_model ppe.onnx  

Key Features

    Separate models for person detection and PPE checks
    Maintains original image context for visualization
    Exports models to ONNX format
    Batch processing support

Requirements

    Python 3.8+
    Ultralytics YOLOv8
    OpenCV 4.5+
    ONNX runtime
