# Real-Time License Plate Recognition Using Smart Edge Devices for Surveillance

This repository contains the source code, evaluation tools, and sample datasets for the Master's Thesis: **Real-Time License Plate Recognition Using Smart Edge Devices for Surveillance**, developed at **Instituto Superior de Engenharia de Lisboa (ISEL)** by Sebastião Póvoa Ferreira.

## Overview
This project implements a lightweight, deep-learning-based Automatic License Plate Recognition (ALPR) pipeline optimized specifically for resource-constrained edge hardware. By distributing the computational workload across a heterogeneous System-on-Chip (SoC), the system achieves real-time, on-device inference without relying on cloud connectivity or heavy server-grade GPUs.


### Pipeline Demonstration

https://github.com/user-attachments/assets/928ab38b-9235-4f7e-93aa-de833ade606f

*(Sample test video sourced from Pexels).*

**Example Outputs:**
<p align="center">
  <img src="assets/0506ej.png" width="400">
  <img src="assets/6900ma.png" width="400">
</p>

## Installation

**1. Clone the repository:**
```bash
git clone [https://github.com/SebastiaoPFerreira/projeto_TFM_LPR_Edge.git](https://github.com/SebastiaoPFerreira/projeto_TFM_LPR_Edge.git)
cd projeto_TFM_LPR_Edge
```

**2. Install dependencies:**
Ensure you are running this on the Khadas VIM3 with the KSNN libraries installed natively. 
```bash
pip install -r requirements.txt
```

**3. Model Files:**
Ensure the compiled `.nb` and `.so` files for YOLOv8n, and the `.onnx` file for PaddleOCR are placed in the `models/` directory.

## Usage

The entire ALPR pipeline is consolidated into a single executable script: `pipeline.py`. It generates a detailed per-stage latency benchmark table upon completion.

**To run a batch evaluation on an entire dataset directory:**
```bash
python pipeline.py \
  --library models/libnn_yolov8n.so \
  --model models/yolov8n_fp16.nb \
  --ocr_onnx models/paddleocr_v4_mobile_opset11.onnx \
  --folder data/sample_images/ \
  --output ./output_results
```

**To run the pipeline on a saved video file:**
```bash
python pipeline.py \
  --library models/libnn_yolov8n.so \
  --model models/yolov8n_fp16.nb \
  --ocr_onnx models/paddleocr_v4_mobile_opset11.onnx \
  --source data/test_videos/pexels_traffic.mp4 \
  --output ./output_results
```

**To run a live USB Webcam (Camera 0):**
```bash
python pipeline.py \
  --library models/libnn_yolov8n.so \
  --model models/yolov8n_fp16.nb \
  --ocr_onnx models/paddleocr_v4_mobile_opset11.onnx \
  --source 0 \
  --output ./output_results
```
*(Note: If you experience display issues on Wayland when running the live video feed, run `export QT_QPA_PLATFORM=xcb` before executing the script).*

## Citation
If you use this code, datasets, or methodology in your research, please refer to the thesis publication:

> Ferreira, Sebastião Póvoa. *Real-Time License Plate Recognition Using Smart Edge Devices for Surveillance.* Master's Thesis, Instituto Superior de Engenharia de Lisboa (ISEL), 2026.

## License & Acknowledgments
* Developed under the guidance of Dr. Mário Pereira Véstias.
* Published and presented at the YEF-ECE and INForum 2026 conferences.
* Royalty-free test video sourced from Pexels.
