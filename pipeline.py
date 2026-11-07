import numpy as np
import os
import argparse
import sys
import time
import glob
import re
import cv2 as cv
import onnxruntime as ort
from ksnn.api import KSNN
from ksnn.types import *

GRID0 = 20
GRID1 = 40
GRID2 = 80
LISTSIZE = 65
SPAN = 1
NUM_CLS = 1
OBJ_THRESH = 0.6
NMS_THRESH = 0.5
mean = [0, 0, 0]
var = [255]

constant_martix = np.array([[i for i in range(16)]]).T
CLASSES = ("license_plate",)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x, axis=0):
    x = np.exp(x)
    return x / x.sum(axis=axis, keepdims=True)

def process(input_data):
    grid_h, grid_w = map(int, input_data.shape[0:2])
    box_class_probs = sigmoid(input_data[..., :NUM_CLS])
    
    box_0 = softmax(input_data[..., NUM_CLS: NUM_CLS + 16], -1)
    box_1 = softmax(input_data[..., NUM_CLS + 16:NUM_CLS + 32], -1)
    box_2 = softmax(input_data[..., NUM_CLS + 32:NUM_CLS + 48], -1)
    box_3 = softmax(input_data[..., NUM_CLS + 48:NUM_CLS + 64], -1)
    
    result = np.zeros((grid_h, grid_w, 1, 4))
    result[..., 0] = np.dot(box_0, constant_martix)[..., 0]
    result[..., 1] = np.dot(box_1, constant_martix)[..., 0]
    result[..., 2] = np.dot(box_2, constant_martix)[..., 0]
    result[..., 3] = np.dot(box_3, constant_martix)[..., 0]

    col = np.tile(np.arange(0, grid_w), grid_w).reshape(-1, grid_w).reshape(grid_h, grid_w, 1, 1)
    row = np.tile(np.arange(0, grid_h).reshape(-1, 1), grid_h).reshape(grid_h, grid_w, 1, 1)
    grid = np.concatenate((col, row), axis=-1)

    result[..., 0:2] = (0.5 - result[..., 0:2] + grid) / (grid_w, grid_h)
    result[..., 2:4] = (0.5 + result[..., 2:4] + grid) / (grid_w, grid_h)
    return result, box_class_probs

def filter_boxes(boxes, box_class_probs):
    box_classes = np.argmax(box_class_probs, axis=-1)
    box_class_scores = np.max(box_class_probs, axis=-1)
    pos = np.where(box_class_scores >= OBJ_THRESH)
    return boxes[pos], box_classes[pos], box_class_scores[pos]

def nms_boxes(boxes, scores):
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1e-5)
        h = np.maximum(0.0, yy2 - yy1 + 1e-5)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= NMS_THRESH)[0]
        order = order[inds + 1]
    return np.array(keep)

def yolov8_npu_post_process(input_data):
    boxes, classes, scores = [], [], []
    for i in range(3):
        res, conf = process(input_data[i])
        b, c, s = filter_boxes(res, conf)
        boxes.append(b)
        classes.append(c)
        scores.append(s)

    boxes = np.concatenate(boxes)
    classes = np.concatenate(classes)
    scores = np.concatenate(scores)

    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        b, cl, sc = boxes[inds], classes[inds], scores[inds]
        keep = nms_boxes(b, sc)
        nboxes.append(b[keep])
        nclasses.append(cl[keep])
        nscores.append(sc[keep])

    if not nclasses or len(nclasses[0]) == 0:
        return None, None, None

    return np.concatenate(nboxes), np.concatenate(nscores), np.concatenate(nclasses)

def process_frame(orig_img, yolov3, ocr_session, ocr_input_name, char_list):
    # Stage 1: Pre-processing (resize, normalize, transpose)
    t_pre_start = time.time()
    img = cv.resize(orig_img, (640, 640)).astype(np.float32)
    img = img / 255.0
    img = np.ascontiguousarray(img.transpose(2, 0, 1))
    t_pre_end = time.time()
    lat_pre = (t_pre_end - t_pre_start) * 1000.0

    # Stage 2: YOLOv8n Inference
    t_yolo_start = time.time()
    data = yolov3.nn_inference([img], platform='ONNX', reorder='2 1 0', output_tensor=3, output_format=output_format.OUT_FORMAT_FLOAT32)
    t_yolo_end = time.time()
    lat_yolo_npu = (t_yolo_end - t_yolo_start) * 1000.0

    # Stage 3: Post-processing (reshaping, confidence filter, NMS)
    t_post_start = time.time()
    in0 = data[2].reshape(SPAN, LISTSIZE, GRID0, GRID0).transpose(2, 3, 0, 1)
    in1 = data[1].reshape(SPAN, LISTSIZE, GRID1, GRID1).transpose(2, 3, 0, 1)
    in2 = data[0].reshape(SPAN, LISTSIZE, GRID2, GRID2).transpose(2, 3, 0, 1)
    boxes, scores, classes = yolov8_npu_post_process([in0, in1, in2])
    t_post_end = time.time()
    lat_post_nms = (t_post_end - t_post_start) * 1000.0

    # Stage accumulators for OCR passes on this frame
    lat_crop = 0.0
    lat_ocr_infer = 0.0
    lat_ctc = 0.0
    detected_text = "N/A"

    if boxes is not None:
        for box, score, cl in zip(boxes, scores, classes):
            # Stage 4: RoI Cropping and Preprocessing
            t_crop_start = time.time()
            bx1, by1, bx2, by2 = box
            x1 = max(0, int(bx1 * orig_img.shape[1]))
            y1 = max(0, int(by1 * orig_img.shape[0]))
            x2 = min(orig_img.shape[1], int(bx2 * orig_img.shape[1]))
            y2 = min(orig_img.shape[0], int(by2 * orig_img.shape[0]))

            raw_crop = orig_img[y1:y2, x1:x2]
            if raw_crop.size == 0:
                continue

            crop_resized = cv.resize(raw_crop, (160, 48))
            img_float = crop_resized.astype(np.float32) / 255.0
            img_float = (img_float - 0.5) / 0.5
            img_tensor = np.expand_dims(np.transpose(img_float, (2, 0, 1)), axis=0)
            t_crop_end = time.time()
            lat_crop += (t_crop_end - t_crop_start) * 1000.0

            # Stage 5: PaddleOCR Inference (ONNX Runtime)
            t_ocr_start = time.time()
            preds = ocr_session.run(None, {ocr_input_name: img_tensor})[0]
            t_ocr_end = time.time()
            lat_ocr_infer += (t_ocr_end - t_ocr_start) * 1000.0

            # Stage 6: CTC Decoding & Text Filtering
            t_ctc_start = time.time()
            pred_idx = preds[0].argmax(axis=1)
            blank_idx = preds.shape[2] - 1
            raw_text = ""
            prev_idx = -1
            for i_idx in pred_idx:
                if i_idx != blank_idx and i_idx != prev_idx:
                    if i_idx < len(char_list):
                        raw_text += char_list[i_idx]
                prev_idx = i_idx
            detected_text = re.sub(r'[^A-Z0-9]', '', raw_text.upper())
            t_ctc_end = time.time()
            lat_ctc += (t_ctc_end - t_ctc_start) * 1000.0

            cv.rectangle(orig_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv.putText(orig_img, f"{detected_text}", (x1, max(y1 - 8, 20)), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    lat_total = lat_pre + lat_yolo_npu + lat_post_nms + lat_crop + lat_ocr_infer + lat_ctc
    
    return orig_img, lat_pre, lat_yolo_npu, lat_post_nms, lat_crop, lat_ocr_infer, lat_ctc, lat_total, detected_text

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", required=True, help="Path to .so library file")
    parser.add_argument("--model", required=True, help="Path to .nb model file")
    parser.add_argument("--ocr_onnx", required=True, help="Path to OCR .onnx file")
    parser.add_argument("--folder", default=None, help="Path to folder containing test images")
    parser.add_argument("--source", default=None, help="Path to video file or camera index (e.g., 0)")
    parser.add_argument("--output", default="./output_npu", help="Directory to save output images")
    args = parser.parse_args()

    if args.folder is None and args.source is None:
        sys.exit("You must provide either --folder or --source")

    if not os.path.exists(args.model):
        sys.exit(f"Model '{args.model}' does not exist.")
    if not os.path.exists(args.library):
        sys.exit(f"Library '{args.library}' does not exist.")
    if not os.path.exists(args.ocr_onnx):
        sys.exit(f"OCR ONNX '{args.ocr_onnx}' does not exist.")

    os.makedirs(args.output, exist_ok=True)

    # ONNX OCR
    print("Initializing ONNX OCR (CPU)...")
    ocr_session = ort.InferenceSession(args.ocr_onnx, providers=['CPUExecutionProvider'])
    ocr_input_name = ocr_session.get_inputs()[0].name
    
    # PaddleOCR character dictionary
    dict_path = 'utils/en_dict.txt'
    if not os.path.exists(dict_path):
        sys.exit(f"Dictionary file not found at {dict_path}")
        
    with open(dict_path, 'r', encoding='utf-8') as f:
        char_list = [line.strip('\n') for line in f.readlines()]
    char_list.append(' ')
    char_list.insert(0, '')

    print("Warming up OCR...")
    dummy_input = np.random.randn(1, 3, 48, 160).astype(np.float32)
    for _ in range(3):
        ocr_session.run(None, {ocr_input_name: dummy_input})
    print("Warm-up complete!\n")

    # KSNN NPU
    yolov3 = KSNN('VIM3')
    print(f" KSNN Version: {yolov3.get_nn_version()}")
    print("Initializing Neural Network on NPU...")
    yolov3.nn_init(library=args.library, model=args.model, level=0)
    print("Done.\n")

    # Detailed stage latency accumulators (in milliseconds)
    t_pre_list, t_yolo_npu_list, t_post_nms_list = [], [], []
    t_crop_list, t_ocr_infer_list, t_ctc_list, t_total_list = [], [], [], []

    # ==========================================
    # VIDEO / CAMERA MODE
    # ==========================================
    if args.source is not None:
        source_val = int(args.source) if args.source.isdigit() else args.source
        cap = cv.VideoCapture(source_val)
        
        if not cap.isOpened():
            sys.exit(f"Failed to open video source: {args.source}")
            
        print(f"Starting video stream from {args.source}. Press 'q' to quit.")
        idx = 0
        
        while True:
            ret, orig_img = cap.read()
            if not ret:
                break
            
            idx += 1
            
            orig_img, lat_pre, lat_yolo_npu, lat_post_nms, lat_crop, lat_ocr_infer, lat_ctc, lat_total, detected_text = process_frame(
                orig_img, yolov3, ocr_session, ocr_input_name, char_list
            )
            
            # Record latencies
            t_pre_list.append(lat_pre)
            t_yolo_npu_list.append(lat_yolo_npu)
            t_post_nms_list.append(lat_post_nms)
            t_crop_list.append(lat_crop)
            t_ocr_infer_list.append(lat_ocr_infer)
            t_ctc_list.append(lat_ctc)
            t_total_list.append(lat_total)
            
            print(f"[Frame {idx:04d}] Total: {lat_total:6.2f} ms | Text: {detected_text}")
            
            cv.imshow("LPR Pipeline Stream", orig_img)
            
            # Press 'q' to quit the video window
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv.destroyAllWindows()

    # ==========================================
    # FOLDER IMAGE MODE
    # ==========================================
    elif args.folder is not None:
        if not os.path.isdir(args.folder):
            sys.exit(f"Folder '{args.folder}' does not exist.")
            
        img_paths = sorted(glob.glob(os.path.join(args.folder, "*.[jJ][pP][gG]")) +
                           glob.glob(os.path.join(args.folder, "*.[jJ][pP][eE][gG]")) +
                           glob.glob(os.path.join(args.folder, "*.[pP][nN][gG]")))

        if not img_paths:
            sys.exit(f"No image files found in '{args.folder}'")

        print(f"Found {len(img_paths)} images. Starting per-stage latency benchmarking...\n" + "-"*65)

        for idx, path in enumerate(img_paths, 1):
            filename = os.path.basename(path)
            orig_img = cv.imread(path)
            if orig_img is None:
                continue

            orig_img, lat_pre, lat_yolo_npu, lat_post_nms, lat_crop, lat_ocr_infer, lat_ctc, lat_total, detected_text = process_frame(
                orig_img, yolov3, ocr_session, ocr_input_name, char_list
            )

            # Record latencies
            t_pre_list.append(lat_pre)
            t_yolo_npu_list.append(lat_yolo_npu)
            t_post_nms_list.append(lat_post_nms)
            t_crop_list.append(lat_crop)
            t_ocr_infer_list.append(lat_ocr_infer)
            t_ctc_list.append(lat_ctc)
            t_total_list.append(lat_total)

            print(f"[{idx:03d}/{len(img_paths):03d}] Total: {lat_total:6.2f} ms | YOLO(NPU): {lat_yolo_npu:5.2f} ms | OCR(CPU): {lat_ocr_infer:5.2f} ms | Text: {detected_text}")
            cv.imwrite(os.path.join(args.output, filename), orig_img)

    # ==========================================
    # FINAL METRICS PRINT OUT
    # ==========================================
    if len(t_total_list) > 0:
        avg_pre = np.mean(t_pre_list)
        avg_yolo = np.mean(t_yolo_npu_list)
        avg_post = np.mean(t_post_nms_list)
        avg_crop = np.mean(t_crop_list)
        avg_ocr = np.mean(t_ocr_infer_list)
        avg_ctc = np.mean(t_ctc_list)
        avg_total = np.mean(t_total_list)
        fps = 1000.0 / avg_total if avg_total > 0 else 0.0

        print("\nEND-TO-END Pipeline latency")
        print(f"| Pipeline Stage                               | Compute Domain | Latency (ms) |")
        print(f"| :------------------------------------------- | :------------- | :----------- |")
        print(f"| Pre-processing (resize, normalize)           | CPU            | {avg_pre:10.2f}   |")
        print(f"| YOLOv8n Inference                            | NPU            | {avg_yolo:10.2f}   |")
        print(f"| CPU Post-processing (confidence filter + NMS)| CPU            | {avg_post:10.2f}   |")
        print(f"| RoI Cropping                                 | CPU            | {avg_crop:10.2f}   |")
        print(f"| PaddleOCR Inference (ONNX Runtime)           | CPU            | {avg_ocr:10.2f}   |")
        print(f"| CTC Decoding                                 | CPU            | {avg_ctc:10.2f}   |")
        print(f"| Total End-to-End Latency                     | —              | {avg_total:8.2f}   |")
        print(f"System Throughput: {fps:.2f} FPS")