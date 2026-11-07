import numpy as np
import os
import argparse
import sys
import time
import glob
import cv2 as cv
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

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    b1_area = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    b2_area = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = b1_area + b2_area - inter_area
    return inter_area / union if union > 0 else 0.0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", required=True, help="Path to .so library file")
    parser.add_argument("--model", required=True, help="Path to .nb model file")
    parser.add_argument("--images", required=True, help="Path to folder containing test images")
    parser.add_argument("--labels", required=True, help="Path to folder containing .txt labels")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        sys.exit(f"Model '{args.model}' does not exist.")
    if not os.path.exists(args.library):
        sys.exit(f"Library '{args.library}' does not exist.")
    if not os.path.isdir(args.images):
        sys.exit(f"Folder '{args.images}' does not exist.")
    if not os.path.isdir(args.labels):
        sys.exit(f"Folder '{args.labels}' does not exist.")

    # 1. Initialize KSNN NPU
    yolov3 = KSNN('VIM3')
    print(f"KSNN Version: {yolov3.get_nn_version()}")
    print("Initializing Neural Network on NPU...")
    yolov3.nn_init(library=args.library, model=args.model, level=0)
    print("Done.\n")

    # 2. Find Images
    img_paths = sorted(glob.glob(os.path.join(args.images, "*.[jJ][pP][gG]")) +
                       glob.glob(os.path.join(args.images, "*.[jJ][pP][eE][gG]")) +
                       glob.glob(os.path.join(args.images, "*.[pP][nN][gG]")))

    if not img_paths:
        sys.exit(f"No image files found in '{args.images}'")

    print(f"Found {len(img_paths)} images. Starting detection evaluation...\n" + "-"*65)

    yolo_times = []
    all_preds = []
    all_gts = {}  
    total_gts = 0

    for idx, path in enumerate(img_paths):
        filename = os.path.basename(path)
        base_name = os.path.splitext(filename)[0]
        txt_path = os.path.join(args.labels, base_name + '.txt')
        
        all_gts[idx] = []
        if os.path.exists(txt_path):
            with open(txt_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        _, cx, cy, w, h = map(float, parts[:5])
                        gt_box = [cx - w/2.0, cy - h/2.0, cx + w/2.0, cy + h/2.0]
                        all_gts[idx].append({'bbox': gt_box, 'used': False})
                        total_gts += 1

        orig_img = cv.imread(path)
        if orig_img is None:
            continue

        # Exact preprocessing from your original working script
        img = cv.resize(orig_img, (640, 640)).astype(np.float32)
        img = img / 255.0
        img = np.ascontiguousarray(img.transpose(2, 0, 1))

        # NPU Inference
        t0_yolo = time.time()
        data = yolov3.nn_inference([img], platform='ONNX', reorder='2 1 0', output_tensor=3, output_format=output_format.OUT_FORMAT_FLOAT32)
        
        in0 = data[2].reshape(SPAN, LISTSIZE, GRID0, GRID0).transpose(2, 3, 0, 1)
        in1 = data[1].reshape(SPAN, LISTSIZE, GRID1, GRID1).transpose(2, 3, 0, 1)
        in2 = data[0].reshape(SPAN, LISTSIZE, GRID2, GRID2).transpose(2, 3, 0, 1)

        boxes, scores, classes = yolov8_npu_post_process([in0, in1, in2])
        t1_yolo = time.time()
        y_time = (t1_yolo - t0_yolo) * 1000
        yolo_times.append(y_time)

        if boxes is not None:
            for box, score in zip(boxes, scores):
                all_preds.append((float(score), idx, box))

        print(f"[{idx+1:03d}/{len(img_paths):03d}] {filename:<25} | YOLO: {y_time:6.2f} ms | Preds: {len(boxes) if boxes is not None else 0}")

    # Compute mAP@50, Precision, and Recall
    all_preds.sort(key=lambda x: x[0], reverse=True)
    TP = np.zeros(len(all_preds))
    FP = np.zeros(len(all_preds))

    for i, (conf, img_idx, pred_box) in enumerate(all_preds):
        gts = all_gts[img_idx]
        best_iou = 0.0
        best_gt_idx = -1
        
        for gt_idx, gt in enumerate(gts):
            iou = calculate_iou(pred_box, gt['bbox'])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= 0.50 and best_gt_idx != -1 and not gts[best_gt_idx]['used']:
            TP[i] = 1
            gts[best_gt_idx]['used'] = True
        else:
            FP[i] = 1

    acc_TP = np.cumsum(TP)
    acc_FP = np.cumsum(FP)

    recalls = acc_TP / total_gts if total_gts > 0 else np.zeros_like(acc_TP)
    precisions = acc_TP / np.maximum(acc_TP + acc_FP, np.finfo(float).eps)

    # Area under Precision-Recall curve
    mAP_50 = np.sum((recalls[1:] - recalls[:-1]) * precisions[1:]) if len(recalls) > 1 else 0.0
    final_precision = precisions[-1] if len(precisions) > 0 else 0.0
    final_recall = recalls[-1] if len(recalls) > 0 else 0.0
    avg_latency = np.mean(yolo_times) if len(yolo_times) > 0 else 0.0

    print("YOLO DETECTION BENCHMARK RESULTS")
    print(f"Total Images:     {len(img_paths)}")
    print(f"Total Labels:     {total_gts}")
    print(f"Total Detections: {len(all_preds)}")
    print(f"mAP@50:           {mAP_50 * 100:.2f}%")
    print(f"Precision:        {final_precision * 100:.2f}%")
    print(f"Recall:           {final_recall * 100:.2f}%")
    print(f"Avg NPU Latency:  {avg_latency:.2f} ms")
