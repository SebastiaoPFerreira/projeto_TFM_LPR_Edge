import numpy as np
import os
import argparse
import sys
import time
import glob
import re
import cv2 as cv
import onnxruntime as ort

def calculate_cer(pred, target):
    if len(target) == 0: return 1.0 if len(pred) > 0 else 0.0
    d = np.zeros((len(target) + 1, len(pred) + 1))
    for i in range(len(target) + 1): d[i, 0] = i
    for j in range(len(pred) + 1): d[0, j] = j
    for i in range(1, len(target) + 1):
        for j in range(1, len(pred) + 1):
            cost = 0 if target[i - 1] == pred[j - 1] else 1
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + cost)
    return d[len(target), len(pred)] / len(target)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ocr_onnx", required=True, help="Path to OCR .onnx file")
    parser.add_argument("--folder", required=True, help="Path to folder containing cropped plate images")
    args = parser.parse_args()

    ocr_session = ort.InferenceSession(args.ocr_onnx, providers=['CPUExecutionProvider'])
    ocr_input_name = ocr_session.get_inputs()[0].name
    
    dict_path = 'utils/en_dict.txt'
    with open(dict_path, 'r', encoding='utf-8') as f:
        char_list = [line.strip('\n') for line in f.readlines()]
    char_list.append(' ')
    char_list.insert(0, '')

    img_paths = sorted(glob.glob(os.path.join(args.folder, "*.[jJ][pP][gG]")) + 
                       glob.glob(os.path.join(args.folder, "*.[pP][nN][gG]")))

    ocr_times, exact_matches, cer_scores, sim_scores = [], [], [], []

    for idx, path in enumerate(img_paths):
        filename = os.path.basename(path)
        
        # Ground Truth from filename
        gt_text = os.path.splitext(filename)[0].upper()
        gt_text = re.sub(r'[^A-Z0-9]', '', gt_text)

        img = cv.imread(path)
        if img is None: continue

        # Resize and normalize
        img_resized = cv.resize(img, (160, 48)).astype(np.float32) / 255.0
        img_float = (img_resized - 0.5) / 0.5
        img_tensor = np.expand_dims(np.transpose(img_float, (2, 0, 1)), axis=0)

        t0 = time.time()
        preds = ocr_session.run(None, {ocr_input_name: img_tensor})[0]
        
        # CTC Decoding
        pred_idx = preds[0].argmax(axis=1)
        blank_idx = preds.shape[2] - 1
        raw_text = ""
        prev_idx = -1
        for i_idx in pred_idx:
            if i_idx != blank_idx and i_idx != prev_idx:
                if i_idx < len(char_list): raw_text += char_list[i_idx]
            prev_idx = i_idx

        t1 = time.time()
        current_latency_ms = (t1 - t0) * 1000
        ocr_times.append(current_latency_ms)

        detected_text = re.sub(r'[^A-Z0-9]', '', raw_text.upper())
        
        # Calculate Metrics
        is_match = 1 if detected_text == gt_text else 0
        cer = calculate_cer(detected_text, gt_text)
        
        exact_matches.append(is_match)
        cer_scores.append(cer)
        sim_scores.append(max(0.0, 1.0 - cer))

        print(f"[{idx+1}/{len(img_paths)}] GT: {gt_text:<8} | Pred: {detected_text:<8} | Match: {bool(is_match)} | Latency: {current_latency_ms:6.2f} ms")

    print("RECOGNITION BENCHMARK RESULTS")
    print(f"Total Images:           {len(img_paths)}")
    print(f"Exact Match Rate:       {np.mean(exact_matches) * 100:.2f}%")
    print(f"Character Similarity:   {np.mean(sim_scores) * 100:.2f}%")
    print(f"Character Error Rate:   {np.mean(cer_scores) * 100:.2f}%")
    print(f"Average Latency (CPU):  {np.mean(ocr_times):.2f} ms")
