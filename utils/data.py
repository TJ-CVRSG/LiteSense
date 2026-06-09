import json
from json import JSONDecodeError

import cv2
import numpy as np

THDR3K_TOF_CONFIGS = {
    "col": 8,
    "row": 8,
    "bin": 18,
    "sx": 60,
    "sy": 0,
    "size": 480
}

def read_rgb(path):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image.astype(np.float32) / 255.0


def read_depth(path):
    depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(f"Cannot read depth: {path}")
    return depth.astype(np.float32) / 1000.0


def read_tof(path, tof_configs=THDR3K_TOF_CONFIGS):
    try:
        with open(path, "r", encoding="utf-8") as f:
            tof_data = json.load(f)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Cannot read ToF JSON: {path}") from exc
    except JSONDecodeError as exc:
        raise ValueError(f"Invalid ToF JSON file: {path}") from exc

    rows = tof_configs["row"]
    cols = tof_configs["col"]
    bins = tof_configs["bin"]
    zone_count = rows * cols

    try:
        distance_mm = tof_data["distance_mm"]
        nb_target_detected = tof_data["nb_target_detected"]
        cnh_values = tof_data["cnh_data"]["bin_values"]
    except KeyError as exc:
        raise KeyError(f"Missing ToF JSON field {exc!s} in: {path}") from exc

    if len(distance_mm) != zone_count:
        raise ValueError(
            f"Invalid distance_mm length in {path}: expected {zone_count}, got {len(distance_mm)}"
        )
    if len(nb_target_detected) != zone_count:
        raise ValueError(
            f"Invalid nb_target_detected length in {path}: expected {zone_count}, got {len(nb_target_detected)}"
        )
    if len(cnh_values) != zone_count * bins:
        raise ValueError(
            f"Invalid cnh_data.bin_values length in {path}: "
            f"expected {zone_count * bins}, got {len(cnh_values)}"
        )

    tof = np.array(distance_mm, dtype=np.float32).reshape(rows, cols)
    mask = np.array(nb_target_detected, dtype=np.float32).reshape(rows, cols)
    tof[mask == 0] = 0
    tof = tof / 1000.0

    cnh = np.array(cnh_values, dtype=np.float32).reshape(zone_count, bins)
    cnh[cnh < 0] = 0
    row_sums = cnh.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cnh = (cnh / row_sums).reshape(rows, cols, bins)
    
    return np.rot90(tof, k=1).copy(), np.rot90(cnh, k=1).copy()


def simulate_tof(depth, zone, rows=8, cols=8, bins=18, cnh_range_max=5.4,
                 dis_max=4.0, filter_invalid=True, valid_depth_min=1e-3):
    x0, y0 = zone[0]
    x1, y1 = zone[1]
    zone_depth = depth[y0:y1, x0:x1]
    cell_h = zone_depth.shape[0] // rows
    cell_w = zone_depth.shape[1] // cols
    
    def calculate_histogram_and_peak(data):
        data = data.flatten()
        if filter_invalid:
            data = data[data > valid_depth_min]
        if data.size == 0:
            return 0.0, np.zeros((bins,), dtype=np.float32)

        hist, _ = np.histogram(data, bins=bins, range=[0, cnh_range_max])

        data_scaled = np.round(data * 1000).astype(int)
        counts = np.bincount(data_scaled)
        peak_idx = np.argmax(counts)
        peak_value = peak_idx / 1000.0

        if np.sum(hist) != 0:
            hist = hist / np.sum(hist)
        if peak_value > dis_max:
            peak_value = 0.0
        return peak_value, hist.astype(np.float32)

    tof = np.zeros((rows, cols), dtype=np.float32)
    cnh = np.zeros((rows, cols, bins), dtype=np.float32)
    for row in range(rows):
        for col in range(cols):
            sy = row * cell_h
            sx = col * cell_w
            ey = zone_depth.shape[0] if row == rows - 1 else sy + cell_h
            ex = zone_depth.shape[1] if col == cols - 1 else sx + cell_w
            tof[row, col], cnh[row, col] = calculate_histogram_and_peak(zone_depth[sy:ey, sx:ex])

    return tof, cnh
