import glob
import os

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from tqdm import tqdm

from model.litesense import PatchDepthEstimator
from utils.config import parse_args
from utils.data import read_depth, read_rgb, read_tof, simulate_tof, THDR3K_TOF_CONFIGS
from utils.io import init_workspace, load_weights, save_prediction_result


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_zone(args, image_width, image_height):
    x0 = max(args.zone_x, 0)
    y0 = max(args.zone_y, 0)
    zone_width = args.zone_size * args.zone_grid_cols
    zone_height = args.zone_size * args.zone_grid_rows
    x1 = min(x0 + zone_width, image_width)
    y1 = min(y0 + zone_height, image_height)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(
            f"Invalid predict zone: ({x0}, {y0}) -> ({x1}, {y1}) for image size "
            f"({image_width}, {image_height})"
        )
    return [[x0, y0], [x1, y1]]


def preprocess_for_model(image, tof, cnh):
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    image = normalize(transform(image)).unsqueeze(0).to(DEVICE).to(torch.float32)
    tof = transform(tof).unsqueeze(0).to(DEVICE).to(torch.float32)
    cnh = transform(cnh).unsqueeze(0).to(DEVICE).to(torch.float32)
    return image, tof, cnh


@torch.no_grad()
def predict_one(model, image, tof, cnh):
    image_t, tof_t, cnh_t = preprocess_for_model(image, tof, cnh)
    tof_t = F.interpolate(tof_t, size=image_t.shape[-2:], mode="nearest")
    pred = model(image_t, tof_t, cnh_t)
    return pred.squeeze().detach().cpu().numpy()


def find_rgb_files(data_path):
    patterns = ["*_rgb.jpg", "*.jpg", "*.png"]
    files = []
    for pattern in patterns:
        files.extend(glob.glob(os.path.join(data_path, pattern)))
    excluded_suffixes = (
        "_depth.png",
        "_fill_depth.png",
        "_pred.png",
        "_pred_viz.png",
    )
    return sorted({
        path for path in files
        if not path.endswith(excluded_suffixes)
    })


def resolve_sample_paths(rgb_path):
    root, ext = os.path.splitext(rgb_path)
    if root.endswith("_rgb"):
        stem_root = root[:-4]
    else:
        stem_root = root

    candidates = {
        "depth": [
            stem_root + "_fill_depth.png",
        ],
        "tof": [
            stem_root + "_tof.json",
        ],
    }
    depth_path = next((path for path in candidates["depth"] if os.path.exists(path)), None)
    tof_path = next((path for path in candidates["tof"] if os.path.exists(path)), None)
    return stem_root, depth_path, tof_path


def prepare_inputs(args, rgb_path):
    image = read_rgb(rgb_path)
    h, w = image.shape[:2]
    stem_root, depth_path, tof_path = resolve_sample_paths(rgb_path)

    zone = make_zone(args, w, h)

    if args.with_tof_data:
        if tof_path is None:
            raise FileNotFoundError(f"ToF JSON is required but not found for: {rgb_path}")
        tof, cnh = read_tof(tof_path, THDR3K_TOF_CONFIGS)
    else:
        if depth_path is None:
            raise FileNotFoundError(f"Depth image is required for simulated ToF but not found for: {rgb_path}")
        depth = read_depth(depth_path)
        tof, cnh = simulate_tof(
            depth=depth,
            zone=zone,
            rows=args.zone_grid_rows,
            cols=args.zone_grid_cols,
            bins=args.sim_cnh_bins,
            cnh_range_max=args.sim_cnh_range,
            dis_max=args.sim_dis_max,
        )

    x0, y0 = zone[0]
    x1, y1 = zone[1]
    image = image[y0:y1, x0:x1]
    stem = os.path.basename(stem_root)
    return stem, image, tof, cnh


def main():
    args = parse_args(mode="predict")
    args.output_path = init_workspace(args.output_path, "pred", args.mode)

    model = PatchDepthEstimator(
        zone_grid_rows=args.zone_grid_rows,
        zone_grid_cols=args.zone_grid_cols,
    ).to(DEVICE)
    load_weights(model, args.weights, map_location=DEVICE, strict=True)
    model.eval()

    rgb_files = find_rgb_files(args.data)
    if not rgb_files:
        raise FileNotFoundError(f"No image files found in: {args.data}")

    for rgb_path in tqdm(rgb_files, desc="Predict"):
        stem, image, tof, cnh = prepare_inputs(args, rgb_path)
        pred = predict_one(model, image, tof, cnh)
        save_prediction_result(
            args.output_path,
            stem,
            pred,
            save_numpy=args.save_numpy,
            save_colormap=args.save_colormap,
        )

    print(f"Results saved to: {args.output_path}")


if __name__ == "__main__":
    main()
