import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from loss.criterion import DepthLoss
from model.litesense import PatchDepthEstimator
from train import build_dataset
from utils.config import parse_args
from utils.log import log_metrics
from utils.io import init_workspace, load_weights, save_eval_visual
from utils.metric import calc_metrics


@torch.no_grad()
def evaluate(model, eval_loader, device, output_path=None, save_error=True):
    model.eval()
    criterion = DepthLoss()

    losses = []
    metrics_list = []
    saved_visuals = 0

    for idx, sample in enumerate(tqdm(eval_loader, desc="Evaluate")):
        img = sample["image"].to(device).to(torch.float32)
        dpt_gt = sample["depth-gt"].to(device).to(torch.float32)
        tof = sample["tof"].to(device).to(torch.float32)
        cnh = sample["cnh"].to(device).to(torch.float32)

        tof = F.interpolate(tof, size=img.shape[-2:], mode="nearest")
        dpt_pred = model(img, tof, cnh)
        loss, _ = criterion(dpt_pred, dpt_gt)

        pred_np = dpt_pred.squeeze().detach().cpu().numpy()
        gt_np = dpt_gt.squeeze().detach().cpu().numpy()
        metrics = calc_metrics(pred_np, gt_np, mask=(gt_np > 0))
        if metrics is None:
            continue

        losses.append(loss.detach().item())
        metrics_list.append(metrics)

        should_save = save_error and output_path is not None
        if should_save:
            save_eval_visual(output_path, idx, dpt_pred, dpt_gt, img)
            saved_visuals += 1

    if not metrics_list:
        raise RuntimeError("No valid evaluation samples: all samples have empty valid depth masks")

    avg_metrics = {
        key: float(np.mean([metrics[key] for metrics in metrics_list]))
        for key in metrics_list[0]
    }
    return float(np.mean(losses)), avg_metrics


def main():
    args = parse_args(mode="evaluate")
    args.output_path = init_workspace(args.output_path, "eval", args.mode)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PatchDepthEstimator(
        zone_grid_rows=args.zone_grid_rows,
        zone_grid_cols=args.zone_grid_cols,
    ).to(device)

    if not os.path.exists(args.weights):
        raise FileNotFoundError(f"Evaluation weights not found: {args.weights}")
    load_weights(model, args.weights, map_location=device, strict=True)

    eval_dataset = build_dataset(args, mode="test")
    eval_loader = DataLoader(
        dataset=eval_dataset,
        shuffle=False,
        batch_size=1,
        num_workers=args.num_workers,
    )

    loss, metrics = evaluate(
        model=model,
        eval_loader=eval_loader,
        device=device,
        output_path=args.output_path,
        save_error=args.save_error,
    )
    log_metrics(loss, metrics)


if __name__ == "__main__":
    main()
