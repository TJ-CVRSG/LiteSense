import os
from datetime import datetime

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision


def init_workspace(output_root, experiment_name, mode):
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    work_path = os.path.join(output_root, f"{experiment_name}-{timestamp}")
    if mode == "train":
        os.makedirs(os.path.join(work_path, "tmp"), exist_ok=True)
        os.makedirs(os.path.join(work_path, "weights"), exist_ok=True)
    return work_path


def save_checkpoint(model, optimizer, epoch, epoch_loss, best_metric, best_epoch,
                    epoch_list, loss_list, loss_eval_list, epochs_since_improvement,
                    early_stop_triggered, checkpoint_path):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": epoch_loss,
        "best_loss": best_metric,
        "best_epoch": best_epoch,
        "epoch_list": epoch_list,
        "loss_list": loss_list,
        "loss_eval_list": loss_eval_list,
        "epochs_since_improvement": epochs_since_improvement,
        "early_stop_triggered": early_stop_triggered,
    }, checkpoint_path)


def load_checkpoint(model, optimizer, checkpoint_path, map_location=None):
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return (
        model,
        optimizer,
        checkpoint["epoch"],
        checkpoint["loss"],
        checkpoint["best_loss"],
        checkpoint.get("best_epoch", 0),
        checkpoint.get("epoch_list", []),
        checkpoint.get("loss_list", []),
        checkpoint.get("loss_eval_list", []),
        checkpoint.get("epochs_since_improvement", 0),
        checkpoint.get("early_stop_triggered", False),
    )


def load_weights(model, weights_path, map_location=None, strict=True):
    weights = torch.load(weights_path, map_location=map_location)
    if isinstance(weights, dict):
        if "model_state_dict" in weights:
            state_dict = weights["model_state_dict"]
        elif "state_dict" in weights:
            state_dict = weights["state_dict"]
        else:
            state_dict = weights
    else:
        state_dict = weights

    if isinstance(state_dict, dict):
        state_dict = {
            key.removeprefix("module."): value
            for key, value in state_dict.items()
        }

    return model.load_state_dict(state_dict, strict=strict)


def save_epoch_visuals(output_path, pred_depth, gt_depth, image):
    torchvision.utils.save_image(pred_depth.detach().cpu(), os.path.join(output_path, "tmp", "epoch_pred_dpt.jpg"), nrow=4, normalize=True)
    torchvision.utils.save_image(gt_depth.detach().cpu(), os.path.join(output_path, "tmp", "epoch_real_dpt.jpg"), nrow=4, normalize=True)
    torchvision.utils.save_image(image.detach().cpu(), os.path.join(output_path, "tmp", "epoch_img.jpg"), nrow=4)


def _tensor_to_numpy_image(tensor):
    array = tensor.detach().cpu()
    if array.ndim == 4:
        array = array[0]
    array = array.numpy()
    if array.ndim == 3:
        array = np.transpose(array, (1, 2, 0))
    return np.squeeze(array)


def _rgb_to_bgr_uint8(image):
    image = np.asarray(image, dtype=np.float32)
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    image = np.clip(image, 0.0, 1.0)
    return (image[:, :, ::-1] * 255.0).astype(np.uint8)


def save_eval_visual(output_path, index, pred_depth, gt_depth=None, image=None):
    eval_dir = os.path.join(output_path, "visuals")
    os.makedirs(eval_dir, exist_ok=True)

    pred = _tensor_to_numpy_image(pred_depth)
    gt = _tensor_to_numpy_image(gt_depth) if gt_depth is not None else np.zeros_like(pred)
    rgb = _tensor_to_numpy_image(image) if image is not None else np.zeros((*pred.shape, 3), dtype=np.float32)

    rgb_vis = _rgb_to_bgr_uint8(rgb)
    pred_vis = colorize_depth(pred)
    gt_vis = colorize_depth(gt)
    error_vis = colorize_depth(np.abs(pred - gt), color="rainbow")

    target_size = (pred_vis.shape[1], pred_vis.shape[0])
    rgb_vis = cv2.resize(rgb_vis, target_size, interpolation=cv2.INTER_LINEAR)
    error_vis = cv2.resize(error_vis, target_size, interpolation=cv2.INTER_NEAREST)
    gt_vis = cv2.resize(gt_vis, target_size, interpolation=cv2.INTER_NEAREST)

    top = np.concatenate([rgb_vis, error_vis], axis=1)
    bottom = np.concatenate([gt_vis, pred_vis], axis=1)
    result = np.concatenate([top, bottom], axis=0)
    cv2.imwrite(os.path.join(eval_dir, f"{index:06d}.jpg"), result)


def colorize_depth(depth, color="Spectral"):
    depth = np.asarray(depth, dtype=np.float32)
    denom = float(depth.max() - depth.min())
    if denom < 1e-6:
        normalized = np.zeros_like(depth, dtype=np.uint8)
    else:
        normalized = ((depth - depth.min()) / denom * 255).astype(np.uint8)
    cmap = matplotlib.colormaps.get_cmap(color)
    return (cmap(normalized)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)


def save_prediction_result(output_dir, stem, pred, save_numpy=True, save_colormap=True):
    os.makedirs(output_dir, exist_ok=True)
    if save_colormap:
        cv2.imwrite(os.path.join(output_dir, f"{stem}_pred_viz.png"), colorize_depth(pred))
    if save_numpy:
        cv2.imwrite(os.path.join(output_dir, f"{stem}_pred.png"), np.array(pred * 1000.0, dtype=np.uint16))


def save_loss_curve(output_path, epoch_list, loss_list, loss_eval_list):
    plt.plot(epoch_list, loss_list, c="#A3CDEA", ls="-")
    if loss_eval_list:
        plt.plot(epoch_list[:len(loss_eval_list)], loss_eval_list, c="#F49568", ls="-")
    plt.savefig(os.path.join(output_path, "weights", "loss.png"))
    plt.clf()
    plt.close()
