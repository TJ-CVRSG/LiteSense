import os
import random
import time

import matplotlib
matplotlib.use("agg")
import numpy as np
import torch
import torch.nn.functional as F

from dataloader.nyu import NYUv2
from dataloader.thdr import THDR3K
from loss.criterion import DepthLoss
from model.litesense import PatchDepthEstimator
from utils.config import parse_args
from utils.io import (
    init_workspace,
    load_checkpoint,
    load_weights,
    save_checkpoint,
    save_epoch_visuals,
    save_loss_curve,
)
from utils.log import (
    log_early_stop,
    log_train_batch,
    log_train_epoch,
    log_validate_batch,
    log_validate_epoch,
    log_training_config,
    save_log_file,
)
from utils.metric import calc_metrics


GLOBAL_SEED = 42


def set_global_seed(seed=GLOBAL_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    seed = worker_seed + worker_id
    random.seed(seed)
    np.random.seed(seed)


def adjust_learning_rate(optimizer, epoch, total_epochs, warmup_epochs, initial_lr, lr_min):
    if warmup_epochs > 0 and epoch < warmup_epochs:
        lr = initial_lr * (epoch + 1) / warmup_epochs
    else:
        anneal_epochs = max(total_epochs - warmup_epochs, 1)
        lr = lr_min + 0.5 * (initial_lr - lr_min) * (
            1 + np.cos(np.pi * (epoch - warmup_epochs) / anneal_epochs)
        )

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def build_dataset(args, mode):
    dataset_name = args.dataset.lower()

    if mode == "train":
        data_path = args.data_path
        data_list = args.data_list
    else:
        data_path = args.data_path_eval
        data_list = args.data_list_eval

    if dataset_name == "nyuv2":
        return NYUv2(
            data_path=data_path,
            data_list=data_list,
            mode=mode,
            input_size=[args.input_width, args.input_height],
            zone_size=args.zone_size,
            zone_grid_rows=args.zone_grid_rows,
            zone_grid_cols=args.zone_grid_cols,
            sim_cnh_bins=args.sim_cnh_bins,
            sim_cnh_range=args.sim_cnh_range,
            sim_dis_max=args.sim_dis_max,
        )
    if dataset_name == "thdr3k":
        return THDR3K(
            data_path=data_path,
            data_list=data_list,
            mode=mode,
            input_size=[args.input_width, args.input_height],
        )
    raise ValueError(f"Unsupported dataset: {args.dataset}")


def build_dataloaders(args):
    generator = torch.Generator()
    generator.manual_seed(GLOBAL_SEED)

    train_data = build_dataset(args, mode="train")
    train_loader = torch.utils.data.DataLoader(
        dataset=train_data,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        worker_init_fn=worker_init_fn,
        generator=generator,
    )

    test_loader = None
    if args.validate:
        test_data = build_dataset(args, mode="test")
        test_loader = torch.utils.data.DataLoader(
            dataset=test_data,
            shuffle=False,
            batch_size=1,
            num_workers=0,
            worker_init_fn=worker_init_fn,
            generator=generator,
        )
    return train_loader, test_loader


@torch.no_grad()
def validate(model, test_loader, log_file, device):
    model.eval()
    criterion = DepthLoss()

    epoch_loss = 0.0
    epoch_loss_items = {}
    epoch_metric = [0.0, 0.0, 0.0]
    batch_num = 0

    for sample in test_loader:
        img = sample["image"].to(device).to(torch.float32)
        dpt_gt = sample["depth-gt"].to(device).to(torch.float32)
        tof = sample["tof"].to(device).to(torch.float32)
        cnh = sample["cnh"].to(device).to(torch.float32)

        tof = F.interpolate(tof, size=img.shape[-2:], mode="nearest")
        dpt_pred = model(img, tof, cnh)
        loss, loss_items = criterion(dpt_pred, dpt_gt)

        dpt_pred_np = dpt_pred.squeeze().detach().cpu().numpy()
        dpt_gt_np = dpt_gt.squeeze().detach().cpu().numpy()
        metrics = calc_metrics(dpt_pred_np, dpt_gt_np, mask=(dpt_gt_np > 0))
        if metrics is None:
            continue

        loss_value = loss.detach().item()
        loss_item_values = {name: value.detach().item() for name, value in loss_items.items()}
        log_validate_batch(loss_value, metrics, loss_items=loss_item_values)
        epoch_loss += loss_value
        for name, value in loss_item_values.items():
            epoch_loss_items[name] = epoch_loss_items.get(name, 0.0) + value
        epoch_metric[0] += metrics["a1"]
        epoch_metric[1] += metrics["rmse"]
        epoch_metric[2] += metrics["abs_rel"]
        batch_num += 1

    if batch_num == 0:
        raise RuntimeError("No valid validation samples: all samples have empty valid depth masks")

    epoch_loss = epoch_loss / batch_num
    epoch_loss_items = {name: value / batch_num for name, value in epoch_loss_items.items()}
    epoch_metric = np.array(epoch_metric) / batch_num
    log_validate_epoch(epoch_loss, epoch_metric, log_file, loss_items=epoch_loss_items)
    return epoch_loss, epoch_metric


def train(model, train_loader, optimizer, criterion, device, epoch, total_epochs, output_path, log_file,
          warmup_epochs, initial_lr, lr_min):
    model.train()
    epoch_start_time = time.time()
    epoch_loss = 0.0
    epoch_loss_items = {}
    batch_num = 0

    adjust_learning_rate(
        optimizer,
        epoch,
        total_epochs=total_epochs,
        warmup_epochs=warmup_epochs,
        initial_lr=initial_lr,
        lr_min=lr_min,
    )

    for idx, sample in enumerate(train_loader):
        optimizer.zero_grad()

        img = sample["image"].to(device).to(torch.float32)
        dpt_gt = sample["depth"].to(device).to(torch.float32)
        tof = sample["tof"].to(device).to(torch.float32)
        cnh = sample["cnh"].to(device).to(torch.float32)

        tof = F.interpolate(tof, size=img.shape[-2:], mode="nearest")
        dpt_pred = model(img, tof, cnh)
        loss, loss_items = criterion(dpt_pred, dpt_gt)

        loss.backward()
        optimizer.step()

        loss_value = loss.detach().item()
        loss_item_values = {name: value.detach().item() for name, value in loss_items.items()}
        epoch_loss += loss_value
        for name, value in loss_item_values.items():
            epoch_loss_items[name] = epoch_loss_items.get(name, 0.0) + value
        batch_num += 1

        eta_seconds = ((time.time() - epoch_start_time) / (idx + 1) * (len(train_loader) - idx - 1))
        log_train_batch(
            epoch=epoch,
            total_epochs=total_epochs,
            batch_idx=idx,
            batch_size=len(img),
            total_batches=len(train_loader),
            dataset_size=len(train_loader.dataset),
            loss_value=loss_value,
            loss_items=loss_item_values,
            eta_seconds=eta_seconds,
        )

    save_epoch_visuals(output_path, dpt_pred, dpt_gt, img)

    epoch_loss = epoch_loss / batch_num
    epoch_loss_items = {name: value / batch_num for name, value in epoch_loss_items.items()}
    elapsed = time.time() - epoch_start_time
    progress = int((idx + 1) / len(train_loader) * len(train_loader.dataset))
    log_train_epoch(
        epoch=epoch,
        total_epochs=total_epochs,
        processed_samples=progress,
        dataset_size=len(train_loader.dataset),
        total_batches=len(train_loader),
        loss_value=epoch_loss,
        loss_items=epoch_loss_items,
        elapsed_seconds=elapsed,
        log_file=log_file,
    )
    return epoch_loss


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PatchDepthEstimator(
        zone_grid_rows=args.zone_grid_rows,
        zone_grid_cols=args.zone_grid_cols,
    ).to(device)

    if args.pretrain is not None:
        if not os.path.exists(args.pretrain):
            raise FileNotFoundError(f"Pretrained weights not found: {args.pretrain}")
        load_weights(model, args.pretrain, map_location=device, strict=True)
        print(f"Loaded pretrained weights: {args.pretrain}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
    )
    criterion = DepthLoss()
    train_loader, test_loader = build_dataloaders(args)

    epoch_list = []
    loss_eval_list = []
    loss_list = []
    best_metric = np.inf
    best_epoch = 0
    epochs_since_improvement = 0
    early_stop_patience = max(getattr(args, "early_stop_patience", 0), 0)
    early_stop_triggered = False
    checkpoint_path = os.path.join(args.output_path, "weights", "checkpoint.pth")

    start_epoch = 0
    if args.resume is not None:
        if not os.path.exists(args.resume):
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")
        print(f"Resuming from checkpoint: {args.resume}")
        (
            model,
            optimizer,
            checkpoint_epoch,
            _,
            best_metric,
            best_epoch,
            epoch_list,
            loss_list,
            loss_eval_list,
            epochs_since_improvement,
            early_stop_triggered,
        ) = load_checkpoint(model, optimizer, args.resume, map_location=device)
        start_epoch = checkpoint_epoch + 1

    with save_log_file(args.output_path) as log_file:
        for epoch in range(start_epoch, args.epoch):
            epoch_loss = train(
                model=model,
                train_loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                epoch=epoch,
                total_epochs=args.epoch,
                output_path=args.output_path,
                log_file=log_file,
                warmup_epochs=args.warmup_epochs,
                initial_lr=args.lr,
                lr_min=args.lr_min,
            )
            epoch_list.append(epoch + 1)
            loss_list.append(epoch_loss)

            if args.validate and test_loader is not None:
                loss_eval, eval_metrics = validate(model, test_loader, log_file, device)
                loss_eval_list.append(loss_eval)

                current_metric = eval_metrics[1]
                if best_metric - current_metric > 0.002:
                    best_metric = current_metric
                    epochs_since_improvement = 0
                    best_epoch = epoch
                    torch.save(model.state_dict(), os.path.join(args.output_path, "weights", "best.pt"))
                else:
                    epochs_since_improvement += 1
                    if early_stop_patience > 0 and epochs_since_improvement >= early_stop_patience:
                        log_early_stop(early_stop_patience, best_epoch, log_file)
                        early_stop_triggered = True

            save_loss_curve(args.output_path, epoch_list, loss_list, loss_eval_list)
            save_checkpoint(
                model,
                optimizer,
                epoch,
                epoch_loss,
                best_metric,
                best_epoch,
                epoch_list,
                loss_list,
                loss_eval_list,
                epochs_since_improvement,
                early_stop_triggered,
                checkpoint_path,
            )

            if early_stop_triggered:
                break


def main():
    set_global_seed()
    args = parse_args()
    args.output_path = init_workspace(args.output_path, args.name, args.mode)
    log_training_config(args)

    start_time = time.time()
    run(args)
    end_time = time.time()

    with save_log_file(args.output_path) as log_file:
        msg = f"Training process finished in {(end_time - start_time) // 60} minutes."
        print(msg)
        log_file.write(msg)


if __name__ == "__main__":
    main()
