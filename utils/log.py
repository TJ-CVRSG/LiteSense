import os

def format_duration(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_loss_items(loss_items, precision=4):
    if not loss_items:
        return ""
    return " | ".join(f"{name}: {value:.{precision}f}" for name, value in loss_items.items())


def log_train_batch(epoch, total_epochs, batch_idx, batch_size, total_batches, dataset_size, loss_value, eta_seconds, loss_items=None):
    msg = (
        f"\r\033[K"
        f"Epoch {epoch:>3}/{total_epochs:>3} | Train: "
        f"[{(batch_idx + 1) * batch_size}/{dataset_size} "
        f"({100. * (batch_idx + 1) / total_batches:.0f}%)] | "
        f"{format_loss_items(loss_items)} | ETA: {format_duration(eta_seconds)} "
    )
    print(msg, end="")


def log_train_epoch(epoch, total_epochs, processed_samples, dataset_size, total_batches, loss_value, elapsed_seconds, log_file, loss_items=None):
    msg = (
        f"Epoch {epoch:>3}/{total_epochs:>3} | Train: "
        f"[{processed_samples}/{dataset_size} "
        f"({100. * processed_samples / dataset_size:.0f}%)] | "
        f"{format_loss_items(loss_items)} | TIME: {format_duration(elapsed_seconds)}"
    )
    print(f"\r{msg}        ")
    log_file.write(msg + "\n")
    log_file.flush()


def log_validate_batch(loss_value, metrics, loss_items=None):
    msg = (
        f"\r\033[K"
        f"     Validate | #Loss: {loss_value:.4f}  "
        f"#ACC-1: {metrics['a1']:.4f}  #RMSE: {metrics['rmse']:.4f}  "
        f"#ABS-REL: {metrics['abs_rel']:.4f} "
    )
    print(msg, end="")


def log_validate_epoch(epoch_loss, epoch_metric, log_file, loss_items=None):
    msg = (
        f"     Validate | #Loss: {epoch_loss:.4f}  "
        f"#ACC-1: {epoch_metric[0]:.4f}  #RMSE: {epoch_metric[1]:.4f}  "
        f"#ABS-REL: {epoch_metric[2]:.4f}"
    )
    print(f"\r{msg}")
    log_file.write(msg + "\n")
    log_file.flush()


def log_early_stop(patience, best_epoch, log_file):
    msg = (
        f"Early stopping triggered after {patience} validation checks "
        f"without improvement. Best epoch at {best_epoch}."
    )
    print(msg)
    log_file.write(msg + "\n")
    log_file.flush()


def save_log_file(output_path):
    return open(os.path.join(output_path, "tmp", "log.txt"), "a", encoding="utf-8")


def log_training_config(args):
    with save_log_file(args.output_path) as log_file:
        print("\n==== Training Config ====")
        log_file.write("==== Training Config ====\n")
        for arg in vars(args):
            value = getattr(args, arg)
            print(f"{arg}: {value}")
            log_file.write(f"{arg}: {value}\n")
        print("=========================\n")
        log_file.write("=========================\n")


def log_metrics(loss, metrics):
    items = [("loss", loss)] + list(metrics.items())
    col_width = 12
    metrics_per_row = 6
    border = "+" + "+".join(["-" * col_width] * metrics_per_row) + "+"

    print("==== Evaluation Results ====")
    for start in range(0, len(items), metrics_per_row):
        row = items[start:start + metrics_per_row]
        padded_row = row + [("", 0.0)] * (metrics_per_row - len(row))
        names = "|" + "|".join(f"{name:^{col_width}}" for name, _ in padded_row) + "|"
        values = "|" + "|".join(
            f"{value:^{col_width}.6f}" if name else " " * col_width
            for name, value in padded_row
        ) + "|"
        print(border)
        print(names)
        print(values)
    print(border)
