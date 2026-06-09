import numpy as np

def calc_metrics(pred, target, mask):
    eps = 1e-6
    valid = mask & (pred > eps) & (target > eps)
    if not np.any(valid):
        return None

    pred = pred[valid]
    target = target[valid]
    
    # Accuracy
    thresh = np.maximum((target / pred), (pred / target))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    # Relative Error
    abs_rel = np.mean(np.abs(target - pred) / target)
    sq_rel = np.mean(((target - pred) ** 2) / target)

    # MAE & MSE
    mae = np.mean(np.abs(target - pred))
    mse = np.mean((target - pred) ** 2)

    # RMSE
    rmse = (target - pred) ** 2
    rmse = np.sqrt(rmse.mean())
    
    # Log RMS
    rmse_log = (np.log(target) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    # SILog
    err = np.log(pred) - np.log(target)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    # Log MAE
    log_10 = (np.abs(np.log10(target) - np.log10(pred))).mean()

    metrics = {
        "a1": a1,
        "a2": a2,
        "a3": a3,
        "abs_rel": abs_rel,
        "sq_rel": sq_rel,
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "rmse_log": rmse_log,
        "silog": silog,
        "log_10": log_10
    }

    return metrics
