import argparse

def build_parser():
    parser = argparse.ArgumentParser()

    # -------------------------------------------------------------------------
    # Common arguments
    # -------------------------------------------------------------------------
    parser.add_argument("--mode", type=str, choices=["train", "evaluate", "predict"], default="train", help="Run mode")
    parser.add_argument("--config", type=str, default=None, help="Path to config.txt file (optional)")
    parser.add_argument("--name", type=str, default="exp-litesense", help="Experiment name")
    parser.add_argument("--input-width", type=int, default=544, help="Maximum source data width used to validate the ToF zone")
    parser.add_argument("--input-height", type=int, default=416, help="Maximum source data height used to validate the ToF zone")
    parser.add_argument("--output-path", type=str, default="./Runs", help="Path to save the output")
    parser.add_argument("--weights", type=str, default=None, help="Path to model weights for evaluation or prediction")

    # -------------------------------------------------------------------------
    # Train arguments
    # -------------------------------------------------------------------------
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--epoch", type=int, default=100, help="Number of epochs for training")
    parser.add_argument("--validate", action=argparse.BooleanOptionalAction, default=True, help="Whether to validate during training")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint for resuming training")
    parser.add_argument("--pretrain", type=str, default=None, help="Path to pretrained model weights (.pt/.pth) for model initialization")
    parser.add_argument("--early-stop-patience", type=int, default=20, help="Validation checks without improvement before early stopping. Set to 0 to disable.")

    # -------------------------------------------------------------------------
    # Dataloader benchmark arguments
    # -------------------------------------------------------------------------
    parser.add_argument("--num-workers", type=int, default=8, help="Workers for dataloader")
    parser.add_argument("--dataset", type=str, default="NYUv2", help="Dataset type")
    parser.add_argument("--data-path", type=str, default="./Datasets/dataset_nyu", help="Path to the training dataset")
    parser.add_argument("--data-list", type=str, default="./Datasets/dataset_nyu/nyu_24k.json", help="File to the training dataset list")
    parser.add_argument("--data-path-eval", type=str, default="./Datasets/dataset_nyu", help="Path to the test dataset")
    parser.add_argument("--data-list-eval", type=str, default="./Datasets/dataset_nyu/nyu_24k.json", help="File to the test dataset list")
    parser.add_argument("--save-error", action=argparse.BooleanOptionalAction, default=True, help="Whether to save evaluation visualizations")

    # -------------------------------------------------------------------------
    # Predict arguments
    # -------------------------------------------------------------------------
    parser.add_argument("--data", type=str, default=None, help="Path to the folder containing input data for prediction")
    parser.add_argument("--with-tof-data", action=argparse.BooleanOptionalAction, default=False, help="Whether to use real ToF data for prediction")
    parser.add_argument("--save-numpy", action=argparse.BooleanOptionalAction, default=False, help="Save uint16 prediction depth")
    parser.add_argument("--save-colormap", action=argparse.BooleanOptionalAction, default=True, help="Save colorized prediction image")

    # -------------------------------------------------------------------------
    # Optimizer / scheduler hyperparameters
    # -------------------------------------------------------------------------
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--lr-min", type=float, default=1e-5, help="Minimum learning rate for cosine annealing")
    parser.add_argument("--warmup-epochs", type=int, default=5, help="Number of warmup epochs")
    parser.add_argument("--weight-decay", type=float, default=1e-2, help="AdamW weight decay")
    parser.add_argument("--adam-beta1", type=float, default=0.9, help="AdamW beta1")
    parser.add_argument("--adam-beta2", type=float, default=0.999, help="AdamW beta2")
    parser.add_argument("--adam-eps", type=float, default=1e-8, help="AdamW epsilon")

    # -------------------------------------------------------------------------
    # ToF / zone arguments
    # -------------------------------------------------------------------------
    parser.add_argument("--zone-x", type=int, default=60, help="ToF zone left x for prediction")
    parser.add_argument("--zone-y", type=int, default=0, help="ToF zone top y for prediction")

    parser.add_argument("--zone-size", type=int, default=52, help="Size of each zone cell in pixels")
    parser.add_argument("--zone-grid-rows", type=int, default=8, help="Number of zone grid rows")
    parser.add_argument("--zone-grid-cols", type=int, default=8, help="Number of zone grid columns")
    parser.add_argument("--sim-cnh-bins", type=int, default=18, help="Number of CNH histogram bins")
    parser.add_argument("--sim-cnh-range", type=float, default=5.4, help="CNH histogram range in meters")
    parser.add_argument("--sim-dis-max", type=float, default=4.0, help="Maximum distance in meters")
    
    return parser

def _load_config(path):
    args_list = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("//"):
                continue
            args_list.extend(line.split())
    return args_list

def _validate_args(args):
    assert not (args.resume and args.pretrain), "--resume and --pretrain should not be used together"
    if args.mode in ["evaluate", "predict"]:
        assert args.weights is not None, "--weights is required when mode is evaluate or predict"
        
    assert args.input_width > 0, "input_width must be positive"
    assert args.input_height > 0, "input_height must be positive"

    assert args.zone_grid_rows > 0 and args.zone_grid_cols > 0, "Zone grid dimensions must be positive integers"
    assert args.zone_size > 0, "zone_size must be a positive integer"
    assert args.zone_size % 4 == 0, f"zone_size ({args.zone_size}) must be divisible by 4"

    zone_total_width = args.zone_size * args.zone_grid_cols
    zone_total_height = args.zone_size * args.zone_grid_rows
    assert zone_total_width <= args.input_width, f"Zone total width ({zone_total_width}) cannot exceed input width limit ({args.input_width})"
    assert zone_total_height <= args.input_height, f"Zone total height ({zone_total_height}) cannot exceed input height limit ({args.input_height})"
    assert zone_total_width % 16 == 0, f"Zone total width ({zone_total_width}) must be divisible by 16"
    assert zone_total_height % 16 == 0, f"Zone total height ({zone_total_height}) must be divisible by 16"

def parse_args(mode=None):
    parser = build_parser()
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=str)
    config_args, remaining = config_parser.parse_known_args()
    cfg_args = _load_config(config_args.config) if config_args.config else []
    mode_args = ["--mode", mode] if mode is not None else []
    args = parser.parse_args(
        cfg_args
        + remaining
        + (["--config", config_args.config] if config_args.config else [])
        + mode_args
    )
    _validate_args(args)
    return args
