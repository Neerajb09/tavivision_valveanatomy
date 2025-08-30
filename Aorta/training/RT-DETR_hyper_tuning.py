from ultralytics import RTDETR
import itertools
import torch
import os

data_yaml = "/mnt/nvme_disk2/User_data/hp927k/cardioVision/Phase2_/Datasets/data.yaml"
base_model = "rtdetr-l.pt"

# Corrected hyperparameter grid
hyperparams_grid = {
    'lr0': [1e-4, 5e-4, 1e-3],
    'freeze': [0, 10, 25],
    'imgsz': [640, 800],
    'mosaic': [True, False],
    'weight_decay': [0.0, 5e-5],
    'lrf': [0.01, 0.1]
}

device = 0 if torch.cuda.is_available() else 'cpu'

# Create a directory to save logs
log_dir = "rtdetr_logs"
os.makedirs(log_dir, exist_ok=True)

# Open the log file
log_file_path = os.path.join(log_dir, "training_metrics_log.txt")

for idx, (lr0,freeze, imgsz, mosaic, weight_decay, lrf) in enumerate(itertools.product(
        hyperparams_grid['lr0'],
        hyperparams_grid['freeze'],
        hyperparams_grid['imgsz'],
        hyperparams_grid['mosaic'],
        hyperparams_grid['weight_decay'],
        hyperparams_grid['lrf'])):

    run_name = f"run_{idx}_lr{lr0}_freeze{freeze}_sz{imgsz}_mosaic{mosaic}_wd{weight_decay}_lrf{lrf}"

    # Load RT-DETR model
    model = RTDETR(base_model)

    # Training (silent mode, no console output)
    model.train(
        data=data_yaml,
        epochs=50,
        imgsz=imgsz,
        lr0=lr0,
        freeze=freeze,
        mosaic=mosaic,
        weight_decay=weight_decay,
        lrf=lrf,
        batch=8,  # fixed suitable batch size
        device=device,
        name=f"{run_name}",
        verbose=False,  # Suppress intermediate training output
        show=False
    )

    # Evaluation
    metrics = model.val()

    # Write metrics to a text file
    with open(log_file_path, "a") as log_file:
        log_file.write(f"Run: {run_name}\n")
        log_file.write(f"mAP50-95: {metrics.box.map:.4f}\n")
        log_file.write(f"mAP50: {metrics.box.map50:.4f}\n")
        log_file.write(f"Precision: {metrics.box.mp:.4f}\n")
        log_file.write(f"Recall: {metrics.box.mr:.4f}\n")
        log_file.write("=" * 50 + "\n")

print("Hyperparameter tuning complete.")
