"""Training script for Ultralytics YOLO on the prepared NEU-DET dataset."""

import argparse
import logging
import shutil
from pathlib import Path

from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def train_model(
    data_yaml: Path,
    base_weights: str = "yolo11n.pt",
    epochs: int = 80,
    imgsz: int = 640,
    batch: int = 16,
    device: str = "0",
    project: str = "runs/detect",
    name: str = "neu_det_yolo11n",
    seed: int = 42,
    export_best_to: Path = Path("weights/best.pt"),
) -> Path:
    """Train a YOLO model on NEU-DET and copy the best checkpoint to weights/best.pt."""
    if not data_yaml.exists():
        raise FileNotFoundError(
            f"Dataset YAML not found: {data_yaml}. Run 'python -m src.data_prep' first."
        )

    logger.info("Initializing YOLO model from '%s'...", base_weights)
    model = YOLO(base_weights)

    logger.info("Starting training for %d epochs (imgsz=%d, batch=%d)...", epochs, imgsz, batch)
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=name,
        seed=seed,
        pretrained=True,
        patience=20,
        plots=True,
    )

    best_ckpt = Path(project) / name / "weights" / "best.pt"
    if best_ckpt.exists():
        export_best_to.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_ckpt, export_best_to)
        logger.info("Copied best checkpoint to %s", export_best_to)
        return export_best_to

    logger.warning("Could not locate expected checkpoint at %s", best_ckpt)
    return best_ckpt


def main() -> None:
    """CLI entry point for training YOLO on NEU-DET."""
    parser = argparse.ArgumentParser(description="Train YOLO model on NEU-DET dataset.")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("configs/data.yaml"),
        help="Path to Ultralytics data.yaml file.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="yolo11n.pt",
        help="Base YOLO model weights (e.g. yolo11n.pt or yolo11s.pt).",
    )
    parser.add_argument("--epochs", type=int, default=80, help="Number of training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size.")
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Training device ('0' for first GPU, 'cpu' for CPU).",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="neu_det_yolo11n",
        help="Experiment run name under runs/detect/.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()

    train_model(
        data_yaml=args.data,
        base_weights=args.weights,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        name=args.name,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
