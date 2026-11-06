"""Evaluation script for computing test-set metrics and exporting plots to docs/."""

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List

from ultralytics import YOLO

from src.data_prep import CLASSES

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PLOT_FILES: List[str] = [
    "confusion_matrix.png",
    "confusion_matrix_normalized.png",
    "PR_curve.png",
    "F1_curve.png",
    "P_curve.png",
    "R_curve.png",
]


def evaluate_model(
    weights_path: Path,
    data_yaml: Path,
    split: str = "test",
    imgsz: int = 640,
    device: str = "cpu",
    docs_dir: Path = Path("docs"),
) -> Dict[str, Any]:
    """Run YOLO validation on the specified split and save plots + metrics to docs/."""
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Model weights not found at {weights_path}. "
            "Place your trained best.pt under weights/best.pt first."
        )
    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset YAML not found at {data_yaml}.")

    model = YOLO(str(weights_path))
    project_dir = Path("runs/eval")
    run_name = f"neu_det_{split}"

    logger.info("Evaluating '%s' on split='%s'...", weights_path, split)
    metrics = model.val(
        data=str(data_yaml),
        split=split,
        imgsz=imgsz,
        device=device,
        project=str(project_dir),
        name=run_name,
        exist_ok=True,
        plots=True,
    )

    per_class_results: List[Dict[str, Any]] = []
    for idx, cls_name in enumerate(CLASSES):
        p, r, ap50, ap = metrics.box.class_result(idx)
        per_class_results.append(
            {
                "class": cls_name,
                "precision": round(float(p), 4),
                "recall": round(float(r), 4),
                "mAP50": round(float(ap50), 4),
                "mAP50_95": round(float(ap), 4),
            }
        )

    summary: Dict[str, Any] = {
        "split": split,
        "weights": str(weights_path),
        "overall": {
            "precision": round(float(metrics.box.mp), 4),
            "recall": round(float(metrics.box.mr), 4),
            "mAP50": round(float(metrics.box.map50), 4),
            "mAP50_95": round(float(metrics.box.map), 4),
        },
        "per_class": per_class_results,
    }

    docs_dir.mkdir(parents=True, exist_ok=True)
    metrics_json_path = docs_dir / "metrics.json"
    metrics_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Saved structured evaluation metrics to %s", metrics_json_path)

    eval_out_dir = Path(metrics.save_dir)
    for plot_name in PLOT_FILES:
        src_plot = eval_out_dir / plot_name
        if src_plot.exists():
            shutil.copy2(src_plot, docs_dir / plot_name)
            logger.info("Copied %s -> %s", plot_name, docs_dir / plot_name)

    return summary


def format_markdown_table(summary: Dict[str, Any]) -> str:
    """Format evaluation summary dictionary into a Markdown table for README.md."""
    lines = [
        "| Class | Precision | Recall | mAP@50 | mAP@50-95 |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ]
    for item in summary["per_class"]:
        lines.append(
            f"| `{item['class']}` | {item['precision']:.3f} | {item['recall']:.3f} "
            f"| {item['mAP50']:.3f} | {item['mAP50_95']:.3f} |"
        )
    ov = summary["overall"]
    lines.append(
        f"| **Overall (All)** | **{ov['precision']:.3f}** | **{ov['recall']:.3f}** "
        f"| **{ov['mAP50']:.3f}** | **{ov['mAP50_95']:.3f}** |"
    )
    return "\n".join(lines)


def main() -> None:
    """CLI entry point for model evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate trained YOLO model on NEU-DET.")
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("weights/best.pt"),
        help="Path to trained YOLO checkpoint.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("configs/data.yaml"),
        help="Path to data.yaml.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate on (default: test).",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Evaluation image size.")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run evaluation on ('cpu' or '0').",
    )
    parser.add_argument(
        "--docs-dir",
        type=Path,
        default=Path("docs"),
        help="Directory to export plots and metrics.json.",
    )
    args = parser.parse_args()

    summary = evaluate_model(
        weights_path=args.weights,
        data_yaml=args.data,
        split=args.split,
        imgsz=args.imgsz,
        device=args.device,
        docs_dir=args.docs_dir,
    )
    print("\n=== Markdown Table for README.md ===\n")
    print(format_markdown_table(summary))


if __name__ == "__main__":
    main()
