"""Data preparation script for converting NEU-DET Pascal VOC XML annotations to YOLO format.

Handles nested or flat NEU-DET directory layouts, converts bounding boxes to normalized
YOLO coordinates, and performs a deterministic stratified train/val/test split (70/15/15).
"""

import argparse
import logging
import random
import shutil
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

CLASSES: List[str] = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]
CLASS_TO_ID: Dict[str, int] = {cls_name: idx for idx, cls_name in enumerate(CLASSES)}
IMAGE_EXTENSIONS: Tuple[str, ...] = (".jpg", ".jpeg", ".bmp", ".png")


def convert_bbox_voc_to_yolo(
    img_width: int,
    img_height: int,
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
) -> Tuple[float, float, float, float]:
    """Convert Pascal VOC (xmin, ymin, xmax, ymax) box to normalized YOLO (x_center, y_center, w, h)."""
    xmin = max(0.0, min(float(img_width), xmin))
    xmax = max(0.0, min(float(img_width), xmax))
    ymin = max(0.0, min(float(img_height), ymin))
    ymax = max(0.0, min(float(img_height), ymax))

    box_w = max(0.0, xmax - xmin)
    box_h = max(0.0, ymax - ymin)
    x_center = xmin + box_w / 2.0
    y_center = ymin + box_h / 2.0

    return (
        x_center / img_width,
        y_center / img_height,
        box_w / img_width,
        box_h / img_height,
    )


def parse_voc_xml(xml_path: Path) -> Tuple[List[str], Optional[str]]:
    """Parse a single Pascal VOC XML file and return YOLO label lines and primary class name."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size_node = root.find("size")
    if size_node is None:
        raise ValueError(f"Missing <size> tag in {xml_path}")

    width = int(size_node.findtext("width", default="0"))
    height = int(size_node.findtext("height", default="0"))
    if width <= 0 or height <= 0:
        # NEU-DET images are standard 200x200 if XML metadata has 0
        width, height = 200, 200

    yolo_lines: List[str] = []
    primary_class: Optional[str] = None

    for obj in root.findall("object"):
        raw_name = obj.findtext("name", default="").strip().lower()
        # Normalize spaces to underscores in case an annotation uses "pitted surface"
        cls_name = raw_name.replace(" ", "_")
        if cls_name not in CLASS_TO_ID:
            logger.warning("Skipping unknown class '%s' in %s", cls_name, xml_path.name)
            continue

        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue

        xmin = float(bndbox.findtext("xmin", default="0"))
        ymin = float(bndbox.findtext("ymin", default="0"))
        xmax = float(bndbox.findtext("xmax", default="0"))
        ymax = float(bndbox.findtext("ymax", default="0"))

        if xmax <= xmin or ymax <= ymin:
            logger.warning("Skipping invalid bounding box in %s", xml_path.name)
            continue

        xc, yc, w, h = convert_bbox_voc_to_yolo(width, height, xmin, ymin, xmax, ymax)
        cls_id = CLASS_TO_ID[cls_name]
        yolo_lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

        if primary_class is None:
            primary_class = cls_name

    return yolo_lines, primary_class


def infer_class_from_stem(stem: str) -> str:
    """Infer NEU-DET defect class from filename stem (e.g. 'crazing_12' -> 'crazing')."""
    lower_stem = stem.lower()
    for cls_name in CLASSES:
        if lower_stem.startswith(cls_name) or lower_stem.startswith(cls_name.replace("_", "-")):
            return cls_name
    return "unknown"


def discover_image_xml_pairs(raw_dir: Path) -> List[Tuple[Path, Path]]:
    """Recursively find matching (image_path, xml_path) pairs inside raw_dir.

    Supports both flat NEU-DET layouts (IMAGES/ and ANNOTATIONS/) and pre-split or
    class-subfolder layouts (train/images/crazing/*.jpg, etc.).
    """
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"Raw dataset directory not found: {raw_dir}. "
            "Please download NEU-DET and place it under this path."
        )

    image_map: Dict[str, Path] = {}
    for file_path in raw_dir.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS:
            image_map[file_path.stem.lower()] = file_path

    xml_map: Dict[str, Path] = {}
    for file_path in raw_dir.rglob("*.xml"):
        if file_path.is_file():
            xml_map[file_path.stem.lower()] = file_path

    common_stems = sorted(set(image_map.keys()) & set(xml_map.keys()))
    if not common_stems:
        raise RuntimeError(
            f"No matching image and XML pairs found under {raw_dir}. "
            f"Found {len(image_map)} images and {len(xml_map)} XML files."
        )

    missing_xml = len(image_map) - len(common_stems)
    if missing_xml > 0:
        logger.warning("%d images did not have a matching XML file and were skipped.", missing_xml)

    return [(image_map[stem], xml_map[stem]) for stem in common_stems]


def stratified_split(
    pairs: List[Tuple[Path, Path]],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, List[Tuple[Path, Path]]]:
    """Split (image, xml) pairs into train, val, and test sets stratified by defect class."""
    grouped: Dict[str, List[Tuple[Path, Path]]] = defaultdict(list)
    for img_path, xml_path in pairs:
        cls_key = infer_class_from_stem(img_path.stem)
        grouped[cls_key].append((img_path, xml_path))

    rng = random.Random(seed)
    splits: Dict[str, List[Tuple[Path, Path]]] = {"train": [], "val": [], "test": []}

    for cls_key in sorted(grouped.keys()):
        cls_items = sorted(grouped[cls_key], key=lambda item: item[0].stem)
        rng.shuffle(cls_items)

        n_total = len(cls_items)
        n_train = int(round(n_total * train_ratio))
        n_val = int(round(n_total * val_ratio))

        splits["train"].extend(cls_items[:n_train])
        splits["val"].extend(cls_items[n_train : n_train + n_val])
        splits["test"].extend(cls_items[n_train + n_val :])

    return splits


def prepare_dataset(
    raw_dir: Path,
    output_dir: Path,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
    clean: bool = True,
) -> Dict[str, int]:
    """Convert NEU-DET dataset from VOC XML to YOLO directory structure."""
    pairs = discover_image_xml_pairs(raw_dir)
    logger.info("Discovered %d valid image-XML pairs in %s", len(pairs), raw_dir)

    splits = stratified_split(pairs, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed)

    if clean and output_dir.exists():
        shutil.rmtree(output_dir)

    summary_counts: Dict[str, int] = {}
    for split_name, split_pairs in splits.items():
        img_out_dir = output_dir / "images" / split_name
        lbl_out_dir = output_dir / "labels" / split_name
        img_out_dir.mkdir(parents=True, exist_ok=True)
        lbl_out_dir.mkdir(parents=True, exist_ok=True)

        written = 0
        for img_path, xml_path in split_pairs:
            yolo_lines, _ = parse_voc_xml(xml_path)
            shutil.copy2(img_path, img_out_dir / img_path.name)
            label_file = lbl_out_dir / f"{img_path.stem}.txt"
            label_file.write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""), encoding="utf-8")
            written += 1

        summary_counts[split_name] = written
        logger.info("Split '%s': %d samples written.", split_name, written)

    return summary_counts


def main() -> None:
    """CLI entry point for dataset preparation."""
    parser = argparse.ArgumentParser(description="Prepare NEU-DET dataset in YOLO format.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/config.yaml"),
        help="Path to project config.yaml file.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help="Override path to raw NEU-DET directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override path to processed YOLO dataset directory.",
    )
    args = parser.parse_args()

    cfg = {}
    if args.config.exists() and yaml is not None:
        with args.config.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    ds_cfg = cfg.get("dataset", {})
    raw_dir = args.raw_dir or Path(ds_cfg.get("raw_dir", "data/raw/NEU-DET"))
    output_dir = args.output_dir or Path(ds_cfg.get("processed_dir", "data/processed/NEU-DET"))
    seed = int(ds_cfg.get("seed", 42))
    ratios = ds_cfg.get("split_ratios", {})
    train_ratio = float(ratios.get("train", 0.70))
    val_ratio = float(ratios.get("val", 0.15))

    prepare_dataset(
        raw_dir=raw_dir,
        output_dir=output_dir,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
    )


if __name__ == "__main__":
    main()
