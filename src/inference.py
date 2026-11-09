"""YOLO inference module for steel surface defect detection.

Loads precomputed detection cache for the 10 preset scenarios (so switching images and
adjusting the confidence slider takes <5ms even on 0.1 vCPU cloud instances), and lazily
loads the trained YOLOv11 model when a user uploads a brand-new custom image.
"""

import gc
import hashlib
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]

from src.data_prep import CLASSES

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Distinct RGB colors for each of the 6 NEU-DET defect classes (no orange used)
CLASS_COLORS: Dict[str, Tuple[int, int, int]] = {
    "crazing": (230, 57, 70),
    "inclusion": (37, 99, 235),
    "patches": (42, 157, 143),
    "pitted_surface": (233, 196, 106),
    "rolled-in_scale": (99, 102, 241),
    "scratches": (138, 201, 38),
}


def load_config(config_path: Path = Path("configs/config.yaml")) -> Dict[str, Any]:
    """Load YAML configuration file."""
    if not config_path.exists() or yaml is None:
        return {}
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def describe_region(xc_norm: float, yc_norm: float) -> str:
    """Convert normalized (x_center, y_center) coordinates into a human-readable spatial region."""
    vert = "top" if yc_norm < 0.33 else ("bottom" if yc_norm > 0.67 else "middle")
    horiz = "left" if xc_norm < 0.33 else ("right" if xc_norm > 0.67 else "center")
    if vert == "middle" and horiz == "center":
        return "center"
    return f"{vert}-{horiz}"


def compute_image_fingerprint(pil_img: Image.Image) -> np.ndarray:
    """Compute a 16x16 grayscale fingerprint resilient to browser JPEG re-compression."""
    small = pil_img.resize((16, 16), Image.Resampling.BILINEAR).convert("L")
    return np.asarray(small, dtype=np.float32).flatten()


def build_structured_payload(
    detections: List[Dict[str, Any]],
    conf_threshold: float,
    image_size: Tuple[int, int],
    using_custom_weights: bool = True,
) -> Dict[str, Any]:
    """Aggregate raw detection list into a structured JSON summary for LLM/rule-based reporting."""
    width, height = image_size
    class_stats: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {"count": 0, "confidences": [], "regions": []}
    )

    for det in detections:
        cls_name = str(det["class_name"])
        conf = float(det["confidence"])
        bbox = det["bbox_xyxy"]
        xc_norm = ((bbox[0] + bbox[2]) / 2.0) / max(1, width)
        yc_norm = ((bbox[1] + bbox[3]) / 2.0) / max(1, height)
        region = det.get("region") or describe_region(xc_norm, yc_norm)

        class_stats[cls_name]["count"] += 1
        class_stats[cls_name]["confidences"].append(round(conf, 4))
        class_stats[cls_name]["regions"].append(region)

    summary_by_class: Dict[str, Dict[str, Any]] = {}
    for cls_name, stats in sorted(class_stats.items()):
        confs = stats["confidences"]
        summary_by_class[cls_name] = {
            "count": stats["count"],
            "mean_confidence": round(sum(confs) / len(confs), 4),
            "max_confidence": round(max(confs), 4),
            "min_confidence": round(min(confs), 4),
            "regions": sorted(set(stats["regions"])),
        }

    return {
        "total_detections": len(detections),
        "confidence_threshold": round(float(conf_threshold), 2),
        "image_size": {"width": width, "height": height},
        "using_trained_neu_weights": using_custom_weights,
        "summary_by_class": summary_by_class,
        "detections": detections,
    }


def draw_detections_pil(image: Image.Image, detections: List[Dict[str, Any]]) -> Image.Image:
    """Draw bounding boxes and labels onto a copy of a PIL RGB image."""
    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)

    for det in detections:
        cls_name = str(det["class_name"])
        conf = float(det["confidence"])
        x1, y1, x2, y2 = [int(round(v)) for v in det["bbox_xyxy"]]
        color = CLASS_COLORS.get(cls_name, (0, 180, 216))

        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        label = f"{cls_name} {conf:.2f}"
        text_y = max(0, y1 - 14)
        draw.rectangle([x1, text_y, x1 + len(label) * 6 + 6, text_y + 14], fill=color)
        draw.text((x1 + 3, text_y + 1), label, fill=(255, 255, 255))

    return annotated


class DefectDetector:
    """Wrapper around Ultralytics YOLO with instant precomputed lookup for preset images."""

    def __init__(
        self,
        weights_path: Optional[Union[str, Path]] = None,
        config_path: Path = Path("configs/config.yaml"),
        precomputed_cache_path: Path = Path("assets/precomputed/presets_cache.json"),
    ) -> None:
        self.cfg = load_config(config_path)
        model_cfg = self.cfg.get("model", {})

        primary_path = Path(weights_path or model_cfg.get("weights_path", "weights/best.pt"))
        fallback_name = str(model_cfg.get("fallback_weights", "yolo11n.pt"))

        self.default_conf: float = float(model_cfg.get("default_conf_threshold", 0.25))
        self.iou_threshold: float = float(model_cfg.get("iou_threshold", 0.45))
        self.image_size: int = int(model_cfg.get("image_size", 320))

        self.using_custom_weights: bool = primary_path.exists()
        self.resolved_weights: str = str(primary_path) if self.using_custom_weights else fallback_name
        self._model: Any = None
        self._raw_cache: Dict[str, List[Dict[str, Any]]] = {}
        self._preset_entries: List[Tuple[str, np.ndarray, List[Dict[str, Any]]]] = []
        self._preset_by_name: Dict[str, List[Dict[str, Any]]] = {}
        self._load_precomputed_cache(precomputed_cache_path)

    def _load_precomputed_cache(self, cache_path: Path) -> None:
        """Load disk-persisted detections for the 10 built-in example images."""
        if not cache_path.exists():
            return
        try:
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            for fname, entry in data.items():
                fp = np.asarray(entry.get("fingerprint", []), dtype=np.float32)
                boxes = entry.get("boxes", [])
                self._preset_by_name[fname] = boxes
                if fp.size == 256:
                    self._preset_entries.append((fname, fp, boxes))
        except Exception as exc:
            logger.warning("Failed to load precomputed cache (%s)", exc)

    def _match_preset_boxes(self, pil_img: Image.Image) -> Optional[List[Dict[str, Any]]]:
        """Match a PIL image against the 10 preset fingerprints (handles JPEG recompression)."""
        if not self._preset_entries:
            return None
        fp = compute_image_fingerprint(pil_img)
        best_diff = 999.0
        best_boxes: Optional[List[Dict[str, Any]]] = None
        for _, preset_fp, boxes in self._preset_entries:
            diff = float(np.mean(np.abs(fp - preset_fp)))
            if diff < best_diff:
                best_diff = diff
                best_boxes = boxes
        # Mean absolute pixel difference < 6.0 out of 255 guarantees it is the same preset image
        if best_diff < 6.0 and best_boxes is not None:
            return best_boxes
        return None

    def _get_model(self) -> Any:
        """Lazy-load the Ultralytics YOLO model only when a custom non-preset image is uploaded."""
        if self._model is None:
            import torch
            from ultralytics import YOLO

            torch.set_num_threads(1)

            if not self.using_custom_weights:
                logger.warning(
                    "Trained weights not found at 'weights/best.pt'. "
                    "Falling back to base '%s' weights for demo/testing.",
                    self.resolved_weights,
                )
            self._model = YOLO(self.resolved_weights)
            gc.collect()
        return self._model

    def predict(
        self,
        image: Union[Image.Image, np.ndarray, str, Path],
        conf_threshold: Optional[float] = None,
    ) -> Tuple[Image.Image, List[Dict[str, Any]], Dict[str, Any]]:
        """Run detection on an image and return (annotated_image, detections_list, structured_json)."""
        preset_raw_boxes: Optional[List[Dict[str, Any]]] = None

        if isinstance(image, (str, Path)):
            img_path = Path(image)
            pil_img = Image.open(img_path).convert("RGB")
            if img_path.name in self._preset_by_name:
                preset_raw_boxes = self._preset_by_name[img_path.name]
        elif isinstance(image, np.ndarray):
            pil_img = Image.fromarray(image).convert("RGB")
        elif isinstance(image, Image.Image):
            pil_img = image.convert("RGB")
        else:
            raise TypeError(f"Unsupported image input type: {type(image)}")

        conf = float(conf_threshold if conf_threshold is not None else self.default_conf)
        width, height = pil_img.size

        if preset_raw_boxes is None:
            preset_raw_boxes = self._match_preset_boxes(pil_img)

        if preset_raw_boxes is None:
            img_hash = hashlib.sha1(pil_img.tobytes()).hexdigest()
            if img_hash not in self._raw_cache:
                model = self._get_model()
                results = model.predict(
                    source=pil_img,
                    conf=0.05,
                    iou=self.iou_threshold,
                    imgsz=self.image_size,
                    verbose=False,
                )

                raw_detections: List[Dict[str, Any]] = []
                if results:
                    res = results[0]
                    names_map: Dict[int, str] = getattr(res, "names", {}) or {
                        i: c for i, c in enumerate(CLASSES)
                    }
                    boxes = getattr(res, "boxes", None)
                    if boxes is not None and len(boxes) > 0:
                        xyxy_arr = boxes.xyxy.cpu().numpy()
                        conf_arr = boxes.conf.cpu().numpy()
                        cls_arr = boxes.cls.cpu().numpy().astype(int)

                        for xyxy, score, cls_id in zip(xyxy_arr, conf_arr, cls_arr):
                            x1, y1, x2, y2 = [round(float(v), 1) for v in xyxy.tolist()]
                            xc_norm = ((x1 + x2) / 2.0) / max(1, width)
                            yc_norm = ((y1 + y2) / 2.0) / max(1, height)
                            raw_cls_name = names_map.get(int(cls_id), f"class_{cls_id}")

                            raw_detections.append(
                                {
                                    "class_id": int(cls_id),
                                    "class_name": str(raw_cls_name),
                                    "confidence": round(float(score), 4),
                                    "bbox_xyxy": [x1, y1, x2, y2],
                                    "region": describe_region(xc_norm, yc_norm),
                                }
                            )
                raw_detections.sort(key=lambda d: d["confidence"], reverse=True)
                if len(self._raw_cache) >= 64:
                    self._raw_cache.clear()
                self._raw_cache[img_hash] = raw_detections
                gc.collect()
            preset_raw_boxes = self._raw_cache[img_hash]

        detections: List[Dict[str, Any]] = []
        for det in preset_raw_boxes:
            if float(det["confidence"]) >= conf:
                det_copy = dict(det)
                det_copy["id"] = len(detections) + 1
                detections.append(det_copy)

        annotated_img = draw_detections_pil(pil_img, detections)
        payload = build_structured_payload(
            detections=detections,
            conf_threshold=conf,
            image_size=(width, height),
            using_custom_weights=self.using_custom_weights,
        )
        return annotated_img, detections, payload
