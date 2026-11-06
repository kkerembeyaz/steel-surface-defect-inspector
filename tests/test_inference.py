"""Unit tests for YOLO inference helper functions and structured JSON formatting."""

import unittest
from PIL import Image

from src.inference import (
    build_structured_payload,
    describe_region,
    draw_detections_pil,
)


def test_describe_region() -> None:
    """Verify spatial region mapping from normalized coordinates."""
    assert describe_region(0.1, 0.1) == "top-left"
    assert describe_region(0.5, 0.5) == "center"
    assert describe_region(0.85, 0.85) == "bottom-right"
    assert describe_region(0.2, 0.5) == "middle-left"


def test_build_structured_payload_aggregates_correctly() -> None:
    """Ensure detection list is accurately summarized by class, count, and confidence."""
    detections = [
        {
            "id": 1,
            "class_id": 5,
            "class_name": "scratches",
            "confidence": 0.88,
            "bbox_xyxy": [10.0, 20.0, 50.0, 180.0],
            "region": "middle-left",
        },
        {
            "id": 2,
            "class_id": 5,
            "class_name": "scratches",
            "confidence": 0.62,
            "bbox_xyxy": [120.0, 30.0, 160.0, 170.0],
            "region": "middle-right",
        },
        {
            "id": 3,
            "class_id": 1,
            "class_name": "inclusion",
            "confidence": 0.75,
            "bbox_xyxy": [80.0, 80.0, 110.0, 110.0],
            "region": "center",
        },
    ]

    payload = build_structured_payload(
        detections=detections,
        conf_threshold=0.25,
        image_size=(200, 200),
        using_custom_weights=True,
    )

    assert payload["total_detections"] == 3
    assert payload["confidence_threshold"] == 0.25
    assert "scratches" in payload["summary_by_class"]
    assert "inclusion" in payload["summary_by_class"]

    scratches_summary = payload["summary_by_class"]["scratches"]
    assert scratches_summary["count"] == 2
    assert scratches_summary["mean_confidence"] == 0.75
    assert scratches_summary["max_confidence"] == 0.88
    assert scratches_summary["min_confidence"] == 0.62


def test_draw_detections_pil_returns_valid_rgb_image() -> None:
    """Verify bounding box drawing preserves image dimensions and RGB mode."""
    base_img = Image.new("L", (200, 200), color=128)
    detections = [
        {
            "id": 1,
            "class_id": 0,
            "class_name": "crazing",
            "confidence": 0.91,
            "bbox_xyxy": [20.0, 20.0, 100.0, 100.0],
            "region": "top-left",
        }
    ]
    out_img = draw_detections_pil(base_img, detections)
    assert out_img.size == (200, 200)
    assert out_img.mode == "RGB"


class TestInferenceUnit(unittest.TestCase):
    """Standard unittest runner wrapper for environments without pytest installed."""

    def test_all(self) -> None:
        test_describe_region()
        test_build_structured_payload_aggregates_correctly()
        test_draw_detections_pil_returns_valid_rgb_image()


if __name__ == "__main__":
    unittest.main()
