"""Unit tests for the LLM prompt builder and rule-based fallback report generator."""

import os
import unittest

from src.report import (
    SYSTEM_PROMPT,
    build_user_prompt,
    generate_quality_report,
    generate_rule_based_report,
)


def test_system_prompt_contains_anti_hallucination_rules() -> None:
    """Ensure the system prompt explicitly forbids inventing non-existent defects."""
    assert "ASLA uydurma" in SYSTEM_PROMPT
    assert "Olası Kök Neden" in SYSTEM_PROMPT
    assert "Önerilen Aksiyon" in SYSTEM_PROMPT
    assert "Güven Notu" in SYSTEM_PROMPT


def test_rule_based_report_zero_detections() -> None:
    """Verify rule-based report handles zero detections cleanly without inventing classes."""
    empty_payload = {
        "total_detections": 0,
        "confidence_threshold": 0.30,
        "using_trained_neu_weights": True,
        "summary_by_class": {},
        "detections": [],
    }
    report = generate_rule_based_report(empty_payload)
    assert "herhangi bir kusur tespit edilmemiştir" in report
    assert "0.30" in report
    for cls_name in ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]:
        assert cls_name not in report


def test_rule_based_report_with_detections_and_fallback_without_hf_token() -> None:
    """Ensure generate_quality_report falls back to rule-based template when HF_TOKEN is unset."""
    old_token = os.environ.pop("HF_TOKEN", None)
    try:
        payload = {
            "total_detections": 2,
            "confidence_threshold": 0.25,
            "using_trained_neu_weights": True,
            "summary_by_class": {
                "rolled-in_scale": {
                    "count": 2,
                    "mean_confidence": 0.82,
                    "max_confidence": 0.89,
                    "min_confidence": 0.75,
                    "regions": ["center", "top-left"],
                }
            },
            "detections": [
                {
                    "id": 1,
                    "class_id": 4,
                    "class_name": "rolled-in_scale",
                    "confidence": 0.89,
                    "bbox_xyxy": [20.0, 20.0, 60.0, 60.0],
                    "region": "top-left",
                }
            ],
        }

        report_md, mode = generate_quality_report(payload)
        assert mode == "rule_based"
        assert "rolled-in_scale" in report_md
        assert "2 adet" in report_md
        assert "Özet" in report_md
        assert "Olası Kök Neden" in report_md
        assert "Önerilen Aksiyon" in report_md
        assert "Güven Notu" in report_md
        assert "scratches" not in report_md
    finally:
        if old_token is not None:
            os.environ["HF_TOKEN"] = old_token


def test_build_user_prompt_embeds_structured_json() -> None:
    """Verify user prompt serializes the YOLO JSON payload cleanly."""
    payload = {
        "total_detections": 1,
        "confidence_threshold": 0.40,
        "summary_by_class": {"patches": {"count": 1, "mean_confidence": 0.77}},
        "detections": [],
    }
    prompt = build_user_prompt(payload)
    assert '"patches"' in prompt
    assert '"total_detections": 1' in prompt


class TestReportUnit(unittest.TestCase):
    """Standard unittest runner wrapper for environments without pytest installed."""

    def test_all(self) -> None:
        test_system_prompt_contains_anti_hallucination_rules()
        test_rule_based_report_zero_detections()
        test_rule_based_report_with_detections_and_fallback_without_hf_token()
        test_build_user_prompt_embeds_structured_json()


if __name__ == "__main__":
    unittest.main()
