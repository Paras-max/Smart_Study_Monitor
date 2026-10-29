"""
Unit Tests for modules/phone_detector.py
=========================================
Tests YOLO model loading, inference error handling,
frame interval caching, and phone class filtering.
"""

from unittest.mock import MagicMock
import numpy as np
import pytest
import torch
from modules.phone_detector import PhoneDetector


def test_phone_detector_initialization_with_real_model():
    """Verify loading real yolov8n.pt weights on CPU."""
    detector = PhoneDetector(model_path="yolov8n.pt", confidence_threshold=0.45)
    assert detector.is_loaded is True
    assert detector.model is not None
    assert detector.load_error is None


def test_phone_detector_handles_missing_model_safely():
    """Verify that a nonexistent model path does not crash the application."""
    detector = PhoneDetector(model_path="nonexistent_model_weights.pt")
    assert detector.is_loaded is False
    assert detector.model is None
    assert detector.load_error is not None

    # Detecting on missing model should return offline status safely
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    res = detector.detect(dummy_frame)
    assert res["phone_detected"] is False
    assert res["inference_success"] is False
    assert res["status_message"] == "Phone Detector Offline"


def test_phone_detector_inference_on_blank_frame():
    """Verify running inference on an empty blank frame produces 'No Phone Detected'."""
    detector = PhoneDetector(model_path="yolov8n.pt", confidence_threshold=0.45)
    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    res = detector.detect(blank_frame, force_inference=True)
    assert res["inference_success"] is True
    assert res["phone_detected"] is False
    assert res["status_message"] == "No Phone Detected"
    assert len(res["boxes"]) == 0


def test_phone_detector_frame_interval_skipping():
    """Verify that inference is skipped on non-interval frames to preserve CPU FPS."""
    detector = PhoneDetector(model_path="yolov8n.pt", inference_interval=3)
    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Frame 1: skipped (1 % 3 != 0) -> returns initial cached result
    res1 = detector.detect(blank_frame, force_inference=False)
    assert res1["phone_detected"] is False

    # Frame 2: skipped
    res2 = detector.detect(blank_frame, force_inference=False)
    assert res2["phone_detected"] is False

    # Frame 3: executed (3 % 3 == 0)
    res3 = detector.detect(blank_frame, force_inference=False)
    assert res3["inference_success"] is True


def test_phone_detector_parses_target_class_phone():
    """Test parsing YOLO boxes when class 67 (cell phone) is detected."""
    detector = PhoneDetector(model_path="yolov8n.pt", confidence_threshold=0.40, target_class_id=67)

    # Mock YOLO model inference output
    mock_boxes = MagicMock()
    mock_boxes.__len__.return_value = 2

    # Box 0: Class 67 (phone) conf 0.85, coords [50, 60, 200, 300]
    # Box 1: Class 0 (person) conf 0.90, coords [0, 0, 400, 400]
    mock_boxes.cls = [torch.tensor(67), torch.tensor(0)]
    mock_boxes.conf = [torch.tensor(0.85), torch.tensor(0.90)]
    mock_boxes.xyxy = [
        torch.tensor([50.0, 60.0, 200.0, 300.0]),
        torch.tensor([0.0, 0.0, 400.0, 400.0]),
    ]

    mock_result = MagicMock()
    mock_result.boxes = mock_boxes

    detector.model = MagicMock()
    detector.model.predict.return_value = [mock_result]
    detector.is_loaded = True

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    res = detector.detect(dummy_frame, force_inference=True)

    assert res["phone_detected"] is True
    assert res["status_message"] == "Put the Phone Away!"
    assert len(res["boxes"]) == 1
    assert res["boxes"][0]["conf"] == 0.85
    assert res["boxes"][0]["x1"] == 50
    assert res["boxes"][0]["y1"] == 60
