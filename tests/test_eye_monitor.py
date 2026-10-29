"""
Unit Tests for modules/eye_monitor.py
======================================
Tests EAR calculation, monotonic timer logic, blink handling,
drowsiness triggering, and face-missing edge cases.
"""

from collections import namedtuple
import pytest
from modules.eye_monitor import EyeMonitor, calculate_single_ear, euclidean_distance_2d


# Mock class to simulate MediaPipe landmark objects
MockLandmark = namedtuple("MockLandmark", ["x", "y", "z"])


class MockNormalizedLandmarkList:
    def __init__(self, landmark_dict, total_count=478):
        self.landmark = [MockLandmark(x=0.5, y=0.5, z=0.0) for _ in range(total_count)]
        for idx, (x, y) in landmark_dict.items():
            self.landmark[idx] = MockLandmark(x=x, y=y, z=0.0)


def create_synthetic_eye_landmarks(is_closed=False):
    """
    Creates normalized coordinates for left and right eyes.
    Calibrated for standard 1280x720 video frames:
    - Open eye: EAR ~ 0.36
    - Closed eye: EAR ~ 0.06
    """
    # In 1280x720:
    # Horizontal width = (0.40 - 0.30) * 1280 = 128 pixels.
    # Vertical offset:
    # Open: 0.032 * 720 = 23.04 px -> total height = 46.08 px -> EAR = (46.08 + 46.08)/(2*128) = 0.36
    # Closed: 0.005 * 720 = 3.6 px -> total height = 7.2 px -> EAR = (7.2 + 7.2)/(2*128) = 0.056
    v_offset = 0.005 if is_closed else 0.032

    # Left eye mapping: p1=33, p4=133, p2=160, p6=144, p3=158, p5=153
    left_coords = {
        33: (0.30, 0.40),                  # p1 (outer)
        133: (0.40, 0.40),                 # p4 (inner)
        160: (0.33, 0.40 - v_offset),      # p2 (top 1)
        144: (0.33, 0.40 + v_offset),      # p6 (bottom 1)
        158: (0.37, 0.40 - v_offset),      # p3 (top 2)
        153: (0.37, 0.40 + v_offset),      # p5 (bottom 2)
    }

    # Right eye mapping: p1=362, p4=263, p2=385, p6=380, p3=387, p5=373
    right_coords = {
        362: (0.60, 0.40),                 # p1 (outer)
        263: (0.70, 0.40),                 # p4 (inner)
        385: (0.63, 0.40 - v_offset),      # p2 (top 1)
        380: (0.63, 0.40 + v_offset),      # p6 (bottom 1)
        387: (0.67, 0.40 - v_offset),      # p3 (top 2)
        373: (0.67, 0.40 + v_offset),      # p5 (bottom 2)
    }

    combined = {**left_coords, **right_coords}
    return MockNormalizedLandmarkList(combined)


def test_calculate_single_ear_geometry():
    """Verify standard EAR geometric formula calculation."""
    # Horizontal width = 100
    # Vertical heights = 30 and 30
    # Expected EAR = (30 + 30) / (2 * 100) = 0.30
    points = {
        "p1": (100.0, 200.0),
        "p4": (200.0, 200.0),
        "p2": (130.0, 185.0),
        "p6": (130.0, 215.0),
        "p3": (170.0, 185.0),
        "p5": (170.0, 215.0),
    }
    ear = calculate_single_ear(points)
    assert pytest.approx(ear, 0.01) == 0.30


def test_calculate_single_ear_zero_division_guard():
    """Verify division by zero is safely handled when points collapse."""
    points = {
        "p1": (100.0, 200.0),
        "p4": (100.0, 200.0),  # Same point -> horizontal distance = 0
        "p2": (100.0, 190.0),
        "p6": (100.0, 210.0),
        "p3": (100.0, 190.0),
        "p5": (100.0, 210.0),
    }
    ear = calculate_single_ear(points)
    assert ear == 0.0


def test_eye_monitor_open_eyes():
    """Verify open eye state produces expected EAR and no drowsiness."""
    monitor = EyeMonitor(ear_threshold=0.22, drowsiness_duration_sec=2.0)
    open_landmarks = create_synthetic_eye_landmarks(is_closed=False)

    res = monitor.process(open_landmarks, frame_width=1280, frame_height=720, current_time=100.0)

    assert res["is_face_present"] is True
    assert res["avg_ear"] > 0.22
    assert res["are_eyes_closed"] is False
    assert res["is_drowsy"] is False
    assert res["closed_duration"] == 0.0


def test_eye_monitor_brief_blink_does_not_trigger_drowsy():
    """A natural human blink lasts 0.1-0.4s. Verify it does not trigger drowsiness."""
    monitor = EyeMonitor(ear_threshold=0.22, drowsiness_duration_sec=2.0)
    closed_landmarks = create_synthetic_eye_landmarks(is_closed=True)
    open_landmarks = create_synthetic_eye_landmarks(is_closed=False)

    # Blink starts at t=10.0
    res1 = monitor.process(closed_landmarks, 1280, 720, current_time=10.0)
    assert res1["are_eyes_closed"] is True
    assert res1["is_drowsy"] is False

    # Blink continues at t=10.3s (0.3s elapsed)
    res2 = monitor.process(closed_landmarks, 1280, 720, current_time=10.3)
    assert res2["are_eyes_closed"] is True
    assert res2["is_drowsy"] is False
    assert pytest.approx(res2["closed_duration"], 0.05) == 0.3

    # Eyes reopen at t=10.4s -> timer must reset
    res3 = monitor.process(open_landmarks, 1280, 720, current_time=10.4)
    assert res3["are_eyes_closed"] is False
    assert res3["is_drowsy"] is False
    assert res3["closed_duration"] == 0.0


def test_eye_monitor_prolonged_closure_triggers_drowsy():
    """Verify that eyes closed beyond 2.0s triggers drowsiness alert."""
    monitor = EyeMonitor(ear_threshold=0.22, drowsiness_duration_sec=2.0)
    closed_landmarks = create_synthetic_eye_landmarks(is_closed=True)

    # Eyes close at t=10.0
    monitor.process(closed_landmarks, 1280, 720, current_time=10.0)

    # At t=11.5s (1.5s closed -> not drowsy yet)
    mid_res = monitor.process(closed_landmarks, 1280, 720, current_time=11.5)
    assert mid_res["are_eyes_closed"] is True
    assert mid_res["is_drowsy"] is False

    # At t=12.1s (2.1s closed -> DROWSY!)
    drowsy_res = monitor.process(closed_landmarks, 1280, 720, current_time=12.1)
    assert drowsy_res["are_eyes_closed"] is True
    assert drowsy_res["is_drowsy"] is True
    assert "Wake Up & Study!" in drowsy_res["status_text"]


def test_eye_monitor_resets_when_face_is_lost():
    """When the user leaves the camera, eye closure timer must reset."""
    monitor = EyeMonitor(ear_threshold=0.22, drowsiness_duration_sec=2.0)
    closed_landmarks = create_synthetic_eye_landmarks(is_closed=True)

    # Eyes close at t=10.0
    monitor.process(closed_landmarks, 1280, 720, current_time=10.0)

    # Face leaves at t=11.0 (landmarks = None)
    res_no_face = monitor.process(None, 1280, 720, current_time=11.0)
    assert res_no_face["is_face_present"] is False
    assert res_no_face["is_drowsy"] is False
    assert res_no_face["closed_duration"] == 0.0


def test_eye_monitor_handles_invalid_landmarks_safely():
    """Verify safe fallback when landmark object is empty or corrupted."""
    monitor = EyeMonitor()

    # Empty object
    res = monitor.process(object(), 1280, 720)
    assert res["is_face_present"] is False
    assert res["is_drowsy"] is False
