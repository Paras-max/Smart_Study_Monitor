"""
Integration Tests for Smart Study Monitor Phase 2
=================================================
Validates the status priority hierarchy and multi-module state coordination.
"""

import pytest


def determine_status_priority(is_face_detected: bool, phone_detected: bool, is_drowsy: bool):
    """Encapsulates the status hierarchy logic used in main.py."""
    if not is_face_detected:
        return "WARNING: Face Not Detected!"
    elif phone_detected:
        return "WARNING: Put the Phone Away!"
    elif is_drowsy:
        return "WARNING: Wake Up & Study!"
    else:
        return "STATUS: Focused & Studying"


def test_priority_face_missing_overrides_everything():
    """Even if phone is present and eyes are closed, face missing is Priority 1."""
    status = determine_status_priority(is_face_detected=False, phone_detected=True, is_drowsy=True)
    assert status == "WARNING: Face Not Detected!"


def test_priority_phone_overrides_drowsy():
    """Phone distraction is Priority 2, taking precedence over drowsiness."""
    status = determine_status_priority(is_face_detected=True, phone_detected=True, is_drowsy=True)
    assert status == "WARNING: Put the Phone Away!"


def test_priority_drowsy_overrides_normal():
    """Drowsiness is Priority 3, taking precedence over normal study status."""
    status = determine_status_priority(is_face_detected=True, phone_detected=False, is_drowsy=True)
    assert status == "WARNING: Wake Up & Study!"


def test_priority_normal_study_status():
    """When face is detected, no phone is found, and eyes are open, normal status shows."""
    status = determine_status_priority(is_face_detected=True, phone_detected=False, is_drowsy=False)
    assert status == "STATUS: Focused & Studying"


def test_multi_module_pipeline_frame_handling():
    """Validates that FaceMesh, EyeMonitor, and PhoneDetector coordinate cleanly on a frame."""
    import numpy as np
    import mediapipe as mp
    from modules.eye_monitor import EyeMonitor
    from modules.phone_detector import PhoneDetector

    # Initialize modules
    mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)
    eye_mon = EyeMonitor()
    phone_det = PhoneDetector(model_path="yolov8n.pt", confidence_threshold=0.45)

    try:
        # Create a test frame (480x640x3)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame_rgb = frame[:, :, ::-1]

        # 1. FaceMesh process
        mesh_out = mesh.process(frame_rgb)
        face_lms = mesh_out.multi_face_landmarks[0] if mesh_out.multi_face_landmarks else None

        # 2. EyeMonitor process with no face
        eye_out = eye_mon.process(face_lms, 640, 480)
        assert eye_out["is_face_present"] is False
        assert eye_out["is_drowsy"] is False

        # 3. PhoneDetector process
        phone_out = phone_det.detect(frame, force_inference=True)
        assert phone_out["inference_success"] is True
        assert phone_out["phone_detected"] is False

        # 4. Status determination
        status = determine_status_priority(
            is_face_detected=eye_out["is_face_present"],
            phone_detected=phone_out["phone_detected"],
            is_drowsy=eye_out["is_drowsy"],
        )
        assert status == "WARNING: Face Not Detected!"

    finally:
        mesh.close()

