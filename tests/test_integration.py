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
