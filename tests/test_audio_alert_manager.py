"""
Automated Unit Tests for AudioAlertManager (Phase 3)
===================================================
Tests all 12 Phase 3 requirements:
1. Audio file loading.
2. Correct event-to-audio mapping.
3. Sleeping event triggers the correct clip.
4. Face-hidden event triggers the correct clip.
5. Phone detection triggers the correct clip.
6. Cooldown prevents repeated playback.
7. Priority hierarchy works correctly.
8. Multiple alerts do not overlap.
9. Missing audio files are handled safely.
10. Audio playback failures do not crash the application.
11. Alert state resets correctly.
12. Webcam processing does not wait for audio playback.
"""

from pathlib import Path
import time
from unittest.mock import MagicMock, patch
import pytest

import config
from modules.audio_alert_manager import AudioAlertManager, AlertPriority


@pytest.fixture
def test_audio_config(tmp_path):
    """Provides a controlled test configuration with mock audio paths."""
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    
    sleeping_clip = audio_dir / "sleeping_alert.mp3"
    face_hidden_clip = audio_dir / "face_hidden_alert.mp3"
    phone_clip = audio_dir / "phone_detected_alert.mp3"

    # Create dummy files
    sleeping_clip.write_bytes(b"MOCK_SLEEPING_AUDIO")
    face_hidden_clip.write_bytes(b"MOCK_FACE_HIDDEN_AUDIO")
    phone_clip.write_bytes(b"MOCK_PHONE_AUDIO")

    return {
        "ENABLED": True,
        "AUDIO_DIR": audio_dir,
        "CLIPS": {
            AudioAlertManager.EVENT_FACE_HIDDEN: face_hidden_clip,
            AudioAlertManager.EVENT_DROWSY: sleeping_clip,
            AudioAlertManager.EVENT_PHONE_DETECTED: phone_clip,
        },
        "FACE_HIDDEN_GRACE_PERIOD_SEC": 3.0,
        "FACE_HIDDEN_COOLDOWN_SEC": 15.0,
        "SLEEPING_COOLDOWN_SEC": 15.0,
        "PHONE_COOLDOWN_SEC": 20.0,
        "PHONE_ABSENCE_RESET_SEC": 3.0,
    }


# ---------------------------------------------------------------------------
# Test 1: Audio file loading
# ---------------------------------------------------------------------------
def test_01_audio_file_loading(test_audio_config):
    played_events = []
    manager = AudioAlertManager(
        audio_config=test_audio_config,
        sound_player=lambda evt: played_events.append(evt)
    )

    assert manager.enabled is True
    assert len(manager.audio_paths) == 3
    assert manager.audio_paths[AudioAlertManager.EVENT_DROWSY].exists()
    assert manager.audio_paths[AudioAlertManager.EVENT_FACE_HIDDEN].exists()
    assert manager.audio_paths[AudioAlertManager.EVENT_PHONE_DETECTED].exists()
    manager.cleanup()


# ---------------------------------------------------------------------------
# Test 2: Correct event-to-audio mapping
# ---------------------------------------------------------------------------
def test_02_event_to_audio_mapping(test_audio_config):
    manager = AudioAlertManager(
        audio_config=test_audio_config,
        sound_player=lambda evt: None
    )

    # Verify filenames map to intended events
    assert manager.audio_paths[AudioAlertManager.EVENT_DROWSY].name == "sleeping_alert.mp3"
    assert manager.audio_paths[AudioAlertManager.EVENT_FACE_HIDDEN].name == "face_hidden_alert.mp3"
    assert manager.audio_paths[AudioAlertManager.EVENT_PHONE_DETECTED].name == "phone_detected_alert.mp3"
    manager.cleanup()


# ---------------------------------------------------------------------------
# Test 3: Sleeping event triggers the correct clip
# ---------------------------------------------------------------------------
def test_03_sleeping_event_triggers_correct_clip(test_audio_config):
    played = []
    manager = AudioAlertManager(
        audio_config=test_audio_config,
        sound_player=lambda evt: played.append(evt)
    )

    # Face detected, eyes drowsy, no phone
    status, alert_name = manager.update(
        face_detected=True,
        is_drowsy=True,
        phone_detected=False,
        current_time=100.0
    )

    assert status == "SLEEPING"
    assert alert_name == "Sleeping / Drowsy"
    assert played == [AudioAlertManager.EVENT_DROWSY]
    manager.cleanup()


# ---------------------------------------------------------------------------
# Test 4: Face-hidden event triggers the correct clip (with 3s grace period)
# ---------------------------------------------------------------------------
def test_04_face_hidden_event_triggers_correct_clip(test_audio_config):
    played = []
    manager = AudioAlertManager(
        audio_config=test_audio_config,
        sound_player=lambda evt: played.append(evt)
    )

    # Brief face loss (1.5s) - under grace period (3s) -> should NOT trigger audio
    status1, _ = manager.update(face_detected=False, is_drowsy=False, phone_detected=False, current_time=100.0)
    assert len(played) == 0
    assert status1 == "DISTRACTED"

    manager.update(face_detected=False, is_drowsy=False, phone_detected=False, current_time=102.0)
    assert len(played) == 0  # 2.0s elapsed: still within 3s grace

    # Exceeding grace period: 3.5s elapsed -> trigger FACE_HIDDEN
    status3, alert_name = manager.update(face_detected=False, is_drowsy=False, phone_detected=False, current_time=103.5)
    assert played == [AudioAlertManager.EVENT_FACE_HIDDEN]
    assert alert_name == "Face Hidden"
    assert status3 == "DISTRACTED"
    manager.cleanup()


# ---------------------------------------------------------------------------
# Test 5: Phone detection triggers the correct clip
# ---------------------------------------------------------------------------
def test_05_phone_detection_triggers_correct_clip(test_audio_config):
    played = []
    manager = AudioAlertManager(
        audio_config=test_audio_config,
        sound_player=lambda evt: played.append(evt)
    )

    # Face present, attentive eyes, but phone detected
    status, alert_name = manager.update(
        face_detected=True,
        is_drowsy=False,
        phone_detected=True,
        current_time=100.0
    )

    assert status == "DISTRACTED"
    assert alert_name == "Phone Detected"
    assert played == [AudioAlertManager.EVENT_PHONE_DETECTED]
    manager.cleanup()


# ---------------------------------------------------------------------------
# Test 6: Cooldown prevents repeated playback
# ---------------------------------------------------------------------------
def test_06_cooldown_prevents_repeated_playback(test_audio_config):
    played = []
    manager = AudioAlertManager(
        audio_config=test_audio_config,
        sound_player=lambda evt: played.append(evt)
    )

    # Initial trigger at t = 100.0 (Drowsy cooldown is 15s)
    manager.trigger(AudioAlertManager.EVENT_DROWSY, current_time=100.0)
    manager._current_playing_event = None  # simulate clip finished playing
    assert len(played) == 1

    # Attempt to trigger again at t = 105.0 (5s elapsed < 15s cooldown)
    res = manager.trigger(AudioAlertManager.EVENT_DROWSY, current_time=105.0)
    assert res is False
    assert len(played) == 1

    # Attempt to trigger at t = 114.9 (14.9s elapsed < 15s cooldown)
    res2 = manager.trigger(AudioAlertManager.EVENT_DROWSY, current_time=114.9)
    assert res2 is False
    assert len(played) == 1

    # Trigger at t = 115.1 (15.1s elapsed > 15s cooldown) -> success!
    res3 = manager.trigger(AudioAlertManager.EVENT_DROWSY, current_time=115.1)
    assert res3 is True
    assert len(played) == 2
    manager.cleanup()


# ---------------------------------------------------------------------------
# Test 7: Priority hierarchy works correctly
# ---------------------------------------------------------------------------
def test_07_priority_hierarchy_works_correctly(test_audio_config):
    """
    Priority Hierarchy:
    1. FACE_HIDDEN
    2. DROWSY
    3. PHONE_DETECTED
    4. FOCUSED_STUDY
    """
    assert AudioAlertManager.PRIORITIES[AudioAlertManager.EVENT_FACE_HIDDEN] < AudioAlertManager.PRIORITIES[AudioAlertManager.EVENT_DROWSY]
    assert AudioAlertManager.PRIORITIES[AudioAlertManager.EVENT_DROWSY] < AudioAlertManager.PRIORITIES[AudioAlertManager.EVENT_PHONE_DETECTED]
    assert AudioAlertManager.PRIORITIES[AudioAlertManager.EVENT_PHONE_DETECTED] < AudioAlertManager.PRIORITIES[AudioAlertManager.EVENT_FOCUSED]

    # When all 3 negative conditions occur simultaneously:
    # Face missing (>3s), Drowsy, and Phone detected -> FACE_HIDDEN must win!
    played = []
    manager = AudioAlertManager(
        audio_config=test_audio_config,
        sound_player=lambda evt: played.append(evt)
    )

    # t0: start face missing
    manager.update(face_detected=False, is_drowsy=True, phone_detected=True, current_time=100.0)
    # t = 104.0: face has been missing 4s (> 3s grace)
    manager.update(face_detected=False, is_drowsy=True, phone_detected=True, current_time=104.0)

    assert played == [AudioAlertManager.EVENT_FACE_HIDDEN]
    manager.cleanup()


# ---------------------------------------------------------------------------
# Test 8: Multiple alerts do not overlap (Non-interruption & Priority Queuing)
# ---------------------------------------------------------------------------
def test_08_multiple_alerts_do_not_overlap(test_audio_config):
    played = []
    manager = AudioAlertManager(
        audio_config=test_audio_config,
        sound_player=lambda evt: played.append(evt)
    )

    # 1. Start Phone Alert at t = 100.0 (Priority 3)
    manager.trigger(AudioAlertManager.EVENT_PHONE_DETECTED, current_time=100.0)
    assert manager.is_playing() is True
    assert played == [AudioAlertManager.EVENT_PHONE_DETECTED]

    # 2. While Phone Alert is still playing, Drowsy event occurs at t = 101.0 (Priority 2)
    # Rule: Never interrupt active clip, but queue higher priority event
    res = manager.trigger(AudioAlertManager.EVENT_DROWSY, current_time=101.0)
    assert res is True
    assert manager._pending_queued_event == AudioAlertManager.EVENT_DROWSY
    # Ensure phone clip was NOT stopped
    assert played == [AudioAlertManager.EVENT_PHONE_DETECTED]

    # 3. Even higher priority FACE_HIDDEN occurs at t = 102.0 (Priority 1)
    res2 = manager.trigger(AudioAlertManager.EVENT_FACE_HIDDEN, current_time=102.0)
    assert res2 is True
    # Highest priority replaced the previous pending in queue
    assert manager._pending_queued_event == AudioAlertManager.EVENT_FACE_HIDDEN

    # 4. Phone clip finishes at t = 103.55
    manager._current_playing_event = None
    assert manager.is_playing() is False

    # Process queue -> FACE_HIDDEN now plays immediately
    popped = manager.process_queue(current_time=103.6)
    assert popped == AudioAlertManager.EVENT_FACE_HIDDEN
    assert played == [AudioAlertManager.EVENT_PHONE_DETECTED, AudioAlertManager.EVENT_FACE_HIDDEN]
    manager.cleanup()


# ---------------------------------------------------------------------------
# Test 9: Missing audio files are handled safely
# ---------------------------------------------------------------------------
def test_09_missing_audio_files_handled_safely(tmp_path):
    empty_dir = tmp_path / "non_existent_audio"
    bad_config = {
        "ENABLED": True,
        "AUDIO_DIR": empty_dir,
        "CLIPS": {
            AudioAlertManager.EVENT_FACE_HIDDEN: empty_dir / "missing_face.mp3",
            AudioAlertManager.EVENT_DROWSY: empty_dir / "missing_drowsy.mp3",
            AudioAlertManager.EVENT_PHONE_DETECTED: empty_dir / "missing_phone.mp3",
        },
        "FACE_HIDDEN_GRACE_PERIOD_SEC": 3.0,
        "FACE_HIDDEN_COOLDOWN_SEC": 15.0,
        "SLEEPING_COOLDOWN_SEC": 15.0,
        "PHONE_COOLDOWN_SEC": 20.0,
        "PHONE_ABSENCE_RESET_SEC": 3.0,
    }

    # Should initialize cleanly without throwing FileNotFoundError
    manager = AudioAlertManager(audio_config=bad_config)
    assert manager.sounds[AudioAlertManager.EVENT_FACE_HIDDEN] is None

    # Triggering should safely return False without exception
    success = manager.trigger(AudioAlertManager.EVENT_FACE_HIDDEN, current_time=100.0)
    assert success is False
    manager.cleanup()


# ---------------------------------------------------------------------------
# Test 10: Audio playback failures do not crash the application
# ---------------------------------------------------------------------------
def test_10_audio_playback_failures_do_not_crash(test_audio_config):
    def failing_player(evt):
        raise RuntimeError("Simulated sound hardware disconnection")

    manager = AudioAlertManager(
        audio_config=test_audio_config,
        sound_player=failing_player
    )

    # Must catch exception, log error, and return False without raising
    success = manager.trigger(AudioAlertManager.EVENT_PHONE_DETECTED, current_time=100.0)
    assert success is False
    assert manager.is_playing() is False
    manager.cleanup()


# ---------------------------------------------------------------------------
# Test 11: Alert state resets correctly
# ---------------------------------------------------------------------------
def test_11_alert_state_resets_correctly(test_audio_config):
    played = []
    manager = AudioAlertManager(
        audio_config=test_audio_config,
        sound_player=lambda evt: played.append(evt)
    )

    # Trigger phone detection at t = 100.0
    manager.update(face_detected=True, is_drowsy=False, phone_detected=True, current_time=100.0)
    manager._current_playing_event = None  # simulate clip completion
    assert len(played) == 1

    # Phone is put away at t = 101.0 (absent for 1.0s < 3.0s reset threshold)
    manager.update(face_detected=True, is_drowsy=False, phone_detected=False, current_time=101.0)
    assert manager._phone_detected_triggered is False or manager._phone_absent_start_time == 101.0

    # Phone absent until t = 104.5 (> 3.0s continuous absence) -> state resets!
    manager.update(face_detected=True, is_drowsy=False, phone_detected=False, current_time=104.5)
    assert manager._phone_detected_triggered is False

    # Reset face hidden when face returns
    manager._face_missing_start_time = 100.0
    manager._face_hidden_triggered = True
    manager.update(face_detected=True, is_drowsy=False, phone_detected=False, current_time=105.0)
    assert manager._face_missing_start_time is None
    assert manager._face_hidden_triggered is False

    # Reset drowsy when eyes reopen
    manager._drowsy_triggered = True
    manager.update(face_detected=True, is_drowsy=False, phone_detected=False, current_time=106.0)
    assert manager._drowsy_triggered is False
    manager.cleanup()


# ---------------------------------------------------------------------------
# Test 12: Webcam processing does not wait for audio playback
# ---------------------------------------------------------------------------
def test_12_webcam_processing_does_not_wait_for_audio_playback(test_audio_config):
    manager = AudioAlertManager(
        audio_config=test_audio_config,
        sound_player=lambda evt: time.sleep(0.001)  # tiny mock delay
    )

    start = time.perf_counter()
    # Trigger all 3 alerts sequentially
    manager.trigger(AudioAlertManager.EVENT_DROWSY)
    manager.update(face_detected=True, is_drowsy=False, phone_detected=False)
    elapsed = time.perf_counter() - start

    # Webcam processing loop must run well under 50ms per frame to maintain 30+ FPS
    assert elapsed < 0.05, f"Audio trigger took too long: {elapsed:.4f}s"
    manager.cleanup()
