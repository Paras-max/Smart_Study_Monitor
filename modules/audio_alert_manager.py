"""
Audio Alert Manager (Phase 3)
=============================
Handles real-time audio alerts for:
1. Face Hidden Alert: "Aapko koi nahi dekh raha, muh dikhao!"
2. Sleeping / Drowsiness Alert: "Uth ja!"
3. Phone Detected Alert: "Tum IAS ki tayari chhod do!"

Key Architecture:
- Non-blocking asynchronous playback (never freezes OpenCV webcam loop).
- Alert priority hierarchy (FACE_HIDDEN > DROWSY > PHONE_DETECTED > FOCUSED_STUDY).
- Single-stream playback guard (never interrupts an active audio clip).
- Priority queuing for incoming alerts during playback.
- Configurable individual cooldown timers to prevent alert spam.
- State resets when user resumes attentive study.
- Graceful handling of missing files, audio driver absence, and playback errors.
"""

import logging
import os
from pathlib import Path
import time
from typing import Callable, Dict, Optional, Tuple

import pygame

import config

logger = logging.getLogger(__name__)


class AlertPriority:
    """Alert priority hierarchy (lower integer = higher priority)."""
    FACE_HIDDEN = 1
    DROWSY = 2
    PHONE_DETECTED = 3
    FOCUSED_STUDY = 4


class AudioAlertManager:
    """
    Manages detection-to-audio event triggering, cooldown timers,
    asynchronous playback, and priority queuing.
    """

    EVENT_FACE_HIDDEN = "FACE_HIDDEN"
    EVENT_DROWSY = "DROWSY"
    EVENT_PHONE_DETECTED = "PHONE_DETECTED"
    EVENT_FOCUSED = "FOCUSED_STUDY"

    # Human-readable alert display names for HUD
    ALERT_DISPLAY_NAMES = {
        EVENT_FACE_HIDDEN: "Face Hidden",
        EVENT_DROWSY: "Sleeping / Drowsy",
        EVENT_PHONE_DETECTED: "Phone Detected",
        EVENT_FOCUSED: "None",
    }

    PRIORITIES = {
        EVENT_FACE_HIDDEN: AlertPriority.FACE_HIDDEN,
        EVENT_DROWSY: AlertPriority.DROWSY,
        EVENT_PHONE_DETECTED: AlertPriority.PHONE_DETECTED,
        EVENT_FOCUSED: AlertPriority.FOCUSED_STUDY,
    }

    def __init__(
        self,
        audio_config: Optional[dict] = None,
        sound_player: Optional[Callable[[str], None]] = None,
    ):
        """
        Initialize the AudioAlertManager.

        :param audio_config: Configuration dictionary (defaults to config.AUDIO_ALERTS).
        :param sound_player: Optional custom playback function for testing/mocking.
        """
        self.cfg = audio_config or getattr(config, "AUDIO_ALERTS", {})
        self.enabled = self.cfg.get("ENABLED", True)
        self.sound_player = sound_player

        # Cooldown settings (seconds)
        self.cooldowns: Dict[str, float] = {
            self.EVENT_FACE_HIDDEN: float(self.cfg.get("FACE_HIDDEN_COOLDOWN_SEC", 15.0)),
            self.EVENT_DROWSY: float(self.cfg.get("SLEEPING_COOLDOWN_SEC", 15.0)),
            self.EVENT_PHONE_DETECTED: float(self.cfg.get("PHONE_COOLDOWN_SEC", 20.0)),
        }

        # Timers and state thresholds
        self.face_hidden_grace_sec = float(self.cfg.get("FACE_HIDDEN_GRACE_PERIOD_SEC", 3.0))
        self.phone_absence_reset_sec = float(self.cfg.get("PHONE_ABSENCE_RESET_SEC", 3.0))

        # Event tracking states
        self.last_played_time: Dict[str, float] = {}
        self.last_alert_name: str = "None"
        self._current_playing_event: Optional[str] = None
        self._pending_queued_event: Optional[str] = None

        # Continuous condition timers
        self._face_missing_start_time: Optional[float] = None
        self._face_hidden_triggered: bool = False
        self._drowsy_triggered: bool = False
        self._phone_absent_start_time: Optional[float] = None
        self._phone_detected_triggered: bool = False

        # Audio backend & files
        self._audio_available: bool = False
        self.sounds: Dict[str, Optional[pygame.mixer.Sound]] = {}
        self.audio_paths: Dict[str, Path] = {}
        self._channel: Optional[pygame.mixer.Channel] = None

        # Initialize audio engine
        self._init_audio_engine()
        self._load_audio_clips()

    def _init_audio_engine(self) -> None:
        """Initializes pygame mixer safely, handling headless or driverless environments."""
        if not self.enabled:
            logger.info("AudioAlertManager: Audio alerts disabled via configuration.")
            return

        if self.sound_player is not None:
            # Custom / mock sound player injected
            self._audio_available = True
            return

        try:
            if not pygame.mixer.get_init():
                # 44100Hz, 16-bit, stereo, 2048 buffer
                pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=2048)
            # Allocate dedicated channel 0 for alerts to enforce single-clip playback
            pygame.mixer.set_num_channels(4)
            self._channel = pygame.mixer.Channel(0)
            self._audio_available = True
            logger.info("AudioAlertManager: Pygame audio mixer initialized successfully.")
        except Exception as e:
            logger.warning(
                "AudioAlertManager: Could not initialize audio device (%s). "
                "Running in visual-only mode.",
                e,
            )
            self._audio_available = False

    def _load_audio_clips(self) -> None:
        """Loads and verifies the three required MP3 voice alert files."""
        clips_cfg = self.cfg.get("CLIPS", {})

        for event, path in clips_cfg.items():
            file_path = Path(path)
            self.audio_paths[event] = file_path

            if not file_path.exists():
                logger.warning(
                    "AudioAlertManager: Audio file for event '%s' not found at '%s'.",
                    event,
                    file_path,
                )
                self.sounds[event] = None
                continue

            if self.sound_player is not None:
                # Mock mode: mark as loaded
                self.sounds[event] = True  # type: ignore
                continue

            if self._audio_available:
                try:
                    sound = pygame.mixer.Sound(str(file_path))
                    self.sounds[event] = sound
                    logger.info(
                        "AudioAlertManager: Loaded '%s' from '%s' (length: %.2fs)",
                        event,
                        file_path.name,
                        sound.get_length(),
                    )
                except Exception as e:
                    logger.error(
                        "AudioAlertManager: Failed to load audio file '%s': %s",
                        file_path,
                        e,
                    )
                    self.sounds[event] = None
            else:
                self.sounds[event] = None

    def is_playing(self) -> bool:
        """
        Checks whether an audio alert is currently playing.
        Uses the dedicated mixer channel or mock status.
        """
        if not self._audio_available or not self.enabled:
            return False

        if self.sound_player is not None:
            return self._current_playing_event is not None

        if self._channel is not None:
            try:
                busy = self._channel.get_busy()
                if not busy and self._current_playing_event is not None:
                    # Clip completed playback
                    self._current_playing_event = None
                return busy
            except Exception:
                return False

        return False

    def is_on_cooldown(self, event_type: str, current_time: Optional[float] = None) -> bool:
        """Checks if a given event type is currently in its cooldown period."""
        now = time.monotonic() if current_time is None else current_time
        last_time = self.last_played_time.get(event_type)
        if last_time is None:
            return False
        cooldown = self.cooldowns.get(event_type, 0.0)
        return (now - last_time) < cooldown

    def get_remaining_cooldown(self, event_type: str, current_time: Optional[float] = None) -> float:
        """Returns the remaining cooldown time in seconds (0.0 if not on cooldown)."""
        now = time.monotonic() if current_time is None else current_time
        last_time = self.last_played_time.get(event_type)
        if last_time is None:
            return 0.0
        cooldown = self.cooldowns.get(event_type, 0.0)
        remaining = cooldown - (now - last_time)
        return max(0.0, remaining)

    def trigger(self, event_type: str, current_time: Optional[float] = None) -> bool:
        """
        Attempts to trigger and play a voice alert for the specified event.

        Adheres strictly to:
        - Priority hierarchy
        - Non-interruption rule (never stops an active clip)
        - Priority queueing (queues higher priority alert if busy)
        - Cooldown checks

        :param event_type: 'FACE_HIDDEN', 'DROWSY', or 'PHONE_DETECTED'.
        :param current_time: Monotonic timestamp (defaults to time.monotonic()).
        :return: True if playback started immediately or alert was queued, False otherwise.
        """
        if not self.enabled:
            return False

        now = time.monotonic() if current_time is None else current_time

        # Validate event
        if event_type not in self.PRIORITIES or event_type == self.EVENT_FOCUSED:
            return False

        # Check cooldown
        if self.is_on_cooldown(event_type, now):
            return False

        # Check if audio is currently playing
        if self.is_playing():
            # Rule: Never interrupt an audio clip that is already playing.
            # Queue the incoming alert according to priority hierarchy.
            incoming_priority = self.PRIORITIES[event_type]

            if self._pending_queued_event is None:
                # Queue this event
                self._pending_queued_event = event_type
                logger.info(
                    "AudioAlertManager: Clip busy ('%s'). Queued '%s'.",
                    self._current_playing_event,
                    event_type,
                )
                return True
            else:
                # Compare with currently pending event
                existing_priority = self.PRIORITIES[self._pending_queued_event]
                if incoming_priority < existing_priority:
                    # Higher priority replaces lower priority in queue
                    logger.info(
                        "AudioAlertManager: Replacing queued '%s' with higher priority '%s'.",
                        self._pending_queued_event,
                        event_type,
                    )
                    self._pending_queued_event = event_type
                    return True
                else:
                    # Stale / lower-priority or repeat event discarded
                    return False

        # Channel is free: play immediately
        return self._play_clip(event_type, now)

    def _play_clip(self, event_type: str, current_time: float) -> bool:
        """Plays the audio clip for event_type asynchronously."""
        sound = self.sounds.get(event_type)
        self.last_played_time[event_type] = current_time
        self.last_alert_name = self.ALERT_DISPLAY_NAMES.get(event_type, event_type)
        self._current_playing_event = event_type

        # Custom / mock sound player
        if self.sound_player is not None:
            try:
                self.sound_player(event_type)
                return True
            except Exception as e:
                logger.error("AudioAlertManager: Custom sound player failed: %s", e)
                self._current_playing_event = None
                return False

        # Missing audio file check
        if sound is None:
            logger.warning(
                "AudioAlertManager: Cannot play '%s' (audio file missing or failed to load).",
                event_type,
            )
            # Do not crash; clear current playing event
            self._current_playing_event = None
            return False

        # Pygame mixer playback
        try:
            if self._channel is not None:
                self._channel.play(sound)
            else:
                sound.play()
            logger.info("AudioAlertManager: Playing '%s' voice alert.", event_type)
            return True
        except Exception as e:
            logger.error("AudioAlertManager: Playback failed for '%s': %s", event_type, e)
            self._current_playing_event = None
            return False

    def process_queue(self, current_time: Optional[float] = None) -> Optional[str]:
        """
        Checks if the active clip finished playing and pops/plays any pending queued alert.
        Should be called every frame in the main video loop.
        """
        now = time.monotonic() if current_time is None else current_time

        if not self.is_playing() and self._pending_queued_event is not None:
            queued = self._pending_queued_event
            self._pending_queued_event = None
            # Check if queued event expired onto cooldown or can play
            if not self.is_on_cooldown(queued, now):
                self._play_clip(queued, now)
                return queued

        return None

    def update(
        self,
        face_detected: bool,
        is_drowsy: bool,
        phone_detected: bool,
        current_time: Optional[float] = None,
    ) -> Tuple[str, str]:
        """
        Primary interface for the main application loop.
        Evaluates detection conditions, manages timers, triggers alerts,
        processes queue, and returns overall study status.

        :param face_detected: Whether face is currently visible.
        :param is_drowsy: Whether student eyes are closed continuously for drowsiness duration.
        :param phone_detected: Whether cell phone is detected in current frame.
        :param current_time: Monotonic timestamp (defaults to time.monotonic()).
        :return: Tuple of (status_category, triggered_alert_name)
                 status_category: 'FOCUSED', 'SLEEPING', or 'DISTRACTED'
                 triggered_alert_name: human-readable name of last or active alert.
        """
        now = time.monotonic() if current_time is None else current_time

        # Step 1: Process queue from previous playback completion
        self.process_queue(now)

        # Step 2: Track stateful condition timers
        # Condition A: Face Missing (Grace period: 3 seconds)
        face_hidden_alert_ready = False
        if not face_detected:
            if self._face_missing_start_time is None:
                self._face_missing_start_time = now
            elif (now - self._face_missing_start_time) >= self.face_hidden_grace_sec:
                face_hidden_alert_ready = True
        else:
            # Face restored -> reset grace timer and trigger state
            self._face_missing_start_time = None
            self._face_hidden_triggered = False

        # Condition B: Drowsiness (only valid when face is visible)
        drowsy_alert_ready = is_drowsy and face_detected
        if not is_drowsy or not face_detected:
            self._drowsy_triggered = False

        # Condition C: Phone Detection & Absence Reset
        phone_alert_ready = False
        if phone_detected:
            self._phone_absent_start_time = None
            # Only trigger phone alert if face is visible or not in missing state
            if face_detected:
                phone_alert_ready = True
        else:
            if self._phone_absent_start_time is None:
                self._phone_absent_start_time = now
            elif (now - self._phone_absent_start_time) >= self.phone_absence_reset_sec:
                # Phone has been absent continuously for >= 3 seconds -> reset state
                self._phone_detected_triggered = False

        # Step 3: Priority Hierarchy Triggering
        # Priority 1: FACE_HIDDEN
        # Priority 2: DROWSY
        # Priority 3: PHONE_DETECTED
        # Priority 4: FOCUSED_STUDY
        if face_hidden_alert_ready:
            if not self._face_hidden_triggered and not self.is_on_cooldown(self.EVENT_FACE_HIDDEN, now):
                if self.trigger(self.EVENT_FACE_HIDDEN, now):
                    self._face_hidden_triggered = True

        elif drowsy_alert_ready:
            if not self._drowsy_triggered and not self.is_on_cooldown(self.EVENT_DROWSY, now):
                if self.trigger(self.EVENT_DROWSY, now):
                    self._drowsy_triggered = True

        elif phone_alert_ready:
            if not self._phone_detected_triggered and not self.is_on_cooldown(self.EVENT_PHONE_DETECTED, now):
                if self.trigger(self.EVENT_PHONE_DETECTED, now):
                    self._phone_detected_triggered = True

        # Step 4: Determine current high-level student status
        if drowsy_alert_ready or is_drowsy:
            status_category = "SLEEPING"
        elif not face_detected or phone_detected:
            status_category = "DISTRACTED"
        else:
            status_category = "FOCUSED"

        return status_category, self.last_alert_name

    def stop(self) -> None:
        """Stops any active playback immediately."""
        if self._channel is not None:
            try:
                self._channel.stop()
            except Exception:
                pass
        self._current_playing_event = None
        self._pending_queued_event = None

    def cleanup(self) -> None:
        """Stops playback, releases sound objects, and uninitializes mixer."""
        logger.info("AudioAlertManager: Releasing audio resources...")
        self.stop()
        self.sounds.clear()
        if self.sound_player is None and pygame.mixer.get_init():
            try:
                pygame.mixer.quit()
                logger.info("AudioAlertManager: Pygame mixer uninitialized.")
            except Exception as e:
                logger.warning("AudioAlertManager: Error during mixer cleanup: %s", e)
        self._audio_available = False
