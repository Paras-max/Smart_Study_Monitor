"""
Eye Monitoring & Drowsiness Detection Module
=============================================
Calculates Eye Aspect Ratio (EAR) using MediaPipe 3D facial landmarks.
Detects prolonged eye closure using monotonic time tracking to alert the student:
"Wake Up & Study!"

Formula (Soukupova and Cech, 2016):
    EAR = ( ||p2 - p6|| + ||p3 - p5|| ) / ( 2 * ||p1 - p4|| )

Landmark Indices (MediaPipe Face Mesh):
    Left Eye:
        p1 (outer corner): 33
        p4 (inner corner): 133
        p2, p6 (vertical pair 1): 160, 144
        p3, p5 (vertical pair 2): 158, 153
    Right Eye:
        p1 (outer corner): 362
        p4 (inner corner): 263
        p2, p6 (vertical pair 1): 385, 380
        p3, p5 (vertical pair 2): 387, 373

Note: This heuristic serves as a study attention monitor, not a medical-grade diagnostic tool.
"""

import math
import time
from typing import Dict, Optional, Tuple


def euclidean_distance_2d(p_a: Tuple[float, float], p_b: Tuple[float, float]) -> float:
    """Calculates 2D Euclidean distance between two points."""
    return math.hypot(p_a[0] - p_b[0], p_a[1] - p_b[1])


def calculate_single_ear(landmarks_dict: Dict[str, Tuple[float, float]]) -> float:
    """
    Calculates the Eye Aspect Ratio (EAR) for a single eye given 6 boundary points:
    p1: outer corner, p4: inner corner,
    p2, p6: upper/lower vertical pair 1,
    p3, p5: upper/lower vertical pair 2.
    """
    p1 = landmarks_dict["p1"]
    p2 = landmarks_dict["p2"]
    p3 = landmarks_dict["p3"]
    p4 = landmarks_dict["p4"]
    p5 = landmarks_dict["p5"]
    p6 = landmarks_dict["p6"]

    vertical_1 = euclidean_distance_2d(p2, p6)
    vertical_2 = euclidean_distance_2d(p3, p5)
    horizontal = euclidean_distance_2d(p1, p4)

    if horizontal <= 1e-6:
        return 0.0

    return (vertical_1 + vertical_2) / (2.0 * horizontal)


class EyeMonitor:
    """
    Monitors eye openness and tracks prolonged closure durations using monotonic timestamps.
    """

    def __init__(
        self,
        ear_threshold: float = 0.22,
        drowsiness_duration_sec: float = 2.0,
        left_eye_indices: Optional[Dict[str, int]] = None,
        right_eye_indices: Optional[Dict[str, int]] = None,
    ):
        """
        Args:
            ear_threshold: EAR values below this are classified as closed eyes.
            drowsiness_duration_sec: Seconds eyes must remain closed before drowsiness alert fires.
            left_eye_indices: Landmark index mapping for left eye.
            right_eye_indices: Landmark index mapping for right eye.
        """
        self.ear_threshold = ear_threshold
        self.drowsiness_duration_sec = drowsiness_duration_sec

        # Default standard MediaPipe indices
        self.left_eye_indices = left_eye_indices or {
            "p1": 33,
            "p2": 160,
            "p3": 158,
            "p4": 133,
            "p5": 153,
            "p6": 144,
        }
        self.right_eye_indices = right_eye_indices or {
            "p1": 362,
            "p2": 385,
            "p3": 387,
            "p4": 263,
            "p5": 373,
            "p6": 380,
        }

        # Monotonic timestamp marking when continuous eye closure began
        self.closure_start_time: Optional[float] = None
        self.closed_duration: float = 0.0

    def reset(self) -> None:
        """Resets eye closure timer (used when face is lost or eyes reopen)."""
        self.closure_start_time = None
        self.closed_duration = 0.0

    def extract_eye_points(
        self,
        landmarks,
        indices_map: Dict[str, int],
        frame_width: int,
        frame_height: int,
    ) -> Optional[Dict[str, Tuple[float, float]]]:
        """
        Extracts pixel coordinates for an eye from normalized MediaPipe landmarks.
        Returns None if any landmark index is out of bounds or missing.
        """
        if not landmarks or not hasattr(landmarks, "landmark"):
            return None

        total_landmarks = len(landmarks.landmark)
        points = {}

        for key, idx in indices_map.items():
            if idx < 0 or idx >= total_landmarks:
                return None
            lm = landmarks.landmark[idx]
            px = lm.x * frame_width
            py = lm.y * frame_height
            points[key] = (px, py)

        return points

    def process(
        self,
        face_landmarks,
        frame_width: int,
        frame_height: int,
        current_time: Optional[float] = None,
    ) -> Dict:
        """
        Processes a detected face's landmarks to calculate EAR and drowsiness state.
        
        Args:
            face_landmarks: MediaPipe NormalizedLandmarkList (or None if no face).
            frame_width: Video frame width in pixels.
            frame_height: Video frame height in pixels.
            current_time: Optional monotonic timestamp (useful for reproducible unit testing).
            
        Returns:
            dict containing:
                - 'is_face_present': bool
                - 'left_ear': float
                - 'right_ear': float
                - 'avg_ear': float
                - 'are_eyes_closed': bool
                - 'closed_duration': float
                - 'is_drowsy': bool
                - 'status_text': str
        """
        now = time.monotonic() if current_time is None else current_time

        # Condition 1: No face or invalid landmarks object
        if face_landmarks is None or not hasattr(face_landmarks, "landmark"):
            self.reset()
            return {
                "is_face_present": False,
                "left_ear": 0.0,
                "right_ear": 0.0,
                "avg_ear": 0.0,
                "are_eyes_closed": False,
                "closed_duration": 0.0,
                "is_drowsy": False,
                "status_text": "No Face Detected",
            }

        # Condition 2: Extract eye coordinate points
        left_pts = self.extract_eye_points(face_landmarks, self.left_eye_indices, frame_width, frame_height)
        right_pts = self.extract_eye_points(face_landmarks, self.right_eye_indices, frame_width, frame_height)

        if left_pts is None or right_pts is None:
            self.reset()
            return {
                "is_face_present": False,
                "left_ear": 0.0,
                "right_ear": 0.0,
                "avg_ear": 0.0,
                "are_eyes_closed": False,
                "closed_duration": 0.0,
                "is_drowsy": False,
                "status_text": "Invalid Eye Landmarks",
            }

        left_ear = calculate_single_ear(left_pts)
        right_ear = calculate_single_ear(right_pts)
        avg_ear = (left_ear + right_ear) / 2.0

        # Evaluate closure state
        are_eyes_closed = avg_ear < self.ear_threshold

        if are_eyes_closed:
            if self.closure_start_time is None:
                self.closure_start_time = now
            self.closed_duration = max(0.0, now - self.closure_start_time)
        else:
            # Eyes are open: reset closure timer immediately
            self.reset()

        is_drowsy = self.closed_duration >= self.drowsiness_duration_sec

        if is_drowsy:
            status_text = "Wake Up & Study!"
        elif are_eyes_closed:
            status_text = f"Eyes Closed ({self.closed_duration:.1f}s)"
        else:
            status_text = "Eyes Open"

        return {
            "is_face_present": True,
            "left_ear": round(left_ear, 3),
            "right_ear": round(right_ear, 3),
            "avg_ear": round(avg_ear, 3),
            "are_eyes_closed": are_eyes_closed,
            "closed_duration": round(self.closed_duration, 2),
            "is_drowsy": is_drowsy,
            "status_text": status_text,
        }
