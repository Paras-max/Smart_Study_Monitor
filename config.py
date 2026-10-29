"""
Configuration Settings for Smart Study Monitor (Phase 2)
=========================================================
Centralized parameters for camera, facial landmark monitoring,
eye aspect ratio (EAR) drowsiness detection, and YOLO phone detection.
"""

from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
WEIGHTS_PATH = BASE_DIR / "yolov8n.pt"

# Camera Configuration
CAMERA = {
    # Physical laptop webcam (HP True Vision FHD Camera)
    "PRIMARY_INDEX": 0,
    # Set to None by default so virtual cameras (e.g. OBS Virtual Camera on index 1)
    # are not mistaken for the physical webcam unless explicitly enabled.
    "FALLBACK_INDEX": None,
    "WIDTH": 1280,
    "HEIGHT": 720,
    "MIRROR": True,
}

# MediaPipe Face Mesh Configuration
FACE_MESH = {
    "STATIC_IMAGE_MODE": False,
    "MAX_NUM_FACES": 1,
    "REFINE_LANDMARKS": True,
    "MIN_DETECTION_CONFIDENCE": 0.5,
    "MIN_TRACKING_CONFIDENCE": 0.5,
}

# Eye Monitoring & Drowsiness Heuristics
# Standard 468/478 MediaPipe landmark indices for eye contours
EYE_MONITOR = {
    # Left eye landmarks: [outer, inner, top1, bottom1, top2, bottom2]
    # p1=33 (outer), p4=133 (inner), p2=160 (top), p6=144 (bottom), p3=158 (top), p5=153 (bottom)
    "LEFT_EYE_LANDMARKS": {
        "p1": 33,
        "p2": 160,
        "p3": 158,
        "p4": 133,
        "p5": 153,
        "p6": 144,
    },
    # Right eye landmarks:
    # p1=362 (outer), p4=263 (inner), p2=385 (top), p6=380 (bottom), p3=387 (top), p5=373 (bottom)
    "RIGHT_EYE_LANDMARKS": {
        "p1": 362,
        "p2": 385,
        "p3": 387,
        "p4": 263,
        "p5": 373,
        "p6": 380,
    },
    # Eye Aspect Ratio (EAR) Threshold:
    # EAR below this threshold is treated as closed eyes.
    # Normal open eyes: ~0.26 - 0.35. Closed eyes: ~0.14 - 0.20.
    "EAR_THRESHOLD": 0.22,
    # Duration (seconds) eyes must remain continuously closed to trigger drowsiness warning
    "DROWSINESS_DURATION_SEC": 2.0,
}

# YOLO Phone Detection Configuration
PHONE_DETECTOR = {
    "MODEL_PATH": str(WEIGHTS_PATH),
    "CONFIDENCE_THRESHOLD": 0.45,
    # In COCO dataset: Class 67 is 'cell phone'
    "TARGET_CLASS_ID": 67,
    "TARGET_CLASS_NAME": "cell phone",
    # Run YOLO inference every N frames to keep 30+ FPS responsive CPU processing
    "INFERENCE_INTERVAL_FRAMES": 4,
}

# UI & Color Schemes (BGR format for OpenCV)
UI = {
    "WINDOW_NAME": "Smart Study Monitor - Phase 2",
    "COLOR_GREEN": (0, 255, 0),
    "COLOR_RED": (0, 0, 255),
    "COLOR_ORANGE": (0, 140, 255),
    "COLOR_YELLOW": (0, 255, 255),
    "COLOR_WHITE": (255, 255, 255),
    "COLOR_GRAY": (180, 180, 180),
    "COLOR_DARK_GRAY": (20, 20, 20),
    "COLOR_ALERT_BG": (0, 0, 150),
    "FPS_SMOOTHING_FRAMES": 30,
}
