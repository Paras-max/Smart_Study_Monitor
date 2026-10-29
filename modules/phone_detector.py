"""
YOLO Phone Detection Module
===========================
Integrates Ultralytics YOLOv8 for detecting cell phones (COCO Class ID 67).
Features:
- CPU-friendly inference with periodic frame scheduling (preserves 30+ FPS).
- Automatic model downloading and error-resilient loading.
- Distinguishes clearly between 'No Phone Detected', 'Phone Detected', and 'Inference Error'.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np


class PhoneDetector:
    """
    Manages YOLO model inference for detecting cell phones in video frames.
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.45,
        target_class_id: int = 67,
        inference_interval: int = 4,
    ):
        """
        Args:
            model_path: Path or filename of YOLO weights (defaults to yolov8n.pt).
            confidence_threshold: Minimum confidence score to register detection.
            target_class_id: COCO class ID for cell phone (default 67).
            inference_interval: Run inference every N frames to conserve CPU.
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.target_class_id = target_class_id
        self.inference_interval = max(1, inference_interval)

        self.model = None
        self.is_loaded = False
        self.load_error: Optional[str] = None

        # State caching for smooth rendering between skipped frames
        self.frame_counter = 0
        self.last_result: Dict = {
            "phone_detected": False,
            "boxes": [],
            "status_message": "No Phone Detected",
            "inference_success": True,
            "error_detail": None,
        }

        self._initialize_model()

    def _initialize_model(self) -> None:
        """Loads the YOLO model weights on CPU."""
        try:
            from ultralytics import YOLO

            print(f"[INFO] Loading YOLO model from '{self.model_path}'...")
            self.model = YOLO(self.model_path)
            self.is_loaded = True
            self.load_error = None
            print("[SUCCESS] YOLO phone detector model loaded successfully.")
        except Exception as e:
            self.is_loaded = False
            self.load_error = str(e)
            print(f"[ERROR] Failed to load YOLO model: {e}")

    def detect(self, frame: np.ndarray, force_inference: bool = False) -> Dict:
        """
        Runs YOLO inference on the given frame to detect cell phones.
        Uses cached detection results when skipping frames to maximize FPS.

        Returns:
            dict containing:
                - 'phone_detected': bool
                - 'boxes': list of dicts [{'x1', 'y1', 'x2', 'y2', 'conf', 'label'}]
                - 'status_message': str ('Put the Phone Away!', 'No Phone Detected', or error)
                - 'inference_success': bool
                - 'error_detail': Optional[str]
        """
        # If model failed to load, report explicit error without crashing
        if not self.is_loaded or self.model is None:
            return {
                "phone_detected": False,
                "boxes": [],
                "status_message": "Phone Detector Offline",
                "inference_success": False,
                "error_detail": self.load_error or "Model not loaded",
            }

        self.frame_counter += 1

        # Check if this frame should run fresh inference or reuse cached results
        should_run = force_inference or (self.frame_counter % self.inference_interval == 0)

        if not should_run:
            return self.last_result

        try:
            # Run inference on CPU with verbose=False for silent operation
            # Downscaling inference size to 480 or 640 balances CPU speed with accuracy
            results = self.model.predict(
                source=frame,
                conf=self.confidence_threshold,
                classes=[self.target_class_id],
                device="cpu",
                verbose=False,
                imgsz=480,
            )

            detected_boxes = []
            if results and len(results) > 0:
                first_result = results[0]
                if hasattr(first_result, "boxes") and first_result.boxes is not None:
                    boxes = first_result.boxes
                    for i in range(len(boxes)):
                        cls_id = int(boxes.cls[i].item())
                        conf = float(boxes.conf[i].item())

                        if cls_id == self.target_class_id and conf >= self.confidence_threshold:
                            xyxy = boxes.xyxy[i].cpu().numpy().astype(int)
                            detected_boxes.append({
                                "x1": int(xyxy[0]),
                                "y1": int(xyxy[1]),
                                "x2": int(xyxy[2]),
                                "y2": int(xyxy[3]),
                                "conf": round(conf, 2),
                                "label": f"Phone: {int(conf * 100)}%",
                            })

            has_phone = len(detected_boxes) > 0
            status_msg = "Put the Phone Away!" if has_phone else "No Phone Detected"

            self.last_result = {
                "phone_detected": has_phone,
                "boxes": detected_boxes,
                "status_message": status_msg,
                "inference_success": True,
                "error_detail": None,
            }

        except Exception as e:
            self.last_result = {
                "phone_detected": False,
                "boxes": [],
                "status_message": "Inference Error",
                "inference_success": False,
                "error_detail": str(e),
            }

        return self.last_result

    def draw_detections(
        self,
        frame: np.ndarray,
        boxes: List[Dict],
        box_color: Tuple[int, int, int] = (0, 69, 255),  # Vibrant orange-red (BGR)
    ) -> None:
        """Draws bounding boxes and confidence badges on detected phones."""
        for box in boxes:
            x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
            label = box.get("label", "Phone")

            # Bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

            # Label badge
            (lbl_w, lbl_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.55, 1)
            y_badge_top = max(0, y1 - lbl_h - 8)
            cv2.rectangle(
                frame,
                (x1, y_badge_top),
                (x1 + lbl_w + 10, y1),
                box_color,
                -1
            )
            cv2.putText(
                frame,
                label,
                (x1 + 5, y1 - 4),
                cv2.FONT_HERSHEY_DUPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )
