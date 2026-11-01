"""
Smart Study Monitor - Phase 3: Main Application
================================================
Integrates:
1. OpenCV Webcam Video Pipeline with horizontal mirroring.
2. MediaPipe Face Mesh for facial landmarks & presence monitoring.
3. Eye Aspect Ratio (EAR) drowsiness detector ("Wake Up & Study!").
4. Ultralytics YOLOv8 for cell phone distraction detection ("Put the Phone Away!").
5. Intelligent Audio Alert Manager with 3 original voice alerts & priority queueing.
6. Real-time smoothed FPS and clean resource cleanup.

Controls:
    Press 'Q' or 'q' in the video window to quit cleanly.
"""

from collections import deque
import sys
import time
import traceback
import cv2
import mediapipe as mp

import config
from modules.audio_alert_manager import AudioAlertManager
from modules.eye_monitor import EyeMonitor
from modules.phone_detector import PhoneDetector


def draw_status_card(
    image,
    primary_text: str,
    sub_text: str,
    bg_color: tuple,
    text_color: tuple,
    metrics: list,
    y_pos: int = 70,
):
    """
    Renders a unified, semi-transparent HUD card with the primary alert
    and real-time monitoring metrics (Face, EAR, Phone).
    """
    font = cv2.FONT_HERSHEY_DUPLEX
    font_small = cv2.FONT_HERSHEY_SIMPLEX

    # Calculate dimensions
    (text_w, text_h), _ = cv2.getTextSize(primary_text, font, 0.75, 2)
    max_w = text_w

    metric_h = 22
    for label, val in metrics:
        line_str = f"{label}: {val}"
        (lw, _), _ = cv2.getTextSize(line_str, font_small, 0.52, 1)
        if lw > max_w:
            max_w = lw

    padding_x = 18
    padding_y = 12
    card_w = max_w + (padding_x * 2)
    card_h = text_h + padding_y * 2 + (len(metrics) * metric_h) + (18 if sub_text else 0)

    x1 = 20
    y1 = y_pos - text_h - padding_y
    x2 = x1 + card_w
    y2 = y1 + card_h

    # Semi-transparent overlay
    overlay = image.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), bg_color, -1)
    cv2.addWeighted(overlay, 0.8, image, 0.2, 0, image)

    # Card border
    cv2.rectangle(image, (x1, y1), (x2, y2), text_color, 1)

    # Primary status text
    curr_y = y_pos
    cv2.putText(image, primary_text, (x1 + padding_x, curr_y), font, 0.75, text_color, 2, cv2.LINE_AA)

    if sub_text:
        curr_y += 20
        cv2.putText(image, sub_text, (x1 + padding_x, curr_y), font_small, 0.52, (220, 220, 220), 1, cv2.LINE_AA)

    # Separator line
    curr_y += 10
    cv2.line(image, (x1 + 10, curr_y), (x2 - 10, curr_y), (100, 100, 100), 1)

    # Render key metrics
    for label, val in metrics:
        curr_y += metric_h
        cv2.putText(
            image,
            f"{label}: ",
            (x1 + padding_x, curr_y),
            font_small,
            0.50,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )
        (lbl_w, _), _ = cv2.getTextSize(f"{label}: ", font_small, 0.50, 1)
        cv2.putText(
            image,
            str(val),
            (x1 + padding_x + lbl_w, curr_y),
            font_small,
            0.50,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def initialize_webcam():
    """
    Attempts connection to candidate camera indices using DirectShow (CAP_DSHOW),
    Media Foundation (CAP_MSMF), and Default (CAP_ANY) backends sequentially.
    Verifies that a valid frame can actually be captured before accepting the stream.
    """
    # Only test configured indices (ignoring None)
    candidate_indices = [
        idx for idx in [config.CAMERA.get("PRIMARY_INDEX", 0), config.CAMERA.get("FALLBACK_INDEX")]
        if idx is not None
    ]
    backends = [
        ("DirectShow (CAP_DSHOW)", cv2.CAP_DSHOW),
        ("Media Foundation (CAP_MSMF)", cv2.CAP_MSMF),
        ("Default (CAP_ANY)", cv2.CAP_ANY),
    ]

    for idx in candidate_indices:
        print(f"[INFO] Checking camera index {idx}...")
        for b_name, b_api in backends:
            try:
                cap = cv2.VideoCapture(idx, b_api)
                if cap.isOpened():
                    ret, test_frame = cap.read()
                    if ret and test_frame is not None and test_frame.size > 0:
                        h, w = test_frame.shape[:2]
                        print(f"[SUCCESS] Connected to camera index {idx} via {b_name} ({w}x{h} frame captured).")
                        return cap, idx
                    else:
                        print(f"[DEBUG] Camera index {idx} ({b_name}) opened but returned no frame (ret={ret}).")
                        cap.release()
                else:
                    print(f"[DEBUG] Camera index {idx} ({b_name}) failed to open.")
                    cap.release()
            except Exception as e:
                print(f"[DEBUG] Exception testing index {idx} ({b_name}): {e}")

    return None, None


def run_monitor():
    """Main application loop coordinating Face Mesh, Eye Monitor, YOLO Phone Detector, and Audio Alerts."""
    print("=" * 65)
    print("   AI Smart Study Monitor - Phase 3 (Face + Eyes + Phone + Audio)")
    print("=" * 65)

    cap = None
    face_mesh = None
    phone_detector = None
    eye_monitor = None
    audio_alert_manager = None
    window_name = config.UI["WINDOW_NAME"]

    try:
        # 1. Initialize MediaPipe Face Mesh
        print("[INFO] Initializing MediaPipe Face Mesh...")
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=config.FACE_MESH["STATIC_IMAGE_MODE"],
            max_num_faces=config.FACE_MESH["MAX_NUM_FACES"],
            refine_landmarks=config.FACE_MESH["REFINE_LANDMARKS"],
            min_detection_confidence=config.FACE_MESH["MIN_DETECTION_CONFIDENCE"],
            min_tracking_confidence=config.FACE_MESH["MIN_TRACKING_CONFIDENCE"],
        )
        print("[SUCCESS] MediaPipe Face Mesh initialized.")

        # 2. Initialize Eye Monitor
        print("[INFO] Initializing Eye Monitor...")
        eye_monitor = EyeMonitor(
            ear_threshold=config.EYE_MONITOR["EAR_THRESHOLD"],
            drowsiness_duration_sec=config.EYE_MONITOR["DROWSINESS_DURATION_SEC"],
            left_eye_indices=config.EYE_MONITOR["LEFT_EYE_LANDMARKS"],
            right_eye_indices=config.EYE_MONITOR["RIGHT_EYE_LANDMARKS"],
        )
        print("[SUCCESS] Eye Monitor initialized.")

        # 3. Initialize YOLO Phone Detector
        print("[INFO] Initializing YOLO Phone Detector...")
        phone_detector = PhoneDetector(
            model_path=config.PHONE_DETECTOR["MODEL_PATH"],
            confidence_threshold=config.PHONE_DETECTOR["CONFIDENCE_THRESHOLD"],
            target_class_id=config.PHONE_DETECTOR["TARGET_CLASS_ID"],
            inference_interval=config.PHONE_DETECTOR["INFERENCE_INTERVAL_FRAMES"],
        )
        print("[SUCCESS] YOLO Phone Detector initialized.")

        # 4. Initialize Audio Alert Manager (Phase 3)
        print("[INFO] Initializing Audio Alert Manager (Phase 3)...")
        audio_alert_manager = AudioAlertManager()
        print("[SUCCESS] Audio Alert Manager initialized.")

        # 5. Initialize Webcam
        cap, active_cam_idx = initialize_webcam()
        if cap is None:
            print("\n" + "!" * 65)
            print("[ERROR] Could not access any webcam (tried indices 0 and 1).")
            print("Troubleshooting checklist:")
            print(" 1. Check physical privacy shutter on your laptop camera.")
            print(" 2. Ensure no other application (Zoom, Teams, Discord, Meet) is using the webcam.")
            print(" 3. Verify Windows Camera Privacy Settings allow desktop applications.")
            print("!" * 65 + "\n")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA["WIDTH"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA["HEIGHT"])

        # FPS calculation using deque
        frame_timestamps = deque(maxlen=config.UI["FPS_SMOOTHING_FRAMES"])
        fps_display = 0.0

        print(f"[SUCCESS] Active webcam index: {active_cam_idx}.")
        print("[INFO] Video stream active. Press 'Q' or close window to exit.\n")

        # 5. Continuous Video Processing Loop
        while True:
            success, frame = cap.read()
            if not success or frame is None:
                print("\n[WARN] Frame acquisition failed. Camera disconnected?")
                break

            # Mirror webcam feed horizontally
            if config.CAMERA["MIRROR"]:
                frame = cv2.flip(frame, 1)

            frame_height, frame_width = frame.shape[:2]

            # Calculate FPS
            curr_time = time.time()
            frame_timestamps.append(curr_time)
            if len(frame_timestamps) > 1:
                duration = frame_timestamps[-1] - frame_timestamps[0]
                if duration > 0:
                    fps_display = (len(frame_timestamps) - 1) / duration

            # --- MODULE 1: MediaPipe Face Mesh & Facial Presence ---
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False
            mesh_results = face_mesh.process(frame_rgb)
            frame_rgb.flags.writeable = True

            is_face_detected = (
                mesh_results.multi_face_landmarks is not None
                and len(mesh_results.multi_face_landmarks) > 0
            )

            primary_face_landmarks = None
            if is_face_detected:
                primary_face_landmarks = mesh_results.multi_face_landmarks[0]
                # Draw small green landmark points
                for lm in primary_face_landmarks.landmark:
                    cx = int(lm.x * frame_width)
                    cy = int(lm.y * frame_height)
                    cv2.circle(frame, (cx, cy), 1, config.UI["COLOR_GREEN"], -1, cv2.LINE_AA)

            # --- MODULE 2: Eye Monitoring & Drowsiness ---
            eye_results = eye_monitor.process(
                primary_face_landmarks,
                frame_width,
                frame_height,
                current_time=time.monotonic(),
            )

            # --- MODULE 3: YOLO Phone Detection ---
            phone_results = phone_detector.detect(frame)
            if phone_results["phone_detected"]:
                phone_detector.draw_detections(frame, phone_results["boxes"])

            # --- MODULE 4: Audio Alert Manager & Priority Evaluation ---
            now_mono = time.monotonic()
            status_category, last_alert_name = audio_alert_manager.update(
                face_detected=is_face_detected,
                is_drowsy=eye_results["is_drowsy"],
                phone_detected=phone_results["phone_detected"],
                current_time=now_mono,
            )

            # --- STATUS PRIORITY SYSTEM (Visual Badges & Borders) ---
            # Priority 1: Face Hidden (Highest Priority)
            # Priority 2: Sleeping / Drowsiness
            # Priority 3: Phone Detected
            # Priority 4: Focused Study
            if not is_face_detected:
                primary_status = "WARNING: Face Not Visible!"
                sub_status = "Keep Your Face Visible"
                badge_bg = (20, 20, 60)
                badge_color = config.UI["COLOR_RED"]

                # Red border around frame
                cv2.rectangle(frame, (0, 0), (frame_width - 1, frame_height - 1), config.UI["COLOR_RED"], 6)

            elif eye_results["is_drowsy"]:
                primary_status = "WARNING: Wake Up & Study!"
                sub_status = f"Eyes Closed for {eye_results['closed_duration']:.1f}s"
                badge_bg = (20, 40, 70)
                badge_color = config.UI["COLOR_YELLOW"]

                # Yellow warning border
                cv2.rectangle(frame, (0, 0), (frame_width - 1, frame_height - 1), config.UI["COLOR_YELLOW"], 6)

            elif phone_results["phone_detected"]:
                primary_status = "WARNING: Put the Phone Away!"
                sub_status = "Stay Focused on Your Study"
                badge_bg = (20, 20, 60)
                badge_color = config.UI["COLOR_ORANGE"]

                # Orange warning border
                cv2.rectangle(frame, (0, 0), (frame_width - 1, frame_height - 1), config.UI["COLOR_ORANGE"], 6)

            else:
                primary_status = "STATUS: Focused & Studying"
                sub_status = "Face Detected & Attentive"
                badge_bg = (20, 50, 20)
                badge_color = config.UI["COLOR_GREEN"]

            # Format live HUD metrics strictly per Phase 3 specifications:
            # - FACE: VISIBLE / NOT VISIBLE
            # - EYES: OPEN / CLOSED / DROWSY
            # - PHONE: DETECTED / NOT DETECTED
            # - STATUS: FOCUSED / DISTRACTED / SLEEPING
            # - LAST ALERT: Alert name
            face_metric = "VISIBLE" if is_face_detected else "NOT VISIBLE"

            if not is_face_detected:
                eye_metric = "NOT VISIBLE"
            elif eye_results["is_drowsy"]:
                eye_metric = f"DROWSY ({eye_results['closed_duration']:.1f}s)"
            elif eye_results["are_eyes_closed"]:
                eye_metric = f"CLOSED ({eye_results['closed_duration']:.1f}s)"
            else:
                eye_metric = f"OPEN (EAR {eye_results['avg_ear']:.2f})"

            phone_metric = "DETECTED" if phone_results["phone_detected"] else "NOT DETECTED"

            hud_metrics = [
                ("FACE", face_metric),
                ("EYES", eye_metric),
                ("PHONE", phone_metric),
                ("STATUS", status_category),
                ("LAST ALERT", last_alert_name),
            ]

            # --- UI RENDERING ---
            # Top dark bar
            top_bar_height = 42
            header_overlay = frame.copy()
            cv2.rectangle(header_overlay, (0, 0), (frame_width, top_bar_height), config.UI["COLOR_DARK_GRAY"], -1)
            cv2.addWeighted(header_overlay, 0.7, frame, 0.3, 0, frame)

            # Title
            cv2.putText(
                frame,
                "Smart Study Monitor - Phase 2",
                (16, 28),
                cv2.FONT_HERSHEY_DUPLEX,
                0.65,
                config.UI["COLOR_WHITE"],
                1,
                cv2.LINE_AA,
            )

            # FPS (Top Right)
            fps_str = f"FPS: {fps_display:4.1f}"
            (fps_w, _), _ = cv2.getTextSize(fps_str, cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)
            cv2.putText(
                frame,
                fps_str,
                (frame_width - fps_w - 20, 28),
                cv2.FONT_HERSHEY_DUPLEX,
                0.6,
                config.UI["COLOR_YELLOW"],
                1,
                cv2.LINE_AA,
            )

            # Status & Metrics HUD Card
            draw_status_card(
                frame,
                primary_text=primary_status,
                sub_text=sub_status,
                bg_color=badge_bg,
                text_color=badge_color,
                metrics=hud_metrics,
                y_pos=80,
            )

            # Bottom Hint
            cv2.putText(
                frame,
                "Press 'Q' to Exit",
                (20, frame_height - 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                config.UI["COLOR_GRAY"],
                1,
                cv2.LINE_AA,
            )

            # Display window
            cv2.imshow(window_name, frame)

            # Key and window controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                print("[INFO] 'Q' pressed. Exiting application...")
                break

            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                print("[INFO] Window closed by user. Exiting application...")
                break

    except KeyboardInterrupt:
        print("\n[INFO] Keyboard interrupt received (Ctrl+C). Exiting...")
    except Exception as e:
        print(f"\n[ERROR] An unexpected error occurred: {e}")
        traceback.print_exc()
    finally:
        print("[INFO] Cleaning up resources...")
        if cap is not None and cap.isOpened():
            cap.release()
            print("[INFO] Webcam released.")
        cv2.destroyAllWindows()
        print("[INFO] OpenCV windows closed.")
        if audio_alert_manager is not None:
            audio_alert_manager.cleanup()
            print("[INFO] Audio Alert Manager released.")
        if face_mesh is not None:
            face_mesh.close()
            print("[INFO] MediaPipe Face Mesh closed.")
        print("[INFO] Safe shutdown complete. Goodbye!\n")


if __name__ == "__main__":
    run_monitor()
