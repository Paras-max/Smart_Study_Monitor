"""
Smart Study Monitor - Phase 1: Camera & Face Monitoring
=========================================================
Author: AI Computer Vision Engineer & Senior Python Developer
Purpose: Real-time webcam face monitoring using OpenCV and MediaPipe Face Mesh.
         Alerts the student when their face is missing or undetected in the frame.

Controls:
    Press 'q' or 'Q' in the video window to quit cleanly.
"""

from collections import deque
import sys
import time
import traceback
import cv2
import mediapipe as mp


def draw_styled_banner(image, text, bg_color, text_color, y_pos=75, font_scale=0.75, thickness=2, sub_text=None):
    """
    Draws a semi-transparent colored card overlay behind the status text
    for high contrast and readability against any camera background.
    
    Args:
        image: Frame onto which the banner is drawn.
        text: Primary status string.
        bg_color: BGR tuple for overlay background.
        text_color: BGR tuple for text and border.
        y_pos: Vertical baseline position for primary text.
        font_scale: Font size for primary text.
        thickness: Font line thickness.
        sub_text: Optional secondary instruction displayed directly beneath.
    """
    font = cv2.FONT_HERSHEY_DUPLEX
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    sub_w, sub_h = 0, 0
    if sub_text:
        (sub_w, sub_h), _ = cv2.getTextSize(sub_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

    card_w = max(text_w, sub_w)
    padding_x = 16
    padding_y = 10
    
    x1 = 20
    y1 = y_pos - text_h - padding_y
    x2 = x1 + card_w + (padding_x * 2)
    y2 = y_pos + baseline + padding_y + (sub_h + 8 if sub_text else 0)
    
    # Semi-transparent background card
    overlay = image.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), bg_color, -1)
    cv2.addWeighted(overlay, 0.75, image, 0.25, 0, image)
    
    # Crisp outer border
    cv2.rectangle(image, (x1, y1), (x2, y2), text_color, 1)
    
    # Primary status text
    cv2.putText(
        image,
        text,
        (x1 + padding_x, y_pos),
        font,
        font_scale,
        text_color,
        thickness,
        cv2.LINE_AA,
    )
    
    # Optional secondary instruction
    if sub_text:
        cv2.putText(
            image,
            sub_text,
            (x1 + padding_x, y_pos + sub_h + 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )


def initialize_webcam(primary_index=0, fallback_index=1):
    """
    Safely connects to the laptop webcam. Tries the primary camera index first;
    if unavailable, releases the handle cleanly before testing the fallback index.
    
    Returns:
        tuple: (cv2.VideoCapture object or None, selected_camera_index or None)
    """
    print(f"[INFO] Attempting to open primary webcam (device index {primary_index})...")
    cap = cv2.VideoCapture(primary_index)
    
    if cap.isOpened():
        print(f"[SUCCESS] Connected to webcam at index {primary_index}.")
        return cap, primary_index
    
    # Explicitly release handle if primary failed before opening fallback
    print(f"[WARN] Webcam index {primary_index} unavailable. Releasing capture object...")
    cap.release()

    print(f"[INFO] Attempting to open fallback webcam (device index {fallback_index})...")
    cap = cv2.VideoCapture(fallback_index)
    if cap.isOpened():
        print(f"[SUCCESS] Connected to fallback webcam at index {fallback_index}.")
        return cap, fallback_index
    
    # Release fallback if it failed as well
    cap.release()
    return None, None


def run_face_monitor():
    """
    Main execution loop for Phase 1 Smart Study Monitor.
    - Handles webcam frame acquisition with mirroring.
    - Uses MediaPipe Face Mesh for 468/478 facial landmark detection.
    - Displays smoothed real-time FPS via a 30-frame sliding window.
    - Displays green status when face is present, red alerts when face is absent.
    - Enforces safe resource cleanup via try/finally.
    """
    print("=" * 65)
    print("   AI Smart Study Monitor - Phase 1: Camera & Face Monitoring")
    print("=" * 65)

    cap = None
    face_mesh = None
    window_name = "Smart Study Monitor - Phase 1"

    try:
        # Step 1: Initialize MediaPipe Face Mesh
        print("[INFO] Initializing MediaPipe Face Mesh detector...")
        mp_face_mesh = mp.solutions.face_mesh

        # static_image_mode=False: enables temporal landmark tracking between frames
        # max_num_faces=1: monitors primary student sitting at the desk
        # refine_landmarks=True: includes detailed eye contour and iris landmarks
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        print("[SUCCESS] MediaPipe Face Mesh initialized successfully.")

        # Step 2: Initialize Webcam
        cap, active_cam_idx = initialize_webcam(primary_index=0, fallback_index=1)
        if cap is None:
            print("\n" + "!" * 65)
            print("[ERROR] Could not open any local webcam (tried indices 0 and 1).")
            print("Troubleshooting checklist:")
            print(" 1. Check if the laptop webcam has a physical privacy slider closed.")
            print(" 2. Close any apps that lock the webcam (Zoom, Teams, Discord, Meet, Browsers).")
            print(" 3. Check Windows Camera Privacy Settings:")
            print("    Settings > Privacy & Security > Camera -> Turn ON 'Allow desktop apps'.")
            print(" 4. If using an external USB webcam, unplug and firmly reconnect it.")
            print("!" * 65 + "\n")
            return

        # Prefer 720p HD resolution for a balanced field of view and performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # Step 3: Performance & FPS setup (sliding window moving average)
        # We store timestamps of the last 30 frames to smooth out micro-jitter
        frame_timestamps = deque(maxlen=30)
        fps_display = 0.0

        print(f"[SUCCESS] Streaming from webcam index {active_cam_idx}.")
        print("[INFO] Video feed active. Press 'Q' or close window to exit.\n")

        # Step 4: Real-time processing loop
        while True:
            success, frame = cap.read()
            if not success or frame is None:
                print("\n[WARN] Failed to read frame from webcam. Device may have been disconnected.")
                break

            # Mirror the video feed horizontally for an intuitive webcam experience
            frame = cv2.flip(frame, 1)
            frame_height, frame_width = frame.shape[:2]

            # Calculate FPS using sliding window of recent frame timestamps
            current_time = time.time()
            frame_timestamps.append(current_time)
            if len(frame_timestamps) > 1:
                total_duration = frame_timestamps[-1] - frame_timestamps[0]
                if total_duration > 0:
                    fps_display = (len(frame_timestamps) - 1) / total_duration

            # Prepare frame for MediaPipe Face Mesh:
            # 1. OpenCV provides BGR images; MediaPipe requires RGB.
            # 2. Optimization: set writeable flag to False to pass by reference (zero copy).
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False

            # Run neural face mesh inference
            results = face_mesh.process(frame_rgb)

            # Re-enable writeable flag for OpenCV graphics rendering
            frame_rgb.flags.writeable = True

            # Step 5: Evaluate face detection status
            is_face_detected = (
                results.multi_face_landmarks is not None
                and len(results.multi_face_landmarks) > 0
            )

            if is_face_detected:
                # Iterate and draw small green points over all detected facial landmarks
                for face_landmarks in results.multi_face_landmarks:
                    for landmark in face_landmarks.landmark:
                        cx = int(landmark.x * frame_width)
                        cy = int(landmark.y * frame_height)
                        # Radius 1, filled green circle
                        cv2.circle(frame, (cx, cy), 1, (0, 255, 0), -1, cv2.LINE_AA)

                status_text = "STATUS: Face Detected"
                sub_instruction = None
                badge_bg = (20, 50, 20)        # Dark green background
                badge_color = (0, 255, 0)      # Bright green text & border
            else:
                # Warning: face is missing or undetected
                status_text = "WARNING: Face Not Detected!"
                sub_instruction = "Keep Your Face Visible"
                badge_bg = (20, 20, 60)        # Dark red background
                badge_color = (0, 0, 255)      # Bright red text & border

                # Draw prominent red warning perimeter border around the entire feed
                border_thickness = 6
                cv2.rectangle(
                    frame,
                    (0, 0),
                    (frame_width - 1, frame_height - 1),
                    (0, 0, 255),
                    border_thickness
                )

                # Center warning alert box
                center_alert = "! ATTENTION: FACE NOT DETECTED !"
                center_sub = "Please look at the camera and keep your face visible"
                
                alert_font = cv2.FONT_HERSHEY_DUPLEX
                (alert_w, alert_h), _ = cv2.getTextSize(center_alert, alert_font, 0.9, 2)
                (sub_w, sub_h), _ = cv2.getTextSize(center_sub, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)

                box_w = max(alert_w, sub_w) + 40
                box_h = alert_h + sub_h + 35
                cx1 = (frame_width - box_w) // 2
                cy1 = (frame_height - box_h) // 2
                cx2 = cx1 + box_w
                cy2 = cy1 + box_h

                # Dark red warning card overlay
                overlay = frame.copy()
                cv2.rectangle(overlay, (cx1, cy1), (cx2, cy2), (0, 0, 140), -1)
                cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
                cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), (255, 255, 255), 2)

                # Centered alert text
                text_x = cx1 + (box_w - alert_w) // 2
                text_y = cy1 + alert_h + 12
                cv2.putText(frame, center_alert, (text_x, text_y), alert_font, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
                
                sub_x = cx1 + (box_w - sub_w) // 2
                sub_y = text_y + sub_h + 12
                cv2.putText(frame, center_sub, (sub_x, sub_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)

            # Step 6: Render UI Overlays (Header Bar, Title, FPS, Status, Help)
            top_bar_height = 42
            header_overlay = frame.copy()
            cv2.rectangle(header_overlay, (0, 0), (frame_width, top_bar_height), (15, 15, 15), -1)
            cv2.addWeighted(header_overlay, 0.7, frame, 0.3, 0, frame)

            # Project title
            cv2.putText(
                frame,
                "Smart Study Monitor - Phase 1",
                (16, 28),
                cv2.FONT_HERSHEY_DUPLEX,
                0.65,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

            # Real-time smoothed FPS (Top Right)
            fps_str = f"FPS: {fps_display:4.1f}"
            (fps_w, _), _ = cv2.getTextSize(fps_str, cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)
            cv2.putText(
                frame,
                fps_str,
                (frame_width - fps_w - 20, 28),
                cv2.FONT_HERSHEY_DUPLEX,
                0.6,
                (0, 255, 255),
                1,
                cv2.LINE_AA
            )

            # Top Status Badge
            draw_styled_banner(
                frame,
                status_text,
                bg_color=badge_bg,
                text_color=badge_color,
                y_pos=80,
                font_scale=0.75,
                thickness=2,
                sub_text=sub_instruction
            )

            # Bottom Controls Hint
            cv2.putText(
                frame,
                "Press 'Q' to Exit",
                (20, frame_height - 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (200, 200, 200),
                1,
                cv2.LINE_AA
            )

            # Display the annotated frame in the OpenCV desktop window
            cv2.imshow(window_name, frame)

            # Step 7: Handle keyboard & window close events
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                print("[INFO] 'Q' pressed by user. Initiating clean exit...")
                break

            # Detect if user clicked the window 'X' close button
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                print("[INFO] Window closed by user. Initiating clean exit...")
                break

    except KeyboardInterrupt:
        print("\n[INFO] Application interrupted by user (Ctrl+C). Exiting...")
    except Exception as e:
        print(f"\n[ERROR] An unexpected runtime exception occurred: {e}")
        print("[DEBUG] Full traceback:")
        traceback.print_exc()
    finally:
        # Step 8: Reliable resource management in finally block
        print("[INFO] Starting resource cleanup...")
        
        if cap is not None and cap.isOpened():
            cap.release()
            print("[INFO] Webcam hardware released.")
        
        cv2.destroyAllWindows()
        print("[INFO] OpenCV desktop windows closed.")

        if face_mesh is not None:
            face_mesh.close()
            print("[INFO] MediaPipe Face Mesh model closed.")
            
        print("[INFO] Safe shutdown complete. Goodbye!\n")


if __name__ == "__main__":
    run_face_monitor()
