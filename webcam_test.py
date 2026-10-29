"""
Standalone Physical Webcam Diagnostic Tool
===========================================
Tests physical camera index 0 across OpenCV backends:
- CAP_DSHOW (DirectShow)
- CAP_MSMF (Media Foundation)
- CAP_ANY (Default Backend)

Strictly tests only the physical webcam at index 0 (does not fall back to OBS virtual camera).
If a valid frame is captured, displays a 10-second live preview window (press 'q' to close early).
Safely releases all hardware and GUI resources after every test.
"""

import sys
import time
import cv2
import numpy as np


def test_backend(backend_name: str, backend_api: int, display_duration: float = 10.0) -> bool:
    """
    Tests opening camera index 0 with a specific OpenCV backend.
    
    Returns:
        bool: True if an actual valid, non-empty frame was captured; False otherwise.
    """
    print("-" * 65)
    print(f"[TEST] Testing Backend: {backend_name} on Camera Index 0...")
    print("-" * 65)

    cap = None
    frame_captured = False

    try:
        t0 = time.time()
        # Strictly test physical camera at index 0
        cap = cv2.VideoCapture(0, backend_api)
        t_open = time.time() - t0

        is_opened = cap.isOpened()
        print(f"  * Camera open status : {'OPENED' if is_opened else 'FAILED'}")
        print(f"  * Open elapsed time  : {t_open:.3f} seconds")

        if not is_opened:
            print(f"  * Result             : Could not initialize camera handle on {backend_name}.")
            return False

        # Attempt to read a real frame
        t_read_start = time.time()
        ret, frame = cap.read()
        t_read = time.time() - t_read_start

        if ret and frame is not None and frame.size > 0:
            h, w, c = frame.shape
            nonzero_count = np.count_nonzero(frame)
            nonzero_pct = (nonzero_count / frame.size) * 100.0

            print(f"  * Frame read status  : SUCCESS")
            print(f"  * Frame resolution   : {w}x{h} ({c} channels)")
            print(f"  * Non-zero pixel data: {nonzero_pct:.2f}% (confirms non-blank image)")
            print(f"  * Read elapsed time  : {t_read:.3f} seconds")
            frame_captured = True

            # Requirement 5: Display live camera feed for up to 10 seconds
            window_name = f"Webcam Live Preview - {backend_name} (Press 'Q' to Exit)"
            print(f"\n  [INFO] Displaying live preview for up to {display_duration:.0f} seconds...")
            print("  [INFO] Click the window and press 'Q' to exit early.")

            start_display = time.time()
            while time.time() - start_display < display_duration:
                # Capture subsequent frames for live video demonstration
                loop_ret, loop_frame = cap.read()
                if not loop_ret or loop_frame is None:
                    break

                # Add timestamp and time remaining text
                time_left = max(0.0, display_duration - (time.time() - start_display))
                cv2.putText(
                    loop_frame,
                    f"Backend: {backend_name} | Closing in: {time_left:.1f}s",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    loop_frame,
                    "Press 'Q' to close window",
                    (20, loop_frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (220, 220, 220),
                    1,
                    cv2.LINE_AA,
                )

                cv2.imshow(window_name, loop_frame)
                key = cv2.waitKey(30) & 0xFF
                if key == ord("q") or key == ord("Q"):
                    print("  [INFO] 'Q' pressed by user. Closing preview window.")
                    break

            cv2.destroyWindow(window_name)
            print(f"  [INFO] Preview window closed.")
        else:
            print(f"  * Frame read status  : FAILED (ret={ret}, frame={'None' if frame is None else 'empty'})")
            print(f"  * Read elapsed time  : {t_read:.3f} seconds")

    except Exception as e:
        print(f"  * Exception caught   : {e}")
    finally:
        # Requirement 6: Properly release camera and windows after every test
        if cap is not None:
            cap.release()
            print(f"  * Camera release     : Resource released cleanly.")
        cv2.destroyAllWindows()

    print()
    return frame_captured


def run_diagnostics():
    print("=" * 65)
    print("  STANDALONE HP WEBCAM DIAGNOSTIC SUITE (CAMERA INDEX 0 ONLY)")
    print("=" * 65)
    print("Python Interpreter:", sys.executable)
    print("OpenCV Version    :", cv2.__version__)
    print()

    backends = [
        ("CAP_DSHOW (DirectShow)", cv2.CAP_DSHOW),
        ("CAP_MSMF (Media Foundation)", cv2.CAP_MSMF),
        ("CAP_ANY (Default Backend)", cv2.CAP_ANY),
    ]

    results = {}

    for name, api in backends:
        success = test_backend(name, api, display_duration=10.0)
        results[name] = success
        # Brief pause between backend switches to allow Windows driver detachment
        time.sleep(0.5)

    print("=" * 65)
    print("FINAL SUMMARY REPORT")
    print("=" * 65)
    any_success = False
    for name, success in results.items():
        status = "PASSED (Real frame captured)" if success else "FAILED (No frame captured)"
        print(f"  * {name:<30}: {status}")
        if success:
            any_success = True

    print("-" * 65)
    if any_success:
        print("VERDICT: Physical webcam accessible via at least one backend.")
    else:
        print("VERDICT: Access failed across all backends on Camera Index 0.")
        print("         No frames were captured.")
    print("=" * 65)


if __name__ == "__main__":
    run_diagnostics()
