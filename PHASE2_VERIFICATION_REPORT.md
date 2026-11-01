# Phase 2 Verification Report: AI Smart Study Monitor

**Project Location:** `C:\Users\PARAS\Desktop\Face Detection\smart-study-monitor`  
**Date & Time:** November 1, 2026  
**Environment:** Windows 11 (64-bit AMD64), Python 3.11.9 (Desktop Installer), OpenCV 4.10.0.84 / 5.0.0, MediaPipe 0.10.14, Ultralytics YOLOv8n, PyTorch 2.15.0  

---

## 1. Executive Summary

Phase 2 of the **AI Smart Study Monitor** has undergone comprehensive code inspection, offline unit testing, integration pipeline verification, and physical webcam hardware validation.

- **Offline Detection Pipeline:** All 17 automated unit and integration tests passed (100% pass rate).
- **Physical Webcam Access:** Validated on Camera Index 0 (`HP True Vision FHD Camera`). Real video frames `(640x480, 100% non-zero pixels)` were captured successfully across **DirectShow (`CAP_DSHOW`)**, **Media Foundation (`CAP_MSMF`)**, and **Default (`CAP_ANY`)** backends.
- **Overall Verdict:** **READY FOR PHASE 3**.

---

## 2. Features Verified

| Objective / Feature | Status | Notes |
| :--- | :---: | :--- |
| **1. Face Detection (MediaPipe Face Mesh)** | **PASS** | 468/478 3D landmark mesh extracted, normalized, scaled, and rendered with green overlay points. |
| **2. Eye Monitoring & EAR Calculation** | **PASS** | Geometric EAR calculation verified using Soukupova & Cech (2016) formula with zero-division protection. |
| **3. Drowsiness Detection** | **PASS** | Monotonic timing verified: natural blinks (0.1–0.4s) ignored; closure >= 2.0s triggers `"Wake Up & Study!"` alert. |
| **4. Phone Detection (YOLOv8)** | **PASS** | Model loaded on CPU, filtered on COCO class ID 67 (`cell phone`), configurable confidence threshold (0.45). |
| **5. Multi-Module Integration** | **PASS** | Coordinated status priority hierarchy: Face Missing > Phone Detected > Drowsy > Focused Study. |
| **6. Webcam Frame Handling & Pipeline** | **PASS** | Horizontal feed mirroring, FPS smoothing with deque, HUD telemetry card rendering, and clean shutdown. |

---

## 3. Automated Tests Performed

The full test suite was executed via `pytest -v`:

```text
============================= test session starts =============================
platform win32 -- Python 3.11.9, pytest-9.2.0, pluggy-1.7.0
rootdir: C:\Users\PARAS\Desktop\Face Detection\smart-study-monitor
collected 17 items

tests/test_eye_monitor.py::test_calculate_single_ear_geometry PASSED     [  5%]
tests/test_eye_monitor.py::test_calculate_single_ear_zero_division_guard PASSED [ 11%]
tests/test_eye_monitor.py::test_eye_monitor_open_eyes PASSED             [ 17%]
tests/test_eye_monitor.py::test_eye_monitor_brief_blink_does_not_trigger_drowsy PASSED [ 23%]
tests/test_eye_monitor.py::test_eye_monitor_prolonged_closure_triggers_drowsy PASSED [ 29%]
tests/test_eye_monitor.py::test_eye_monitor_resets_when_face_is_lost PASSED [ 35%]
tests/test_eye_monitor.py::test_eye_monitor_handles_invalid_landmarks_safely PASSED [ 41%]
tests/test_integration.py::test_priority_face_missing_overrides_everything PASSED [ 47%]
tests/test_integration.py::test_priority_phone_overrides_drowsy PASSED   [ 52%]
tests/test_integration.py::test_priority_drowsy_overrides_normal PASSED  [ 58%]
tests/test_integration.py::test_priority_normal_study_status PASSED      [ 64%]
tests/test_integration.py::test_multi_module_pipeline_frame_handling PASSED [ 70%]
tests/test_phone_detector.py::test_phone_detector_initialization_with_real_model PASSED [ 76%]
tests/test_phone_detector.py::test_phone_detector_handles_missing_model_safely PASSED [ 82%]
tests/test_phone_detector.py::test_phone_detector_inference_on_blank_frame PASSED [ 88%]
tests/test_phone_detector.py::test_phone_detector_frame_interval_skipping PASSED [ 94%]
tests/test_phone_detector.py::test_phone_detector_parses_target_class_phone PASSED [100%]

============================= 17 passed in 5.44s ==============================
```

---

## 4. Physical Webcam Verification Results

Executed `webcam_test.py` targeting physical camera index 0 (`HP True Vision FHD Camera`):

| Backend | Open Status | Frame Read Status | Resolution | Non-Zero Pixels | Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **DirectShow (`CAP_DSHOW`)** | OPENED (0.48s) | SUCCESS (0.36s) | 640x480 | 100.00% | **PASS** |
| **Media Foundation (`CAP_MSMF`)** | OPENED (0.17s) | SUCCESS (0.45s) | 640x480 | 100.00% | **PASS** |
| **Default (`CAP_ANY`)** | OPENED (0.09s) | SUCCESS (0.42s) | 640x480 | 100.00% | **PASS** |

> Note: Testing strictly addressed hardware camera index 0 and did not rely on OBS virtual camera.

---

## 5. Model Initialization Results

| Model / Framework | Target Hardware | Weights File / Package | Initialization Verdict |
| :--- | :---: | :---: | :---: |
| **MediaPipe Face Mesh** | CPU (TFLite XNNPACK) | Built-in 478 landmark mesh | **PASS** |
| **Ultralytics YOLOv8n** | CPU (PyTorch) | `yolov8n.pt` (6.5 MB local weights) | **PASS** |
| **Eye Aspect Ratio Heuristic** | CPU (Monotonic clock) | Landmark indices 33, 133, 160, 144, 158, 153 (Left) & 362, 263, 385, 380, 387, 373 (Right) | **PASS** |

---

## 6. Bugs Discovered & Resolved

1. **Premature Camera Release in `main.py` (`finally` block bug):**
   - *Problem:* In `initialize_webcam()`, a `finally: if cap is not None: cap.release()` block was executing upon returning the open camera handle, closing the camera before `run_monitor()` could read frame 1. This caused `main.py` to immediately exit with `[WARN] Frame acquisition failed`.
   - *Resolution:* Removed the blanket `finally:` release. Only failed backends/indices are now released, leaving successful camera streams open.

2. **Microsoft Store Python AppContainer Sandbox Restriction:**
   - *Problem:* The original virtual environment was bound to Microsoft Store Python 3.11 (`PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0`). In Windows 11, Store AppContainers are blocked from DirectShow COM hardware capture devices (`backend is generally available but can't be used to capture by index`).
   - *Resolution:* Installed the standard desktop Python 3.11 (`C:\Users\PARAS\AppData\Local\Programs\Python\Python311\python.exe`) via `winget` and recreated the virtual environment. DirectShow and Media Foundation hardware access now work immediately.

3. **Pytest Namespace Collision with `webcam_test.py`:**
   - *Problem:* `test_backend()` in `webcam_test.py` matched Pytest's auto-discovery naming convention and caused fixture lookup errors when running `pytest`.
   - *Resolution:* Renamed function to `verify_backend()` and added optional CLI flags (`--quick`, `--no-preview`).

---

## 7. Files Modified

1. **`main.py`**: Corrected webcam initialization logic to prevent premature stream destruction.
2. **`webcam_test.py`**: Renamed test function, added headless GUI exception guards, and added CLI duration arguments.
3. **`tests/test_integration.py`**: Added multi-module pipeline coordination test verifying FaceMesh, EyeMonitor, and PhoneDetector running on synthetic frames.
4. **`venv/`**: Recreated clean virtual environment using the standard desktop Python 3.11 installer.

---

## 8. Remaining Issues

None. All Phase 2 requirements are functioning without errors.

---

## 9. Phase 3 Readiness Verdict

**Verdict: READY FOR PHASE 3 (PASS)**

The foundation for study analytics, audio alarms, session logging, and UI dashboard enhancements in Phase 3 is fully operational and verified.
