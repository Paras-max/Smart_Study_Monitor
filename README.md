# Smart Study Monitor - Phase 2: Attention & Distraction Tracking

An AI-powered computer vision system designed to keep students focused during study sessions. It operates locally on your laptop using your webcam, combining **MediaPipe Face Mesh** for facial presence and eye aspect ratio (EAR) drowsiness detection with **Ultralytics YOLOv8** for real-time smartphone distraction detection.

---

## 1. Phase 2 Features

1. **Facial Presence & Landmarks (from Phase 1):**
   - High-speed 468/478 3D facial landmark tracking via MediaPipe.
   - Immediate perimeter warning when the student steps away or obscures their face.
2. **Eye Monitoring & Drowsiness Alert:**
   - Real-time **Eye Aspect Ratio (EAR)** calculation for both eyes.
   - Differentiates natural blinks ($\approx 0.1 - 0.4\text{s}$) from fatigue.
   - Triggers **`WARNING: Wake Up & Study!`** when eyes remain closed for longer than $2.0\text{s}$ (configurable).
3. **Smartphone Distraction Detection:**
   - Detects cell phones (COCO Class 67) using **YOLOv8 Nano (`yolov8n.pt`)**.
   - Draws labeled bounding boxes in vibrant orange-red over detected phones.
   - Triggers **`WARNING: Put the Phone Away!`**.
   - Employs frame interval scheduling on CPU to preserve 25–30+ FPS without an external GPU.
4. **Hierarchical Alert Priority System:**
   - Prevents overlapping text and conflicting visual alerts:
     - **Priority 1 (Highest):** Face Missing (`WARNING: Face Not Detected!`)
     - **Priority 2:** Phone Distraction (`WARNING: Put the Phone Away!`)
     - **Priority 3:** Drowsiness (`WARNING: Wake Up & Study!`)
     - **Priority 4:** Focused & Studying (`STATUS: Focused & Studying`)
5. **Real-Time Heads-Up Display (HUD):**
   - Live Smoothed FPS counter (30-frame moving average).
   - Real-time attention metrics: Face Status, Eye Ratio (EAR), and Phone Status.

---

## 2. Project File Structure

```text
smart-study-monitor/
├── main.py                    # Main Phase 2 unified application
├── phase1_face_monitor.py     # Standalone Phase 1 baseline application
├── config.py                  # Central configuration parameters & thresholds
├── requirements.txt           # Pinned dependencies (OpenCV, MediaPipe, Ultralytics, PyTorch, PyTest)
├── run.bat                    # One-click Windows launch script (uses venv automatically)
├── README.md                  # Comprehensive setup and test documentation
├── yolov8n.pt                 # YOLOv8 nano pre-trained model weights (auto-downloaded)
├── modules/
│   ├── __init__.py
│   ├── eye_monitor.py         # EAR calculation & monotonic closure timer
│   └── phone_detector.py      # CPU-optimized YOLO cell-phone detector
└── tests/
    ├── __init__.py
    ├── test_eye_monitor.py    # Unit tests for EAR, blinks, and drowsiness duration
    ├── test_phone_detector.py # Unit tests for YOLO loading, inference, and box parsing
    └── test_integration.py    # Integration tests for status priority hierarchy
```

---

## 3. Quick Start (Windows 10 / 11)

### Option A: One-Click Launch (Recommended)
Simply double-click **`run.bat`** in the project folder.  
It automatically finds your virtual environment (`venv\Scripts\python.exe`) and launches `main.py`.

### Option B: Command Prompt / Terminal
```cmd
cd "C:\Users\PARAS\Desktop\Face Detection\smart-study-monitor"
.\venv\Scripts\python.exe main.py
```

> [!NOTE]
> If running in a new terminal where the virtual environment is not yet created, run:
> ```cmd
> py -3.11 -m venv venv
> .\venv\Scripts\pip install -r requirements.txt
> .\venv\Scripts\python.exe main.py
> ```

---

## 4. Running the Automated Test Suite

We provide a complete automated test suite covering unit calculations, edge cases, model inference, and priority logic:
```cmd
.\venv\Scripts\pytest -v tests
```
**Current Test Status:** `16 passed in ~7.4s` (100% pass rate).

---

## 5. Technical Principles Explained

### 1. Eye Aspect Ratio (EAR)
The Eye Aspect Ratio (Soukupova & Cech, 2016) measures the ratio of the vertical eye aperture to the horizontal eye width using 6 coordinates per eye:

$$\text{EAR} = \frac{\|p_2 - p_6\| + \|p_3 - p_5\|}{2 \times \|p_1 - p_4\|}$$

- **Horizontal Anchor:** $p_1$ (outer corner), $p_4$ (inner corner).
- **Vertical Pairs:** $(p_2, p_6)$ and $(p_3, p_5)$.
- **Combined EAR:** $\frac{\text{EAR}_{\text{left}} + \text{EAR}_{\text{right}}}{2}$

#### Why Monotonic Time is Used:
Human blinks are rapid ($0.1 - 0.4\text{s}$). If EAR drops below `0.22`, a closure timer is recorded using `time.monotonic()`. If the eyes reopen before $2.0\text{s}$, the timer immediately resets to `0.0s`. Only sustained closure beyond `config.EYE_MONITOR["DROWSINESS_DURATION_SEC"]` activates the drowsiness alert.

### 2. YOLOv8 Cell Phone Detection
- Model: `yolov8n.pt` (Nano variant, $\approx 6.2\text{ MB}$, fastest inference on mobile/laptop CPUs).
- Class Target: In the standard COCO dataset, **Class 67** corresponds to **cell phone**.
- Performance Scheduling: Running YOLO on every single frame at 720p can stress laptop CPUs. Our `PhoneDetector` runs inference every 4th frame (`INFERENCE_INTERVAL_FRAMES = 4`) and caches the bounding boxes. This keeps the camera feed silky smooth at 28–30 FPS while maintaining instant phone recognition.

---

## 6. How to Configure Thresholds (`config.py`)

All tuning parameters are isolated in [`config.py`](file:///c:/Users/PARAS/Desktop/Face%20Detection/smart-study-monitor/config.py):
- **EAR Sensitivity:** Change `EYE_MONITOR["EAR_THRESHOLD"]` (default `0.22`). Increase to `0.24` if your eyes are naturally narrow; decrease to `0.20` if small squints trigger alerts.
- **Drowsiness Duration:** Change `EYE_MONITOR["DROWSINESS_DURATION_SEC"]` (default `2.0` seconds).
- **Phone Confidence:** Change `PHONE_DETECTOR["CONFIDENCE_THRESHOLD"]` (default `0.45`). Increase if false phone detections occur from rectangular objects.
- **Inference Interval:** Change `PHONE_DETECTOR["INFERENCE_INTERVAL_FRAMES"]` (default `4`).

---

## 7. Hardware Testing Checklist

| # | Scenario | How to Perform | Expected Application Output |
| :--- | :--- | :--- | :--- |
| **1** | **Face Visible & Eyes Open** | Sit normally at your desk looking at the screen. | • Green points over face.<br>• Top Badge: `STATUS: Focused & Studying`.<br>• Metric shows: `Eyes: Open [EAR: ~0.28]`. |
| **2** | **Natural Blink** | Blink naturally while reading. | • EAR briefly dips below 0.22.<br>• `Wake Up & Study!` does NOT fire.<br>• Status stays green. |
| **3** | **Eyes Closed > 2.0s** | Close your eyes and count to 3. | • At 2.0s: Yellow perimeter border appears.<br>• Badge turns yellow: `WARNING: Wake Up & Study!`.<br>• Metric shows: `Closed (2.3s)`. Reopening instantly restores green. |
| **4** | **Face Missing** | Lean out of camera view or cover face. | • Priority 1 activates immediately.<br>• Thick red border + `WARNING: Face Not Detected!`.<br>• Eye timer resets so no false drowsiness occurs while away. |
| **5** | **Phone Visible in Hand** | Hold your phone up into the camera frame. | • Orange-red bounding box snaps to phone with confidence badge.<br>• Priority 2 triggers: `WARNING: Put the Phone Away!`. |
| **6** | **Phone Absent** | Put your phone in your pocket or off desk. | • Phone box vanishes.<br>• Metric reports: `Phone: No Phone Detected`. |
| **7** | **Phone Partly Hidden** | Hold phone so only half of it is visible. | • Phone is still detected if confidence $\ge 0.45$. |
| **8** | **Webcam Disconnected** | Unplug USB camera or close privacy cover before launch. | • Clean fallback from index 0 to 1, followed by diagnostic checklist without unhandled crash. |
| **9** | **Simultaneous Distractions** | Hold phone while closing your eyes. | • Priority 2 takes precedence: `WARNING: Put the Phone Away!` is displayed. |
| **10**| **Application Exit** | Press `Q` in the window or close with `X`. | • Immediate clean exit, hardware camera LED turns off, terminal logs `Safe shutdown complete`. |
