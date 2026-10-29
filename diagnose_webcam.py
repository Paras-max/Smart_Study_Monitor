"""
Comprehensive Webcam & Camera Diagnostic Utility
=================================================
Diagnoses Windows Camera privacy settings, PnP camera devices,
and tests OpenCV backends across indices 0 to 4.
"""

import os
import subprocess
import sys
import time
import winreg
import cv2
import numpy as np


def check_registry_camera_privacy():
    print("\n" + "=" * 65)
    print(" 1. WINDOWS CAMERA PRIVACY REGISTRY AUDIT")
    print("=" * 65)

    keys = [
        ("System-Wide Consent (HKLM)", winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows\CurrentVersion\CapabilityAccessManager\ConsentStore\webcam"),
        ("Current User Consent (HKCU)", winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\CapabilityAccessManager\ConsentStore\webcam"),
        ("Desktop Apps (NonPackaged HKCU)", winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\CapabilityAccessManager\ConsentStore\webcam\NonPackaged"),
    ]

    for label, hive, subkey in keys:
        try:
            with winreg.OpenKey(hive, subkey) as key:
                value, _ = winreg.QueryValueEx(key, "Value")
                print(f"  [PASS] {label}: {value}")
        except FileNotFoundError:
            print(f"  [INFO] {label}: Key or Value not explicitly set (defaults to Allow)")
        except PermissionError:
            print(f"  [WARN] {label}: Permission denied reading key")
        except Exception as e:
            print(f"  [WARN] {label}: {e}")


def check_pnp_devices():
    print("\n" + "=" * 65)
    print(" 2. WINDOWS HARDWARE & DEVICE MANAGER AUDIT")
    print("=" * 65)

    cmd = (
        'powershell -NoProfile -Command "'
        "Get-PnpDevice | Where-Object { $_.Class -in @('Camera', 'Image') -or $_.FriendlyName -match 'Camera|Webcam|Imaging' } "
        "| Select-Object Status, Class, FriendlyName, InstanceId | Format-Table -AutoSize"
        '"'
    )

    try:
        proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
        output = proc.stdout.strip()
        if output:
            print(output)
        else:
            print("  [WARN] No devices found under 'Camera' or 'Image' class.")
    except Exception as e:
        print(f"  [ERROR] PowerShell query failed: {e}")


def check_conflicting_processes():
    print("\n" + "=" * 65)
    print(" 3. ACTIVE POTENTIAL CAMERA-LOCKING PROCESSES")
    print("=" * 65)

    candidate_names = [
        "zoom", "teams", "discord", "skype", "obs64", "obs32",
        "chrome", "msedge", "firefox", "brave", "slack"
    ]

    cmd = 'powershell -NoProfile -Command "Get-Process | Select-Object -ExpandProperty ProcessName"'
    try:
        proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
        running = set(line.strip().lower() for line in proc.stdout.splitlines() if line.strip())
        found = []
        for name in candidate_names:
            if name in running:
                found.append(name)
        if found:
            print(f"  [WARN] Detected processes currently running that might be locking the webcam: {', '.join(found)}")
        else:
            print("  [PASS] No known video conference or recording apps actively running.")
    except Exception as e:
        print(f"  [ERROR] Process check failed: {e}")


def test_opencv_camera_indices():
    print("\n" + "=" * 65)
    print(" 4. OPENCV BACKEND & INDEX MATRIX TEST (0 to 4)")
    print("=" * 65)

    backends = [
        ("DEFAULT (CAP_ANY)", cv2.CAP_ANY),
        ("DIRECTSHOW (CAP_DSHOW)", cv2.CAP_DSHOW),
        ("MEDIA FOUNDATION (CAP_MSMF)", cv2.CAP_MSMF),
    ]

    working_combinations = []

    for idx in range(5):
        print(f"\n--- Testing Camera Index {idx} ---")
        for backend_name, backend_api in backends:
            sys.stdout.write(f"  Testing {backend_name} on index {idx}... ")
            sys.stdout.flush()

            t0 = time.time()
            cap = None
            try:
                cap = cv2.VideoCapture(idx, backend_api)
                t_open = time.time() - t0

                if not cap.isOpened():
                    print(f"FAILED (could not open in {t_open:.2f}s)")
                    continue

                # Camera opened! Now test actual frame capture
                sys.stdout.write(f"Opened in {t_open:.2f}s. Reading frame... ")
                sys.stdout.flush()

                t_read_start = time.time()
                ret = False
                frame = None
                for _ in range(3):
                    ret, frame = cap.read()
                    if ret and frame is not None and frame.size > 0:
                        break
                    time.sleep(0.05)

                t_read = time.time() - t_read_start

                if ret and frame is not None and frame.size > 0:
                    h, w = frame.shape[:2]
                    nonzero = np.count_nonzero(frame)
                    total_pixels = frame.size
                    pct_nonzero = (nonzero / total_pixels) * 100.0

                    print(f"SUCCESS! Captured {w}x{h} frame ({pct_nonzero:.1f}% non-zero) in {t_read:.2f}s")
                    working_combinations.append({
                        "index": idx,
                        "backend_name": backend_name,
                        "backend_api": backend_api,
                        "width": w,
                        "height": h,
                    })
                else:
                    print(f"FAILED to read valid frame (ret={ret}, elapsed={t_read:.2f}s)")

            except Exception as e:
                print(f"EXCEPTION: {e}")
            finally:
                if cap is not None:
                    cap.release()

    print("\n" + "=" * 65)
    print(" 5. DIAGNOSTIC SUMMARY")
    print("=" * 65)
    if working_combinations:
        print(f"  [SUCCESS] Found {len(working_combinations)} working camera configuration(s):")
        for wc in working_combinations:
            print(f"    - Index {wc['index']} via {wc['backend_name']} ({wc['width']}x{wc['height']})")
    else:
        print("  [FAILURE] Zero working camera configurations found among indices 0 to 4.")


if __name__ == "__main__":
    check_registry_camera_privacy()
    check_pnp_devices()
    check_conflicting_processes()
    test_opencv_camera_indices()
