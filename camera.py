# ============================================================
#  camera.py
#  Handles opening and reading from laptop webcam
#  Uses DirectShow backend — most reliable on Windows
# ============================================================

import cv2

FRAME_W = 640
FRAME_H = 480


def open_camera():
    print("[camera] Opening with DirectShow...")
    for idx in [0, 1, 2]:
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if not cap.isOpened():
            continue
        cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
        cap.set(cv2.CAP_PROP_FPS,          24)
        # Flush stale buffer frames
        for _ in range(15):
            cap.grab()
        ret, frame = cap.read()
        if ret and frame is not None:
            print(f"[OK] Camera {idx} ready — {frame.shape[1]}x{frame.shape[0]}")
            return cap
        cap.release()
    print("[ERROR] No camera found.")
    print("  Fix 1: Windows Settings > Privacy > Camera > turn ON")
    print("  Fix 2: Allow desktop apps to access camera > ON")
    print("  Fix 3: Close Teams / Zoom / Discord if open")
    print("  Fix 4: Restart Spyder and try again")
    return None


def read_frame(cap):
    """
    Reads latest frame using grab+retrieve.
    Faster than cap.read() and avoids buffer lag.
    """
    cap.grab()
    ret, frame = cap.retrieve()
    if not ret or frame is None:
        ret, frame = cap.read()   # fallback
    return ret, frame


def release_camera(cap):
    if cap is not None:
        cap.release()
    print("[camera] Released.")
