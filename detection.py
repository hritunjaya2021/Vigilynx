# ============================================================
#  detection.py  —  VIGILYNX
#  Detects: Phone | Head Turn | Crowd
# ============================================================

import cv2
import numpy as np
import os, glob, time

FRAME_W = 640
FRAME_H = 480


# ── HAAR CASCADE ─────────────────────────────────────────────
def load_haar():
    HAAR_NAME = 'haarcascade_frontalface_default.xml'

    # Method 1: find via cv2.__file__ — works on any install
    try:
        cv2_dir = os.path.dirname(cv2.__file__)
        for sub in ['data', os.path.join('..', 'data'), '.']:
            p = os.path.normpath(os.path.join(cv2_dir, sub, HAAR_NAME))
            if os.path.exists(p):
                c = cv2.CascadeClassifier(p)
                if not c.empty():
                    print(f"[OK] Haar loaded: {p}")
                    return c
    except Exception:
        pass

    # Method 2: cv2.data attribute (newer OpenCV builds)
    try:
        p = cv2.data.haarcascades + HAAR_NAME
        c = cv2.CascadeClassifier(p)
        if not c.empty():
            print(f"[OK] Haar loaded: {p}")
            return c
    except Exception:
        pass

    # Method 3: common Anaconda paths
    extras = [
        r'C:\Users\HP\anaconda3\envs\vigilynx\Library\etc\haarcascades\haarcascade_frontalface_default.xml',
        r'C:\Users\HP\anaconda3\Library\etc\haarcascades\haarcascade_frontalface_default.xml',
        r'C:\Users\HP\anaconda3\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml',
        r'C:\ProgramData\Anaconda3\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml',
        os.path.join(os.path.dirname(os.path.abspath(__file__)), HAAR_NAME),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', HAAR_NAME),
    ]
    for p in extras:
        if os.path.exists(p):
            c = cv2.CascadeClassifier(p)
            if not c.empty():
                print(f"[OK] Haar loaded: {p}")
                return c

    # Method 4: brute-force search entire C drive
    for f in glob.glob(r'C:\**\haarcascade_frontalface_default.xml', recursive=True):
        c = cv2.CascadeClassifier(f)
        if not c.empty():
            print(f"[OK] Haar loaded: {f}")
            return c

    print("[WARN] Haar not found — head turn disabled")
    return None


# ── BACKGROUND SUBTRACTOR ────────────────────────────────────
def make_subtractor():
    return cv2.createBackgroundSubtractorMOG2(
        history=500, varThreshold=50, detectShadows=False)


# ── PREPROCESSING ────────────────────────────────────────────
def preprocess(raw):
    frame   = cv2.resize(raw, (FRAME_W, FRAME_H))
    gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (21, 21), 0)
    return frame, gray, blurred


# ── FOREGROUND MASK ──────────────────────────────────────────
def get_mask(subtractor, blurred):
    fg = subtractor.apply(blurred)
    k  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k)
    return cv2.dilate(fg, k, iterations=2)


# ── CONTOUR DETECTION ────────────────────────────────────────
def get_detections(mask, min_area=100):
    cnts, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        a = cv2.contourArea(c)
        if a < min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        out.append({'bbox': (x, y, w, h), 'area': a, 'duration': 0})
    return out


# ── PHONE DETECTOR ───────────────────────────────────────────
class PhoneDetector:
    """
    LOOSE phone detector — detects ANY portrait-shaped rectangle
    that stays in frame for 5 frames. Low false-negative rate.

    Only 2 checks:
      1. Shape is taller than wide (portrait, ratio >= 1.4)
      2. Not body-sized (area between 500 and 40000 pixels)
      3. Present for 5 consecutive frames

    After alert fires → suppressed 20s for that location.
    """
    MIN_RATIO     = 1.4    # height/width — very loose
    MIN_AREA      = 500    # tiny objects ignored
    MAX_AREA      = 40000  # whole-body contours ignored
    MIN_FRAMES    = 5      # frames before alert (was 20, now 5)
    SUPPRESS_SECS = 20     # seconds before re-alert same spot

    def __init__(self):
        self.frame_count = {}
        self.alerted_at  = {}

    def check(self, dets, frame):
        """
        Returns list of (det, conf, should_alert)
          should_alert=True  → new alert, fire sound+log
          should_alert=False → already alerted recently, draw box only
        """
        results      = []
        current_keys = set()

        for det in dets:
            x, y, w, h = det['bbox']
            area = det['area']
            if not w or not h:
                continue

            # 1. Must be portrait-oriented (taller than wide)
            ratio = max(w, h) / max(min(w, h), 1)
            if w > h:
                continue          # landscape shape — skip
            if ratio < self.MIN_RATIO:
                continue

            # 2. Size filter
            if not (self.MIN_AREA < area < self.MAX_AREA):
                continue

            # 3. Not at very bottom of frame (floor/table noise)
            if (y + h) > FRAME_H * 0.97:
                continue

            # Passed — track consecutive frames
            key = (x // 50, y // 50)
            current_keys.add(key)
            self.frame_count[key] = self.frame_count.get(key, 0) + 1
            count = self.frame_count[key]

            conf = round(min(0.95, 0.55 + ratio * 0.06), 2)

            if count >= self.MIN_FRAMES:
                now  = time.time()
                last = self.alerted_at.get(key, 0)
                if now - last >= self.SUPPRESS_SECS:
                    self.alerted_at[key] = now
                    results.append((det, conf, True))   # fire alert
                else:
                    results.append((det, conf, False))  # draw only

        # Remove keys that disappeared from frame
        for k in list(self.frame_count):
            if k not in current_keys:
                del self.frame_count[k]

        return results


# ── HEAD TURN DETECTOR ───────────────────────────────────────
class HeadTurnDetector:
    SUPPRESS_SECS = 5.0   # seconds before a new head-turn alert can fire

    def __init__(self, haar):
        self.haar        = haar
        self.history     = []
        self.miss_streak = 0
        self.last_alert  = 0.0   # timestamp of last fired alert

    def check(self, gray):
        if self.haar is None or self.haar.empty():
            return False, 0, None

        faces = self.haar.detectMultiScale(
            gray, scaleFactor=1.05, minNeighbors=3,
            minSize=(30, 30), maxSize=(350, 350))

        now = time.time()

        if not len(faces):
            self.miss_streak += 1
            if self.miss_streak > 10:
                self.history = []
            return False, 0, None

        self.miss_streak = 0
        faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
        fx, fy, fw, fh = faces[0]
        cx = fx + fw // 2

        self.history.append((now, cx))
        # Keep only last 2 seconds of history (reduced from 3s to avoid stale span)
        self.history = [(t, x) for t, x in self.history if now - t < 2.0]

        # Still in suppression window → draw box only
        if now - self.last_alert < self.SUPPRESS_SECS:
            return False, 0, (fx, fy, fw, fh)

        if len(self.history) >= 4:
            xs       = [x for _, x in self.history]
            span     = max(xs) - min(xs)
            first_x  = xs[0]
            recent_x = float(np.mean(xs[-3:]))
            if span > 22 and abs(recent_x - first_x) > 12:
                conf = round(min(0.95, 0.55 + span / 150), 2)
                # Reset history + record alert time so we don't re-fire immediately
                self.last_alert = now
                self.history    = []
                return True, conf, (fx, fy, fw, fh)

        return False, 0, (fx, fy, fw, fh)


# ── CROWD DETECTOR ───────────────────────────────────────────
class CrowdDetector:
    def check(self, dets):
        people = [d for d in dets if d['area'] > 1000]
        if len(people) < 10:
            return False, 0
        centres = [
            (d['bbox'][0] + d['bbox'][2] // 2,
             d['bbox'][1] + d['bbox'][3] // 2)
            for d in people
        ]
        close = sum(
            1 for i in range(len(centres))
            for j in range(i+1, len(centres))
            if np.hypot(centres[i][0]-centres[j][0],
                        centres[i][1]-centres[j][1]) < 120
        )
        if close >= 8:
            conf = round(min(0.95, 0.5 + len(people) / 50), 2)
            return True, conf
        return False, 0


# ── SUSPICIOUS DETECTOR (kept for import compatibility) ───────
class SuspiciousDetector:
    def __init__(self):
        self.region_times = {}

    def update(self, dets):
        return []   # disabled


# ── VIOLENCE DETECTOR (kept for import compatibility) ─────────
class ViolenceDetector:
    def __init__(self):
        pass

    def check(self, dets):
        return False, 0


# ── PERSISTENCE TRACKER ──────────────────────────────────────
class PersistenceTracker:
    def __init__(self):
        self.d = {}

    def update(self, dets):
        now = time.time()
        cur = set()
        for det in dets:
            k = (det['bbox'][0] // 80, det['bbox'][1] // 80)
            cur.add(k)
            if k not in self.d:
                self.d[k] = now
            det['duration'] = round(now - self.d[k], 1)
        for k in list(self.d):
            if k not in cur:
                del self.d[k]
