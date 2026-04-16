# ============================================================
#  VIGILYNX — detection.py
#  Detects: Phone | Head Turn | Crowd
# ============================================================

import cv2
import numpy as np
import os, glob, time

FRAME_W = 640
FRAME_H = 480


def load_haar():
    try:
        p = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        c = cv2.CascadeClassifier(p)
        if not c.empty():
            print("[OK] Haar loaded from cv2.data")
            return c
    except Exception:
        pass
    extras = [
        r'C:\Users\HP\anaconda3\envs\vigilynx\Library\etc\haarcascades\haarcascade_frontalface_default.xml',
        r'C:\Users\HP\anaconda3\Library\etc\haarcascades\haarcascade_frontalface_default.xml',
        r'C:\Users\HP\anaconda3\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml',
        r'C:\ProgramData\Anaconda3\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml',
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     'models', 'haarcascade_frontalface_default.xml'),
    ]
    for p in extras:
        if os.path.exists(p):
            c = cv2.CascadeClassifier(p)
            if not c.empty():
                print("[OK] Haar loaded")
                return c
    for f in glob.glob(r'C:\**\haarcascade_frontalface_default.xml', recursive=True):
        c = cv2.CascadeClassifier(f)
        if not c.empty():
            return c
    print("[WARN] Haar not found — head turn disabled")
    return None


def make_subtractor():
    return cv2.createBackgroundSubtractorMOG2(
        history=500, varThreshold=50, detectShadows=False)


def preprocess(raw):
    frame   = cv2.resize(raw, (FRAME_W, FRAME_H))
    gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (21, 21), 0)
    return frame, gray, blurred


def get_mask(subtractor, blurred):
    fg = subtractor.apply(blurred)
    k  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k)
    return cv2.dilate(fg, k, iterations=2)


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
    Loose phone detector.
    Checks: portrait rectangle shape + reasonable size.
    Fires after 5 steady frames. Suppresses same spot 20s.
    Returns (det, conf, should_alert).
    """
    MIN_RATIO     = 1.4
    MIN_AREA      = 500
    MAX_AREA      = 40000
    MIN_FRAMES    = 5
    SUPPRESS_SECS = 20

    def __init__(self):
        self.frame_count = {}
        self.alerted_at  = {}

    def check(self, dets, frame):
        results      = []
        current_keys = set()

        for det in dets:
            x, y, w, h = det['bbox']
            area = det['area']
            if not w or not h:
                continue
            if w > h:
                continue
            ratio = max(w, h) / max(min(w, h), 1)
            if ratio < self.MIN_RATIO:
                continue
            if not (self.MIN_AREA < area < self.MAX_AREA):
                continue
            if (y + h) > FRAME_H * 0.97:
                continue

            key = (x // 50, y // 50)
            current_keys.add(key)
            self.frame_count[key] = self.frame_count.get(key, 0) + 1
            count = self.frame_count[key]
            conf  = round(min(0.95, 0.55 + ratio * 0.06), 2)

            if count >= self.MIN_FRAMES:
                now  = time.time()
                last = self.alerted_at.get(key, 0)
                if now - last >= self.SUPPRESS_SECS:
                    self.alerted_at[key] = now
                    results.append((det, conf, True))
                else:
                    results.append((det, conf, False))

        for k in list(self.frame_count):
            if k not in current_keys:
                del self.frame_count[k]

        return results


# ── HEAD TURN DETECTOR ───────────────────────────────────────
class HeadTurnDetector:
    """
    Tracks face X-position over 2s.
    Fires on sustained lateral movement (not brief twitches).
    """
    def __init__(self, haar):
        self.haar    = haar
        self.history = []

    def check(self, gray):
        if self.haar is None or self.haar.empty():
            return False, 0, None

        faces = self.haar.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5,
            minSize=(40, 40), maxSize=(300, 300))

        if not len(faces):
            self.history = []
            return False, 0, None

        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        fx, fy, fw, fh = faces[0]
        cx  = fx + fw // 2
        now = time.time()

        self.history.append((now, cx))
        self.history = [(t, x) for t, x in self.history if now - t < 2.0]

        if len(self.history) >= 6:
            xs       = [x for _, x in self.history]
            span     = max(xs) - min(xs)
            first_x  = xs[0]
            recent_x = float(np.mean(xs[-3:]))
            if span > 35 and abs(recent_x - first_x) > 20:
                conf = round(min(0.95, 0.55 + span / 150), 2)
                return True, conf, (fx, fy, fw, fh)

        return False, 0, (fx, fy, fw, fh)


# ── CROWD DETECTOR ───────────────────────────────────────────
class CrowdDetector:
    """10+ person-sized contours with 8+ close pairs."""
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
            for j in range(i + 1, len(centres))
            if np.hypot(centres[i][0] - centres[j][0],
                        centres[i][1] - centres[j][1]) < 120
        )
        if close >= 8:
            conf = round(min(0.95, 0.5 + len(people) / 50), 2)
            return True, conf
        return False, 0


# ── STUBS (kept for import compatibility) ────────────────────
class ViolenceDetector:
    def __init__(self): pass
    def check(self, dets): return False, 0

class SuspiciousDetector:
    def __init__(self): self.region_times = {}
    def update(self, dets): return []


# ── PERSISTENCE TRACKER ──────────────────────────────────────
class PersistenceTracker:
    def __init__(self): self.d = {}

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
