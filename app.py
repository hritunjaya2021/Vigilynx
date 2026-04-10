# ============================================================
#  VIGILYNX — app.py  ← RUN THIS FILE
#  K.R Mangalam University | Projexa 26E2132
#  Detects: Phone | Head Turn | Crowd
#  Evidence images saved to vigilynx_evidence/
#  Dashboard: http://localhost:5000
#  Press Q on camera window to quit
# ============================================================

import subprocess, sys
subprocess.run([sys.executable, '-m', 'pip', 'install',
                'flask', 'pymongo', 'flask-cors'],
               capture_output=True)

import cv2
import numpy as np
import os, time, threading
import winsound
from datetime import datetime
from flask import Flask, jsonify, render_template, Response, make_response, request

from camera    import open_camera, read_frame, release_camera
from detection import (preprocess, make_subtractor, get_mask,
                       get_detections, load_haar,
                       PhoneDetector, HeadTurnDetector,
                       CrowdDetector, PersistenceTracker)
from database  import save_incident, get_recent_incidents, get_stats, update_status

# ── CONFIG ───────────────────────────────────────────────────
FRAME_W       = 640
FRAME_H       = 480
SAVE_EVIDENCE = True
EVIDENCE_DIR  = 'vigilynx_evidence'

COOLDOWN = {
    'phone'     : 20.0,
    'head_turn' :  4.0,
    'crowd'     :  8.0,
}

SOUNDS = {
    'phone'     : (1200, 400),
    'head_turn' : (900,  300),
    'crowd'     : (600,  450),
}

alert_log    = []
latest_frame = None
lock         = threading.Lock()

# ── SOUND ────────────────────────────────────────────────────
def play_sound(event_type):
    def _beep():
        try:
            freq, ms = SOUNDS.get(event_type, (1000, 300))
            winsound.Beep(freq, ms)
        except Exception:
            pass
    threading.Thread(target=_beep, daemon=True).start()

# ── FLASK ────────────────────────────────────────────────────
flask_app = Flask(__name__)

@flask_app.after_request
def add_cors(response):
    response.headers['Access-Control-Allow-Origin']  = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    response.headers['Access-Control-Allow-Methods'] = 'GET,POST,OPTIONS'
    return response

@flask_app.route('/')
def index():
    return render_template('dashboard.html')

@flask_app.route('/api/alerts')
def api_alerts():
    db_data = get_recent_incidents(100)
    if db_data:
        return jsonify(db_data)
    with lock:
        return jsonify(list(reversed(alert_log[-100:])))

@flask_app.route('/api/stats')
def api_stats():
    db_stats = get_stats()
    # Return db_stats whether from MongoDB or in-memory — get_stats() handles both
    with lock:
        mem_stats = {
            'total'      : len(alert_log),
            'phone'      : sum(1 for a in alert_log if a.get('type') == 'phone'),
            'head_turn'  : sum(1 for a in alert_log if a.get('type') == 'head_turn'),
            'crowd'      : sum(1 for a in alert_log if a.get('type') == 'crowd'),
            'unreviewed' : sum(1 for a in alert_log if a.get('status') == 'unreviewed'),
            'reviewed'   : sum(1 for a in alert_log if a.get('status') == 'reviewed'),
            'actioned'   : sum(1 for a in alert_log if a.get('status') == 'actioned'),
            'safe'       : sum(1 for a in alert_log if a.get('status') == 'safe'),
            'fine_imposed': sum(1 for a in alert_log if a.get('status') == 'fine_imposed'),
            'pending'    : sum(1 for a in alert_log if a.get('status') == 'pending'),
        }
    # Merge: prefer db_stats values, fill missing keys from mem_stats
    for k, v in mem_stats.items():
        if k not in db_stats:
            db_stats[k] = v
    # If db has 0 but memory has data, use memory counts
    if db_stats.get('total', 0) == 0 and mem_stats.get('total', 0) > 0:
        return jsonify(mem_stats)
    return jsonify(db_stats)

@flask_app.route('/api/update_status', methods=['POST', 'OPTIONS'])
def api_update_status():
    if request.method == 'OPTIONS':
        return '', 200
    data = request.get_json(silent=True)
    if not data:
        return jsonify({'success': False, 'error': 'No JSON body'}), 400
    incident_id = data.get('id')
    new_status  = data.get('status')
    if incident_id is None or not new_status:
        return jsonify({'success': False, 'error': 'Missing id or status'}), 400
    # Always update in-memory alert_log
    with lock:
        for a in alert_log:
            if str(a.get('id')) == str(incident_id):
                a['status'] = new_status
                break
    ok = update_status(incident_id, new_status)
    return jsonify({'success': ok})

@flask_app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@flask_app.route('/video_feed_frame')
def video_feed_frame():
    with lock:
        f = latest_frame
    if f is None:
        return '', 204
    ok, buf = cv2.imencode('.jpg', f, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        return '', 204
    resp = make_response(buf.tobytes())
    resp.headers['Content-Type']  = 'image/jpeg'
    resp.headers['Cache-Control'] = 'no-store'
    return resp

def gen_frames():
    while True:
        time.sleep(0.04)
        with lock:
            f = latest_frame
        if f is None:
            continue
        ok, buf = cv2.imencode('.jpg', f, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            continue
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
               + buf.tobytes() + b'\r\n')

def start_flask():
    flask_app.run(host='0.0.0.0', port=5000,
                  debug=False, use_reloader=False, threaded=True)

# ── ALERT MANAGER ────────────────────────────────────────────
class AlertManager:
    def __init__(self):
        self.last  = {}
        self.total = 0

    def trigger(self, etype, frame, bbox=None, conf=0.0):
        global alert_log
        now = time.time()
        if now - self.last.get(etype, 0) < COOLDOWN.get(etype, 5.0):
            return False
        self.last[etype] = now
        self.total += 1
        ts = datetime.now().strftime('%H:%M:%S')

        # Save evidence image
        img_path = None
        if SAVE_EVIDENCE and frame is not None:
            try:
                os.makedirs(EVIDENCE_DIR, exist_ok=True)
                fname    = f"{etype}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.total}.jpg"
                img_path = os.path.join(EVIDENCE_DIR, fname)
                cv2.imwrite(img_path, frame)
                print(f"  [EVIDENCE] Saved → {img_path}")
            except Exception as e:
                print(f"  [EVIDENCE] Failed: {e}")

        # Save to MongoDB
        save_incident(etype, bbox, conf, img_path)

        labels = {
            'phone'     : 'Phone Detected',
            'head_turn' : 'Head Turn',
            'crowd'     : 'Crowd Detected',
        }
        with lock:
            alert_log.append({
                'id'      : self.total,
                'ts'      : ts,
                'type'    : etype,
                'label'   : labels.get(etype, etype),
                'conf'    : round(conf, 2),
                'status'  : 'unreviewed',
                'img_path': img_path,
            })

        play_sound(etype)
        icons = {'phone':'[PHONE]','head_turn':'[HEAD TURN]','crowd':'[CROWD]'}
        print(f"  ALERT #{self.total} | {ts} | "
              f"{icons.get(etype,etype)} | conf={conf:.0%}")
        return True

# ── DRAWING ──────────────────────────────────────────────────
COLORS = {
    'phone'     : (0,   0, 255),
    'head_turn' : (0, 140, 255),
    'crowd'     : (200,  0, 200),
    'normal'    : (0,  200,  80),
}

flash_until=0; flash_msg=''; flash_col=(0,0,255)

def trigger_flash(msg, color, dur=2.5):
    global flash_until, flash_msg, flash_col
    flash_until=time.time()+dur; flash_msg=msg; flash_col=color

def draw_box(f, x, y, w, h, label, color):
    L = max(14, int(min(w,h)*0.25))
    for px,py,dx,dy in [(x,y,1,1),(x+w,y,-1,1),(x,y+h,1,-1),(x+w,y+h,-1,-1)]:
        cv2.line(f,(px,py),(px+dx*L,py),color,2)
        cv2.line(f,(px,py),(px,py+dy*L),color,2)
    (tw,_),_=cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,0.5,1)
    lx,ly=max(x,2),max(y-24,24)
    cv2.rectangle(f,(lx,ly-16),(lx+tw+10,ly+4),color,-1)
    cv2.putText(f,label,(lx+5,ly),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)

def draw_hud(f, fps, fno, amgr):
    H,W=f.shape[:2]
    ts=datetime.now().strftime('%Y-%m-%d  %H:%M:%S')
    ov=f.copy()
    cv2.rectangle(ov,(0,0),(W,34),(0,0,0),-1)
    cv2.addWeighted(ov,0.65,f,0.35,0,f)
    cv2.putText(f,'VIGILYNX  |  K.R MANGALAM UNIVERSITY',
                (10,22),cv2.FONT_HERSHEY_SIMPLEX,0.48,(0,210,100),1)
    cv2.putText(f,ts,(W-215,22),cv2.FONT_HERSHEY_SIMPLEX,0.44,(160,160,160),1)
    cv2.circle(f,(W-232,17),5,(0,0,255),-1)
    ov2=f.copy()
    cv2.rectangle(ov2,(0,H-28),(W,H),(0,0,0),-1)
    cv2.addWeighted(ov2,0.65,f,0.35,0,f)
    cv2.putText(f,f'FPS:{fps:.0f}  Alerts:{amgr.total}  |  Dashboard: localhost:5000',
                (10,H-9),cv2.FONT_HERSHEY_SIMPLEX,0.4,(110,110,110),1)
    if time.time()<flash_until:
        ov3=f.copy()
        cv2.rectangle(ov3,(0,34),(W,82),flash_col,-1)
        cv2.addWeighted(ov3,0.8,f,0.2,0,f)
        (tw,_),_=cv2.getTextSize(flash_msg,cv2.FONT_HERSHEY_DUPLEX,0.8,2)
        cv2.putText(f,flash_msg,((W-tw)//2,68),
                    cv2.FONT_HERSHEY_DUPLEX,0.8,(255,255,255),2)
    s=int((time.time()*60)%H)
    cv2.line(f,(0,s),(W,s),(30,80,120),1)
    cv2.rectangle(f,(0,0),(W-1,H-1),(30,70,100),1)

# ── MAIN LOOP ────────────────────────────────────────────────
def run_detection():
    global latest_frame

    print("="*58)
    print("  VIGILYNX — AI Campus Surveillance")
    print("  K.R Mangalam University | Projexa 26E2132")
    print("  Detects: Phone | Head Turn | Crowd")
    print(f"  Evidence → {EVIDENCE_DIR}/")
    print("  Dashboard: http://localhost:5000")
    print("  Press Q to quit")
    print("="*58)

    cap  = open_camera()
    if cap is None:
        return

    sub   = make_subtractor()
    haar  = load_haar()
    phone = PhoneDetector()
    hdet  = HeadTurnDetector(haar)
    cdet  = CrowdDetector()
    pt    = PersistenceTracker()
    amgr  = AlertManager()

    fno=fps=fcount=0; fps_t=time.time()
    print("\n[LIVE] Camera window opening...\n")

    while True:
        ret, raw = read_frame(cap)
        if not ret or raw is None:
            print("[ERROR] Lost camera."); break

        fno+=1; fcount+=1
        if time.time()-fps_t>=1.0:
            fps=fcount/(time.time()-fps_t)
            fps_t=time.time(); fcount=0

        frame,gray,blurred = preprocess(raw)
        mask = get_mask(sub,blurred)
        dets = get_detections(mask)
        pt.update(dets)

        # Phone
        for det,pc,should_alert in phone.check(dets,frame):
            x,y,w,h=det['bbox']
            draw_box(frame,x,y,w,h,f"PHONE {pc:.0%}",COLORS['phone'])
            if should_alert:
                if amgr.trigger('phone',frame,det['bbox'],pc):
                    trigger_flash('!! PHONE DETECTED !!',(160,0,0))

        # Head turn
        ist,tc,fb=hdet.check(gray)
        if fb:
            fx,fy,fw,fh=fb
            if ist:
                draw_box(frame,fx,fy,fw,fh,
                         f"HEAD TURN {tc:.0%}",COLORS['head_turn'])
                if amgr.trigger('head_turn',frame,fb,tc):
                    trigger_flash('!! HEAD TURN — EXAM VIOLATION !!',(0,100,200))
            else:
                cv2.rectangle(frame,(fx,fy),(fx+fw,fy+fh),COLORS['normal'],1)

        # Crowd
        isc,cc=cdet.check(dets)
        if isc:
            people=[d for d in dets if d['area']>1000]
            if people:
                xs=[d['bbox'][0] for d in people]
                ys=[d['bbox'][1] for d in people]
                x2=[d['bbox'][0]+d['bbox'][2] for d in people]
                y2=[d['bbox'][1]+d['bbox'][3] for d in people]
                cv2.rectangle(frame,(min(xs)-10,min(ys)-10),
                              (max(x2)+10,max(y2)+10),COLORS['crowd'],2)
            draw_box(frame,10,85,280,44,
                     f"CROWD {len(people)}+ PEOPLE {cc:.0%}",COLORS['crowd'])
            if amgr.trigger('crowd',frame,None,cc):
                trigger_flash('!! CROWD DETECTED !!',(150,0,150))

        draw_hud(frame,fps,fno,amgr)

        with lock:
            latest_frame=frame.copy()

        cv2.imshow('Vigilynx  [Q = quit]',frame)
        key=cv2.waitKey(1)&0xFF
        if key==ord('q') or key==27: break

    release_camera(cap)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    print(f"\n[DONE] {fno} frames | {amgr.total} alerts")
    print(f"Evidence images in: {os.path.abspath(EVIDENCE_DIR)}/")

if __name__ == '__main__':
    t = threading.Thread(target=start_flask, daemon=True)
    t.start()
    print("[flask] Dashboard → http://localhost:5000")
    run_detection()
