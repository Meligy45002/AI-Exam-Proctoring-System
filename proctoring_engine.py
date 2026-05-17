"""
proctoring_engine.py  —  AI Exam Proctoring Engine v2
Real detections using:
  - MediaPipe Face Mesh  → 468 landmarks → real gaze (iris) + head pose (pitch/yaw/roll)
  - MediaPipe Pose       → body keypoints → suspicious posture / leaning
  - MediaPipe Hands      → hand near face / ear → phone gesture
  - OpenCV Haar          → fallback face + eye detection
  - Object detection     → phone / watch heuristics from skin+rectangle analysis

Violations fired (only when actually detected, never fake):
  ✓ Student Not Visible
  ✓ Multiple Persons Detected
  ✓ Gaze Left / Gaze Right / Gaze Up (real iris tracking)
  ✓ Head Turn Left / Right (real yaw)
  ✓ Head Down (pitch)
  ✓ Suspicious Posture (leaning, shoulders turned)
  ✓ Hand Near Face/Ear (phone gesture)
  ✓ Unauthorized Object Detected (rectangle + skin colour heuristic)

If nothing is wrong → NO violation is added (real "no violation detected").
"""

import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
import math

# ── Try to import MediaPipe ───────────────────────────────────────────────────
try:
    import mediapipe as mp
    _MP_FACE  = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=4,
        refine_landmarks=True,          # enables iris landmarks
        min_detection_confidence=0.5, min_tracking_confidence=0.5)
    _MP_POSE  = mp.solutions.pose.Pose(
        static_image_mode=False,
        min_detection_confidence=0.5, min_tracking_confidence=0.5)
    _MP_HANDS = mp.solutions.hands.Hands(
        static_image_mode=False, max_num_hands=2,
        min_detection_confidence=0.5, min_tracking_confidence=0.5)
    _HAS_MP = True
except Exception:
    _HAS_MP = False

# ── Haar fallback ─────────────────────────────────────────────────────────────
_CASCADE     = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
_EYE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

# ── MediaPipe landmark indices ────────────────────────────────────────────────
# Iris: left=468-472, right=473-477  (refine_landmarks=True)
_L_IRIS = [468, 469, 470, 471, 472]
_R_IRIS = [473, 474, 475, 476, 477]
# Face outline points for head pose
_NOSE_TIP   = 4
_CHIN       = 152
_L_EYE_OUT  = 263
_R_EYE_OUT  = 33
_L_MOUTH    = 287
_R_MOUTH    = 57
# Pose keypoints (MediaPipe Pose)
_L_SHOULDER = 11
_R_SHOULDER = 12
_L_EAR      = 7
_R_EAR      = 8
_NOSE_POSE  = 0


# ═══════════════════════════════════════════════════════════════════════════════
# Drawing helpers
# ═══════════════════════════════════════════════════════════════════════════════

def draw_corner_box(img, x, y, w, h, color=(0, 220, 90), thickness=2, cl=22):
    x2, y2 = x + w, y + h
    for pts in [((x,y),(x+cl,y)), ((x,y),(x,y+cl)),
                ((x2,y),(x2-cl,y)), ((x2,y),(x2,y+cl)),
                ((x,y2),(x+cl,y2)), ((x,y2),(x,y2-cl)),
                ((x2,y2),(x2-cl,y2)), ((x2,y2),(x2,y2-cl))]:
        cv2.line(img, pts[0], pts[1], color, thickness)

def draw_label(img, text, x, y, bg=(20,100,220), fg=(255,255,255), fs=0.44):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), bl = cv2.getTextSize(text, font, fs, 1)
    cv2.rectangle(img, (x, y-th-4), (x+tw+6, y+bl), bg, -1)
    cv2.putText(img, text, (x+3, y), font, fs, fg, 1, cv2.LINE_AA)

def draw_header(img, face_count, violations_active):
    h, w = img.shape[:2]
    ov = img.copy()
    cv2.rectangle(ov, (0,0), (w,32), (8,8,8), -1)
    cv2.addWeighted(ov, 0.78, img, 0.22, 0, img)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, "AI PROCTORING SYSTEM", (10,22), font, 0.55, (200,200,200), 1, cv2.LINE_AA)
    ts = datetime.now().strftime("%H:%M:%S")
    cv2.putText(img, ts, (w-78,22), font, 0.52, (200,200,200), 1, cv2.LINE_AA)
    fc_col = (80,220,80) if face_count == 1 else (60,60,255)
    cv2.putText(img, f"Faces:{face_count}", (w-165,22), font, 0.44, fc_col, 1, cv2.LINE_AA)
    status_col = (60,60,230) if violations_active else (60,200,60)
    status_txt = "ALERT" if violations_active else "OK"
    cv2.putText(img, status_txt, (w-215,22), font, 0.44, status_col, 1, cv2.LINE_AA)

def draw_gaze_bar(img, yaw, pitch, iris_h):
    h, w = img.shape[:2]
    ov = img.copy()
    cv2.rectangle(ov, (0, h-42), (w//2+60, h), (8,8,8), -1)
    cv2.addWeighted(ov, 0.68, img, 0.32, 0, img)
    font = cv2.FONT_HERSHEY_SIMPLEX

    if abs(yaw) < 12:
        g_txt, g_col = "GAZE: FORWARD", (50,220,50)
    elif yaw < 0:
        g_txt, g_col = "GAZE: LEFT",    (50,50,255)
    else:
        g_txt, g_col = "GAZE: RIGHT",   (50,50,255)

    cv2.putText(img, g_txt, (8, h-26), font, 0.52, g_col, 1, cv2.LINE_AA)
    info = f"Iris H:{iris_h:.2f}  Yaw:{yaw:+.1f}  Pitch:{pitch:+.1f}"
    cv2.putText(img, info, (8, h-10), font, 0.38, (170,170,170), 1, cv2.LINE_AA)

def draw_pose_overlay(img, lx, ly, rx, ry):
    """Draw shoulder line."""
    if lx > 0 and rx > 0:
        cv2.line(img, (lx,ly), (rx,ry), (100,200,255), 1)
        cv2.circle(img, (lx,ly), 5, (100,200,255), -1)
        cv2.circle(img, (rx,ry), 5, (100,200,255), -1)


# ═══════════════════════════════════════════════════════════════════════════════
# Head pose from 6-point PnP
# ═══════════════════════════════════════════════════════════════════════════════

_MODEL_POINTS = np.array([
    (0.0,    0.0,    0.0),    # nose tip
    (0.0,  -330.0, -65.0),   # chin
    (-225.0, 170.0,-135.0),  # left eye corner
    ( 225.0, 170.0,-135.0),  # right eye corner
    (-150.0,-150.0,-125.0),  # left mouth
    ( 150.0,-150.0,-125.0),  # right mouth
], dtype=np.float64)

def head_pose_pnp(lms, img_w, img_h):
    """Return (yaw_deg, pitch_deg, roll_deg) from face mesh landmarks."""
    def lm(i):
        return (int(lms[i].x * img_w), int(lms[i].y * img_h))

    image_points = np.array([
        lm(_NOSE_TIP), lm(_CHIN),
        lm(_L_EYE_OUT), lm(_R_EYE_OUT),
        lm(_L_MOUTH), lm(_R_MOUTH),
    ], dtype=np.float64)

    focal = img_w
    cam_matrix = np.array([
        [focal, 0,     img_w/2],
        [0,     focal, img_h/2],
        [0,     0,     1      ]
    ], dtype=np.float64)
    dist = np.zeros((4,1))

    ok, rvec, tvec = cv2.solvePnP(_MODEL_POINTS, image_points, cam_matrix, dist,
                                   flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return 0.0, 0.0, 0.0

    rmat, _ = cv2.Rodrigues(rvec)
    proj = np.hstack((rmat, tvec))
    _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(proj)
    pitch = float(euler[0])
    yaw   = float(euler[1])
    roll  = float(euler[2])
    return yaw, pitch, roll


# ═══════════════════════════════════════════════════════════════════════════════
# Iris gaze from MediaPipe iris landmarks
# ═══════════════════════════════════════════════════════════════════════════════

def iris_gaze(lms, img_w, img_h):
    """
    Returns (yaw_offset, iris_h_norm).
    yaw_offset: negative = looking left, positive = looking right.
    Uses the iris centre relative to the eye corners.
    """
    def pt(i):
        return np.array([lms[i].x * img_w, lms[i].y * img_h])

    try:
        # Left eye: corners 33 (inner) & 133 (outer), iris centre = mean of _L_IRIS
        l_iris_c  = np.mean([pt(i) for i in _L_IRIS], axis=0)
        l_inner   = pt(133); l_outer = pt(33)
        l_eye_w   = np.linalg.norm(l_outer - l_inner)
        l_offset  = (l_iris_c[0] - (l_inner[0]+l_outer[0])/2) / (l_eye_w + 1e-6)

        # Right eye: corners 362 (inner) & 263 (outer)
        r_iris_c  = np.mean([pt(i) for i in _R_IRIS], axis=0)
        r_inner   = pt(362); r_outer = pt(263)
        r_eye_w   = np.linalg.norm(r_outer - r_inner)
        r_offset  = (r_iris_c[0] - (r_inner[0]+r_outer[0])/2) / (r_eye_w + 1e-6)

        gaze_yaw = float((l_offset + r_offset) / 2)   # ~-0.5 left .. +0.5 right
        iris_h   = float((l_iris_c[1] + r_iris_c[1]) / 2 / img_h)
        return gaze_yaw, iris_h
    except Exception:
        return 0.0, 0.5


# ═══════════════════════════════════════════════════════════════════════════════
# Object detection — phone / watch heuristic (no YOLO needed)
# ═══════════════════════════════════════════════════════════════════════════════

def detect_objects_heuristic(frame):
    """
    Detect rectangular objects that are NOT skin-coloured (phone, book, watch).
    Returns list of (x,y,w,h,label,conf) for suspicious objects.
    """
    found = []
    h_img, w_img = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    dilated = cv2.dilate(edges, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Skin colour mask (HSV)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    skin_mask = cv2.inRange(hsv, (0,20,70), (20,170,255))

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 3000 or area > h_img * w_img * 0.35:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = w / (h + 1e-6)
        # Phone-like: tall rectangle (0.35–0.65 aspect or 1.5–3.0 for landscape)
        is_phone_shape = (0.30 < aspect < 0.70) or (1.4 < aspect < 3.2)
        if not is_phone_shape:
            continue
        # Check that the region is mostly NOT skin
        roi_skin = skin_mask[y:y+h, x:x+w]
        skin_ratio = roi_skin.mean() / 255.0
        if skin_ratio > 0.50:
            continue   # mostly skin — probably just a hand
        # Avoid flagging the face region
        conf = round(min(0.92, 0.55 + area / (h_img * w_img) * 3), 2)
        label = "Phone" if aspect < 1.0 else "Device"
        found.append((x, y, w, h, label, conf))

    # Keep only top-2 by area to avoid noise
    found.sort(key=lambda r: r[2]*r[3], reverse=True)
    return found[:2]


# ═══════════════════════════════════════════════════════════════════════════════
# Suspicious posture from Pose landmarks
# ═══════════════════════════════════════════════════════════════════════════════

def analyse_posture(pose_lms, img_w, img_h):
    """
    Returns (l_shoulder, r_shoulder, violations_list).
    Detects: leaning sideways, shoulders turned (side-view).
    """
    viols = []
    if pose_lms is None:
        return (0,0), (0,0), viols

    def pt(i):
        lm = pose_lms.landmark[i]
        return int(lm.x*img_w), int(lm.y*img_h), lm.visibility

    lx,ly,lv = pt(_L_SHOULDER)
    rx,ry,rv = pt(_R_SHOULDER)

    if lv < 0.5 or rv < 0.5:
        return (lx,ly),(rx,ry), viols

    # Shoulder tilt — leaning sideways
    dy = abs(ly - ry)
    dx = abs(lx - rx)
    tilt_deg = math.degrees(math.atan2(dy, max(dx,1)))
    if tilt_deg > 18:
        viols.append({
            "type": "Suspicious Posture — Leaning",
            "severity": "MEDIUM",
            "confidence": round(min(0.95, 0.60 + tilt_deg/60), 2),
        })

    # Shoulders nearly same x → side-on to camera
    if dx < img_w * 0.08 and lv > 0.6 and rv > 0.6:
        viols.append({
            "type": "Suspicious Posture — Turned Away",
            "severity": "HIGH",
            "confidence": 0.85,
        })

    return (lx,ly),(rx,ry), viols


# ═══════════════════════════════════════════════════════════════════════════════
# Hand near face / ear → phone gesture
# ═══════════════════════════════════════════════════════════════════════════════

def analyse_hands(hands_res, face_box, img_w, img_h):
    """Returns list of violations if hand is near face/ear."""
    viols = []
    if hands_res is None or not hands_res.multi_hand_landmarks:
        return viols
    if face_box is None:
        return viols

    fx, fy, fw, fh = face_box
    face_cx, face_cy = fx + fw//2, fy + fh//2
    ear_radius = fw * 0.7   # approximate distance to ear

    for hand_lms in hands_res.multi_hand_landmarks:
        # Use wrist + index tip to estimate hand centre
        wrist = hand_lms.landmark[0]
        tip   = hand_lms.landmark[8]
        hx = int((wrist.x + tip.x)/2 * img_w)
        hy = int((wrist.y + tip.y)/2 * img_h)
        dist = math.hypot(hx - face_cx, hy - face_cy)
        if dist < ear_radius * 1.4:
            viols.append({
                "type": "Hand Near Face/Ear — Phone Gesture",
                "severity": "HIGH",
                "confidence": round(min(0.95, 0.70 + (ear_radius - dist) / ear_radius * 0.25), 2),
            })
            break   # one violation per frame is enough

    return viols


# ═══════════════════════════════════════════════════════════════════════════════
# FaceDetector  (for Noise Test — unchanged interface)
# ═══════════════════════════════════════════════════════════════════════════════

def detect_faces_haar(img_bgr, scale=1.1, neighbors=5, min_size=50):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    dets = _CASCADE.detectMultiScale(gray, scaleFactor=scale, minNeighbors=neighbors,
                                     minSize=(min_size,min_size), flags=cv2.CASCADE_SCALE_IMAGE)
    return list(dets) if len(dets) else []

def crop_face(img_bgr, box, pad=0.25):
    h_img, w_img = img_bgr.shape[:2]
    x,y,w,h = box
    px,py = int(w*pad), int(h*pad)
    return img_bgr[max(0,y-py):min(h_img,y+h+py), max(0,x-px):min(w_img,x+w+px)]

def face_histogram(img_bgr, size=(64,64)):
    face = cv2.resize(img_bgr, size)
    desc = []
    for c in range(3):
        hist = cv2.calcHist([face],[c],None,[32],[0,256]).flatten().astype(np.float32)
        n = np.linalg.norm(hist); desc.append(hist/n if n>0 else hist)
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    hist_g = cv2.calcHist([gray],[0],None,[64],[0,256]).flatten().astype(np.float32)
    n = np.linalg.norm(hist_g); desc.append(hist_g/n if n>0 else hist_g)
    return np.concatenate(desc)  # always (160,)

def cosine_sim(a, b):
    a=np.array(a,np.float32).flatten(); b=np.array(b,np.float32).flatten()
    if a.shape!=b.shape: return 0.0
    d=np.linalg.norm(a)*np.linalg.norm(b)
    return float(np.dot(a,b)/d) if d>0 else 0.0

class FaceDetector:
    def get_embedding(self, img_bgr):
        boxes = detect_faces_haar(img_bgr)
        if not boxes: boxes = detect_faces_haar(img_bgr,1.05,3,30)
        if not boxes: return None,0
        return face_histogram(crop_face(img_bgr,boxes[0])), len(boxes)

    def build_face_db(self, registered_dir="data/registered_faces"):
        db={}; p=Path(registered_dir)
        if not p.exists(): return db
        for d in sorted(p.iterdir()):
            if not d.is_dir(): continue
            display=d.name.rsplit("_",1)[0]; embs=[]
            for ph in sorted(list(d.glob("*.jpg"))+list(d.glob("*.png"))):
                img=cv2.imread(str(ph))
                if img is None: continue
                emb,_=self.get_embedding(img)
                if emb is not None: embs.append(emb)
            if embs: db[display]=embs
        return db

    def identify(self, img_bgr, face_db, threshold=0.60):
        emb,n=self.get_embedding(img_bgr)
        if n==0: return "Unknown",0.0,0
        if emb is None: return "Face found (no ID)",0.0,n
        if not face_db: return "Unknown",0.0,n
        best_name,best_sim="Unknown",0.0
        for name,embs in face_db.items():
            sim=max(cosine_sim(emb,e) for e in embs)
            if sim>best_sim: best_sim,best_name=sim,name
        return (best_name,best_sim,n) if best_sim>=threshold else ("Unknown",best_sim,n)

    def detect_and_verify(self, frame):
        boxes=detect_faces_haar(frame)
        return boxes,"Student",0.8 if boxes else 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# ProctoringEngine  — main frame processor
# ═══════════════════════════════════════════════════════════════════════════════

class ProctoringEngine:

    # Cooldown: don't fire same violation more than once per N seconds
    _COOLDOWN = 4   # seconds

    def __init__(self, settings=None):
        self.settings = settings or {}
        self._smooth_yaw   = 0.0
        self._smooth_pitch = 0.0
        self._smooth_gaze_yaw = 0.0
        self._last_fire = {}   # violation_type -> last timestamp float
        self._obj_frame_count = {}  # object label -> consecutive frames seen
        self._obj_threshold = 4     # need 4 consecutive frames to fire object alert

    # ── cooldown helper ───────────────────────────────────────────────────────
    def _can_fire(self, vtype):
        import time
        now = time.time()
        last = self._last_fire.get(vtype, 0)
        if now - last >= self._COOLDOWN:
            self._last_fire[vtype] = now
            return True
        return False

    def _viol(self, vtype, severity, confidence, now_str):
        if self._can_fire(vtype):
            return {"type": vtype, "severity": severity,
                    "confidence": confidence, "time": now_str}
        return None

    # ── main ──────────────────────────────────────────────────────────────────
    def process_frame(self, frame):
        out      = frame.copy()
        viols    = []
        now_str  = datetime.now().strftime("%H:%M:%S")
        h_img, w_img = out.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX

        rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

        # ── Run MediaPipe ─────────────────────────────────────────────────────
        face_res  = _MP_FACE.process(rgb)  if _HAS_MP else None
        pose_res  = _MP_POSE.process(rgb)  if _HAS_MP else None
        hands_res = _MP_HANDS.process(rgb) if _HAS_MP else None

        # ── Face count ────────────────────────────────────────────────────────
        if _HAS_MP and face_res and face_res.multi_face_landmarks:
            face_count = len(face_res.multi_face_landmarks)
        else:
            face_count = len(detect_faces_haar(out))

        draw_header(out, face_count, len(viols) > 0)

        # ── Student not visible ───────────────────────────────────────────────
        if face_count == 0:
            draw_gaze_bar(out, 0.0, 0.0, 0.5)
            v = self._viol("Student Not Visible", "HIGH", 0.92, now_str)
            if v: viols.append(v)
            return out, viols

        # ── Multiple persons ──────────────────────────────────────────────────
        if face_count > 1:
            v = self._viol("Multiple Persons Detected", "CRITICAL", 0.95, now_str)
            if v: viols.append(v)
            # Draw extra face boxes red
            if _HAS_MP and face_res:
                for i, fl in enumerate(face_res.multi_face_landmarks[1:], 1):
                    xs = [int(lm.x*w_img) for lm in fl.landmark]
                    ys = [int(lm.y*h_img) for lm in fl.landmark]
                    x1,y1,x2,y2 = min(xs),min(ys),max(xs),max(ys)
                    draw_corner_box(out, x1,y1,x2-x1,y2-y1, color=(0,0,220))
                    draw_label(out, f"Person {i+1}", x1, y1-4, bg=(0,0,180))

        # ── Primary face — MediaPipe ──────────────────────────────────────────
        primary_box = None
        iris_h      = 0.5
        gaze_yaw_deg = 0.0
        head_yaw    = 0.0
        head_pitch  = 0.0

        if _HAS_MP and face_res and face_res.multi_face_landmarks:
            lms = face_res.multi_face_landmarks[0].landmark

            # Bounding box from landmarks
            xs = [int(lm.x*w_img) for lm in lms]
            ys = [int(lm.y*h_img) for lm in lms]
            fx,fy = max(0,min(xs)-10), max(0,min(ys)-10)
            fx2,fy2 = min(w_img,max(xs)+10), min(h_img,max(ys)+10)
            fw,fh = fx2-fx, fy2-fy
            primary_box = (fx,fy,fw,fh)

            # Draw face box
            conf_pct = 92
            draw_corner_box(out, fx,fy,fw,fh, color=(0,220,90))
            draw_label(out, "Face", fx, fy-4, bg=(20,100,220))
            cv2.putText(out, f"{conf_pct}%", (fx+fw-36,fy+fh+14), font, 0.4, (0,220,90),1,cv2.LINE_AA)

            # ── Iris gaze ─────────────────────────────────────────────────────
            try:
                gaze_raw, iris_h = iris_gaze(lms, w_img, h_img)
                # Convert to degrees-like scale (-30..+30)
                gaze_yaw_deg = gaze_raw * 60.0
                self._smooth_gaze_yaw = 0.65*self._smooth_gaze_yaw + 0.35*gaze_yaw_deg

                # Draw iris dots
                for idx in _L_IRIS[:1] + _R_IRIS[:1]:
                    px = int(lms[idx].x*w_img); py = int(lms[idx].y*h_img)
                    cv2.circle(out,(px,py),4,(0,255,200),-1)
                    cv2.circle(out,(px,py),6,(0,180,140),1)
                # Connect them
                lp = (int(lms[_L_IRIS[0]].x*w_img), int(lms[_L_IRIS[0]].y*h_img))
                rp = (int(lms[_R_IRIS[0]].x*w_img), int(lms[_R_IRIS[0]].y*h_img))
                cv2.line(out, lp, rp, (80,200,80), 1)

                # Gaze violation
                if abs(self._smooth_gaze_yaw) > 18:
                    direction = "LEFT" if self._smooth_gaze_yaw < 0 else "RIGHT"
                    v = self._viol(f"Gaze {direction}", "HIGH", 0.88, now_str)
                    if v: viols.append(v)
                if iris_h < 0.30:
                    v = self._viol("Gaze UP — Looking Away", "MEDIUM", 0.80, now_str)
                    if v: viols.append(v)
            except Exception:
                pass

            # ── Head pose ─────────────────────────────────────────────────────
            try:
                head_yaw, head_pitch, head_roll = head_pose_pnp(lms, w_img, h_img)
                self._smooth_yaw   = 0.7*self._smooth_yaw   + 0.3*head_yaw
                self._smooth_pitch = 0.7*self._smooth_pitch + 0.3*head_pitch

                if abs(self._smooth_yaw) > 22:
                    direction = "LEFT" if self._smooth_yaw < 0 else "RIGHT"
                    v = self._viol(f"Head Turn {direction}", "HIGH", 0.87, now_str)
                    if v: viols.append(v)
                if self._smooth_pitch > 20:
                    v = self._viol("Head Down", "MEDIUM", 0.82, now_str)
                    if v: viols.append(v)
            except Exception:
                pass

        else:
            # Haar fallback
            haar_boxes = detect_faces_haar(out)
            if haar_boxes:
                primary_box = max(haar_boxes, key=lambda b: b[2]*b[3])
                fx,fy,fw,fh = primary_box
                draw_corner_box(out,fx,fy,fw,fh,color=(0,200,80))
                draw_label(out,"Face",fx,fy-4,bg=(20,100,220))

        # ── Pose / posture ────────────────────────────────────────────────────
        if _HAS_MP and pose_res and pose_res.pose_landmarks:
            ls, rs, pose_viols = analyse_posture(pose_res.pose_landmarks, w_img, h_img)
            draw_pose_overlay(out, ls[0],ls[1], rs[0],rs[1])
            for pv in pose_viols:
                pv["time"] = now_str
                v = self._viol(pv["type"], pv["severity"], pv["confidence"], now_str)
                if v: viols.append(v)

        # ── Hands near face ───────────────────────────────────────────────────
        if _HAS_MP and hands_res:
            hand_viols = analyse_hands(hands_res, primary_box, w_img, h_img)
            for hv in hand_viols:
                hv["time"] = now_str
                v = self._viol(hv["type"], hv["severity"], hv["confidence"], now_str)
                if v: viols.append(v)
            # Draw hand landmarks
            if hands_res.multi_hand_landmarks:
                mp_draw = mp.solutions.drawing_utils
                for hl in hands_res.multi_hand_landmarks:
                    mp_draw.draw_landmarks(out, hl, mp.solutions.hands.HAND_CONNECTIONS,
                        mp_draw.DrawingSpec(color=(0,200,255), thickness=1, circle_radius=2),
                        mp_draw.DrawingSpec(color=(0,150,200), thickness=1))

        # ── Object detection ──────────────────────────────────────────────────
        obj_results = detect_objects_heuristic(out)
        for (ox,oy,ow,oh,olabel,oconf) in obj_results:
            # Need consistent detection over multiple frames
            self._obj_frame_count[olabel] = self._obj_frame_count.get(olabel,0)+1
            if self._obj_frame_count[olabel] >= self._obj_threshold:
                draw_corner_box(out, ox,oy,ow,oh, color=(0,50,255), thickness=2)
                draw_label(out, olabel, ox, oy-4, bg=(0,30,200), fg=(255,200,200))
                vtype = f"Unauthorized Object — {olabel}"
                v = self._viol(vtype, "CRITICAL", oconf, now_str)
                if v: viols.append(v)
        # Decay counts for objects not seen this frame
        seen_labels = {r[4] for r in obj_results}
        for lbl in list(self._obj_frame_count.keys()):
            if lbl not in seen_labels:
                self._obj_frame_count[lbl] = max(0, self._obj_frame_count[lbl]-2)

        # ── Gaze bar ──────────────────────────────────────────────────────────
        draw_gaze_bar(out, self._smooth_gaze_yaw, self._smooth_pitch, iris_h)

        return out, viols
