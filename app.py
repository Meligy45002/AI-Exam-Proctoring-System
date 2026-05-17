"""
AI Exam Proctoring System - Main Streamlit Dashboard
Team: Ahmed Mohamed Ezzat, Ahmed Mousa Mousa, Ahmed Ahmed Salah, Ahmed Ehab Kandeel, Ahmed Mohamed El Sayed
"""

import streamlit as st
import cv2
import numpy as np
import time
import json
import os
from datetime import datetime
from pathlib import Path
import threading
# ── Noise test is embedded below — no separate file needed ──

# ══════════════════════════════════════════════════════════════
# NOISE FUNCTIONS
# ══════════════════════════════════════════════════════════════

def _noise_gaussian(img, intensity):
    sigma = {1: 15, 2: 35, 3: 65}[intensity]
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

def _noise_salt_pepper(img, intensity):
    density = {1: 0.03, 2: 0.08, 3: 0.18}[intensity]
    out = img.copy()
    n = int((img.size // 3) * density)
    xs = np.random.randint(0, img.shape[1], n * 2)
    ys = np.random.randint(0, img.shape[0], n * 2)
    out[ys[:n], xs[:n]] = 255
    out[ys[n:], xs[n:]] = 0
    return out

def _noise_blur(img, intensity):
    k = {1: 9, 2: 19, 3: 35}[intensity]
    return cv2.GaussianBlur(img, (k, k), 0)

def _noise_motion_blur(img, intensity):
    size = {1: 10, 2: 20, 3: 40}[intensity]
    kernel = np.zeros((size, size))
    kernel[size // 2, :] = 1.0 / size
    return cv2.filter2D(img, -1, kernel)

def _noise_dark(img, intensity):
    factor = {1: 0.55, 2: 0.30, 3: 0.10}[intensity]
    return np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)

def _noise_bright(img, intensity):
    factor = {1: 1.5, 2: 2.0, 3: 2.8}[intensity]
    return np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)

def _noise_occlude(img, intensity):
    out = img.copy()
    h, w = out.shape[:2]
    s = {1: 0.20, 2: 0.35, 3: 0.50}[intensity]
    y1, y2 = int(h * 0.45), int(h * (0.45 + s))
    x1, x2 = int(w * 0.25), int(w * 0.75)
    out[y1:y2, x1:x2] = [80, 80, 80]
    return out

def _noise_jpeg(img, intensity):
    quality = {1: 30, 2: 12, 3: 3}[intensity]
    _, enc = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)

def _noise_pixel(img, intensity):
    factor = {1: 6, 2: 12, 3: 22}[intensity]
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(1, w // factor), max(1, h // factor)), interpolation=cv2.INTER_NEAREST)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

def _noise_color(img, intensity):
    amount = {1: 30, 2: 60, 3: 100}[intensity]
    out = img.astype(np.int32)
    for c in range(3):
        out[:, :, c] = np.clip(out[:, :, c] + np.random.randint(-amount, amount), 0, 255)
    return out.astype(np.uint8)

def _noise_rotate(img, intensity):
    angle = {1: 10, 2: 22, 3: 40}[intensity]
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

NOISE_CATALOG = {
    "Gaussian Noise":   (_noise_gaussian,    "Bad camera sensor"),
    "Salt & Pepper":    (_noise_salt_pepper,  "Pixel corruption"),
    "Blur":             (_noise_blur,         "Out-of-focus camera"),
    "Motion Blur":      (_noise_motion_blur,  "Fast head movement"),
    "Low Brightness":   (_noise_dark,         "Dark room"),
    "High Brightness":  (_noise_bright,       "Bright backlight"),
    "Face Occlusion":   (_noise_occlude,      "Hand covering face"),
    "JPEG Compression": (_noise_jpeg,         "Low-bandwidth stream"),
    "Pixelation":       (_noise_pixel,        "Extreme compression"),
    "Color Jitter":     (_noise_color,        "Bad white balance"),
    "Rotation":         (_noise_rotate,       "Tilted camera"),
}

def _get_students(data_dir="data/registered_faces"):
    students = {}
    p = Path(data_dir)
    if not p.exists():
        return students
    for d in sorted(p.iterdir()):
        if d.is_dir():
            photos = sorted(list(d.glob("*.jpg")) + list(d.glob("*.png")))
            if photos:
                display = d.name.rsplit("_", 1)[0]
                students[display] = [str(ph) for ph in photos]
    return students

def _get_face_embedding(detector, img_bgr):
    """Extract face embedding. Returns (embedding_or_None, face_count)."""
    try:
        return detector.get_embedding(img_bgr)
    except Exception:
        pass
    # Final fallback
    try:
        from proctoring_engine import detect_faces_haar, face_histogram, crop_face
        boxes = detect_faces_haar(img_bgr)
        if not boxes:
            return None, 0
        crop = crop_face(img_bgr, boxes[0])
        return face_histogram(crop), len(boxes)
    except Exception:
        return None, 0

def _cosine_sim(a, b):
    a, b = np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d > 0 else 0.0

def _build_face_db(detector, students):
    """Build name -> [embeddings] from registered photos."""
    # Use the engine's own method if available
    try:
        if hasattr(detector, 'build_face_db'):
            return detector.build_face_db()
    except Exception:
        pass
    # Manual fallback
    db = {}
    for name, photos in students.items():
        embs = []
        for ph in photos:
            img = cv2.imread(ph)
            if img is None:
                continue
            emb, _ = _get_face_embedding(detector, img)
            if emb is not None:
                embs.append(emb)
        if embs:
            db[name] = embs
    return db

def _run_recognition(detector, img_bgr, face_db=None):
    """Recognize face — uses direct embedding comparison if face_db provided."""
    # Use the FaceDetector.identify() method if available (preferred)
    try:
        if face_db is not None and hasattr(detector, 'identify'):
            return detector.identify(img_bgr, face_db, threshold=0.55)
    except Exception:
        pass

    if face_db:
        emb, n_faces = _get_face_embedding(detector, img_bgr)
        if n_faces == 0:
            return "Unknown", 0.0, 0
        if emb is not None:
            best_name, best_sim = "Unknown", 0.0
            for sname, embs in face_db.items():
                sim = max(_cosine_sim(emb, e) for e in embs)
                if sim > best_sim:
                    best_sim, best_name = sim, sname
            # Histogram similarity threshold is higher than embedding (0.55)
            return (best_name, best_sim, n_faces) if best_sim >= 0.55 else ("Unknown", best_sim, n_faces)
        return "Face found (no ID)", 0.0, n_faces
    # Fallback to detector method
    try:
        faces, name, conf = detector.detect_and_verify(img_bgr)
        return name or "Unknown", float(conf) if conf else 0.0, len(faces)
    except Exception:
        return "Error", 0.0, 0

def _is_match(expected, got):
    if not got or got in ("Unknown", "Error", "Face found (no ID)"):
        return False
    return expected.lower().split()[0] in got.lower() or got.lower().split()[0] in expected.lower()

def _result_card_nt(label, expected, got_name, confidence, faces):
    if faces == 0:
        icon, color, verdict = "😶", "#ef4444", "NO FACE DETECTED"
        bg, border = "rgba(239,68,68,0.12)", "#ef4444"
    elif _is_match(expected, got_name):
        icon, color, verdict = "✅", "#22c55e", f"RECOGNIZED AS: {got_name}"
        bg, border = "rgba(34,197,94,0.12)", "#22c55e"
    else:
        icon, color, verdict = "❌", "#f97316", f"WRONG: GOT '{got_name}'"
        bg, border = "rgba(249,115,22,0.12)", "#f97316"
    st.markdown(f"""
    <div style='background:{bg};border:2px solid {border};border-radius:12px;padding:16px 20px;'>
        <div style='font-size:1.15rem;font-weight:700;color:{color};margin-bottom:8px;'>{icon} {verdict}</div>
        <div style='color:#e0e0e0;font-size:0.92rem;line-height:1.8;'>
            <b>Test:</b> {label}<br>
            <b>Expected name:</b> {expected}<br>
            <b>Model said:</b> {got_name}<br>
            <b>Confidence:</b> {confidence * 100:.1f}%<br>
            <b>Faces found:</b> {faces}
        </div>
    </div>""", unsafe_allow_html=True)

def render_noise_test():
    st.markdown("""
    <div style='background:linear-gradient(135deg,#1a1a2e,#0f3460);
                border-radius:14px;padding:24px 28px;margin-bottom:20px;
                border:1px solid rgba(99,179,237,0.3);'>
        <h1 style='color:#fff;margin:0;font-size:1.85rem;'>🧪 Noise Robustness Test</h1>
        <p style='color:#93c5fd;margin:8px 0 0 0;'>
            Select a registered student → their photo loads automatically →
            noise is applied → the model must still recognize them by name.
        </p>
    </div>""", unsafe_allow_html=True)

    # Load detector once
    if "nt_detector" not in st.session_state:
        with st.spinner("Loading face recognition model..."):
            try:
                from proctoring_engine import FaceDetector
                st.session_state.nt_detector = FaceDetector()
                st.success("✅ Face recognition engine loaded (OpenCV + Haar cascade).")
            except Exception as e:
                st.error(f"❌ Could not load FaceDetector: {e}")
                st.session_state.nt_detector = None

    detector = st.session_state.nt_detector
    if detector is None:
        st.error("❌ FaceDetector not loaded. Make sure proctoring_engine.py is in the same folder.")
        return

    students = _get_students()
    if not students:
        st.warning("⚠️ No registered students found. Go to 👤 Student Registration tab first.")
        return

    # Build face embedding DB once per session
    if "nt_face_db" not in st.session_state:
        with st.spinner("Building face database from registered photos..."):
            st.session_state.nt_face_db = _build_face_db(detector, students)
        db_size = sum(len(v) for v in st.session_state.nt_face_db.values())
        if db_size == 0:
            st.warning("⚠️ Could not extract face embeddings from registered photos. "
                       "InsightFace may not be fully loaded — face detection will still work.")
        else:
            st.success(f"✅ Face database ready: {db_size} embeddings for {len(st.session_state.nt_face_db)} students.")
    face_db = st.session_state.nt_face_db

    # ── STEP 1: Pick student ──────────────────────────────────
    st.markdown("### 👤 Step 1 — Select student")

    names = list(students.keys())
    btn_cols = st.columns(min(len(names), 4))
    for i, n in enumerate(names):
        if btn_cols[i % 4].button(f"👤 {n}", key=f"nt_btn_{n}", use_container_width=True):
            st.session_state["nt_selected"] = n

    typed = st.text_input("Or type a name:", placeholder="e.g. Ahmed Ezzat",
                          help="Registered: " + ", ".join(names))
    if typed.strip():
        hit = next((n for n in names if typed.strip().lower() in n.lower()), None)
        if hit:
            st.session_state["nt_selected"] = hit
        else:
            st.error(f"❌ '{typed}' not found. Available: {', '.join(names)}")

    selected = st.session_state.get("nt_selected")
    if not selected:
        st.info("👆 Click a student button or type a name above.")
        return

    st.success(f"✅ Student: **{selected}**")

    photos = students[selected]
    photo_idx = 0
    if len(photos) > 1:
        photo_idx = st.select_slider(f"Photo ({len(photos)} available)",
                                     options=list(range(len(photos))),
                                     format_func=lambda x: f"Photo {x+1}")

    original_bgr = cv2.imread(photos[photo_idx])
    if original_bgr is None:
        st.error("Could not read photo file.")
        return

    # ── STEP 2: Baseline ──────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🔍 Step 2 — Baseline: recognition on clean photo")

    c1, c2 = st.columns(2)
    with c1:
        st.image(cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB),
                 caption=f"Registered photo of {selected}", use_container_width=True)
    with c2:
        base_name, base_conf, base_faces = _run_recognition(detector, original_bgr, face_db)
        _result_card_nt("Clean image", selected, base_name, base_conf, base_faces)

    if base_faces == 0:
        st.warning("⚠️ No face detected in the clean photo. Try a different photo number or re-register.")
        return

    # ── STEP 3: Choose noise ──────────────────────────────────
    st.markdown("---")
    st.markdown("### 🌫️ Step 3 — Choose noise type and intensity")

    cn, ci = st.columns([2, 1])
    with cn:
        noise_type = st.selectbox("Noise type", list(NOISE_CATALOG.keys()),
                                  format_func=lambda k: f"{k}  —  {NOISE_CATALOG[k][1]}")
    with ci:
        intensity = st.select_slider("Intensity", options=[1, 2, 3], value=2,
                                     format_func=lambda x: {1:"🟢 Low", 2:"🟡 Medium", 3:"🔴 High"}[x])

    # ── STEP 4: Apply & test ──────────────────────────────────
    st.markdown("---")
    st.markdown("### ⚡ Step 4 — Apply noise → test identity recognition")

    fn, _ = NOISE_CATALOG[noise_type]
    try:
        noisy_bgr = fn(original_bgr.copy(), intensity)
    except Exception as e:
        st.error(f"Noise failed: {e}"); return

    cn2, cr2 = st.columns(2)
    with cn2:
        st.image(cv2.cvtColor(noisy_bgr, cv2.COLOR_BGR2RGB),
                 caption=f"{noise_type} — intensity {intensity}", use_container_width=True)
    with cr2:
        noisy_name, noisy_conf, noisy_faces = _run_recognition(detector, noisy_bgr, face_db)
        _result_card_nt(f"After {noise_type}", selected, noisy_name, noisy_conf, noisy_faces)

    # ── STEP 5: Verdict ───────────────────────────────────────
    st.markdown("---")
    lbl = {1:"Low", 2:"Medium", 3:"High"}[intensity]
    correct = _is_match(selected, noisy_name)
    drop = (base_conf - noisy_conf) * 100

    if noisy_faces == 0:
        color, bg = "#ef4444", "rgba(239,68,68,0.12)"
        title = "💀 COMPLETE FAILURE — face not detected"
        body  = f"**{noise_type}** at **{lbl}** intensity destroyed the face entirely. Model couldn't find **{selected}** at all."
    elif correct:
        color, bg = "#22c55e", "rgba(34,197,94,0.1)"
        title = f"✅ ROBUST — model still recognizes {selected}"
        body  = f"Even with **{noise_type}** at **{lbl}** intensity, model correctly identified **{selected}**. Confidence dropped by {drop:.1f}%."
    else:
        color, bg = "#f97316", "rgba(249,115,22,0.12)"
        title = f"❌ CONFUSED — expected '{selected}', model said '{noisy_name}'"
        body  = f"**{noise_type}** at **{lbl}** intensity caused the model to lose **{selected}**'s identity."

    st.markdown(f"""
    <div style='background:{bg};border:2px solid {color};border-radius:14px;padding:18px 24px;'>
        <div style='font-size:1.2rem;font-weight:700;color:{color};'>{title}</div>
    </div>""", unsafe_allow_html=True)
    st.markdown(body)

    m1, m2, m3 = st.columns(3)
    m1.metric("Clean confidence",  f"{base_conf * 100:.1f}%")
    m2.metric("Noisy confidence",  f"{noisy_conf * 100:.1f}%", delta=f"-{drop:.1f}%")
    m3.metric("Faces detected",    noisy_faces)

    # ── STEP 6: Full batch test ───────────────────────────────
    st.markdown("---")
    st.markdown("### 📊 Step 6 — Full batch test (all noise × all intensities)")
    st.caption(f"Tests all 11 noise types at Low / Medium / High on **{selected}**'s photo and checks if the model still says their name.")

    if st.button("🚀 Run full batch test", type="primary", key="nt_batch"):
        import pandas as pd
        rows = []
        total = len(NOISE_CATALOG) * 3
        bar = st.progress(0); done = 0
        lbl_map = {1:"Low", 2:"Medium", 3:"High"}

        for nname, (nfn, ndesc) in NOISE_CATALOG.items():
            for nint in [1, 2, 3]:
                try:
                    nimg = nfn(original_bgr.copy(), nint)
                    gname, gconf, gfaces = _run_recognition(detector, nimg, face_db)
                except Exception:
                    gname, gconf, gfaces = "Error", 0.0, 0

                ok = _is_match(selected, gname)
                if gfaces == 0:
                    status = "💀 No Face"
                elif ok:
                    status = "✅ Correct"
                else:
                    status = f"❌ Said '{gname}'"

                rows.append({
                    "Noise Type":  nname,
                    "Simulates":   ndesc,
                    "Intensity":   lbl_map[nint],
                    "Result":      status,
                    "Confidence":  f"{gconf * 100:.1f}%",
                    "Conf. Drop":  f"−{max(0,(base_conf-gconf)*100):.1f}%",
                })
                done += 1
                bar.progress(done / total)

        bar.empty()
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        correct_n  = sum(1 for r in rows if "Correct"  in r["Result"])
        confused_n = sum(1 for r in rows if "Said"     in r["Result"])
        noface_n   = sum(1 for r in rows if "No Face"  in r["Result"])

        st.markdown("---")
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Total tests",           total)
        b2.metric(f"✅ '{selected}'",       correct_n)
        b3.metric("❌ Wrong identity",      confused_n)
        b4.metric("💀 No face",             noface_n)

        os.makedirs("data", exist_ok=True)
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        rpath = f"data/noise_report_{selected.replace(' ','_')}_{ts}.json"
        with open(rpath, "w") as f:
            json.dump({"student": selected, "timestamp": datetime.now().isoformat(),
                       "baseline_confidence": round(base_conf * 100, 1),
                       "results": rows,
                       "summary": {"total": total, "correct": correct_n,
                                   "confused": confused_n, "no_face": noface_n}}, f, indent=2)
        st.success(f"📄 Report saved → `{rpath}`")

# Page configuration
st.set_page_config(
    page_title="AI Exam Proctoring System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    
    .main { background: #0f1117; color: #e0e0e0; }
    
    .alert-card {
        background: linear-gradient(135deg, #ff4444, #cc0000);
        border-radius: 12px;
        padding: 16px;
        margin: 8px 0;
        color: white;
        box-shadow: 0 4px 15px rgba(255,68,68,0.3);
        animation: pulse 2s infinite;
    }
    
    .warning-card {
        background: linear-gradient(135deg, #ff9800, #e65100);
        border-radius: 12px;
        padding: 16px;
        margin: 8px 0;
        color: white;
    }
    
    .ok-card {
        background: linear-gradient(135deg, #4caf50, #1b5e20);
        border-radius: 12px;
        padding: 16px;
        margin: 8px 0;
        color: white;
    }
    
    .metric-card {
        background: #1e2130;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 1px solid #2d3250;
    }
    
    .metric-value {
        font-size: 2.5em;
        font-weight: 700;
        color: #5c6bc0;
    }
    
    .metric-label {
        font-size: 0.85em;
        color: #9e9e9e;
        margin-top: 4px;
    }
    
    .student-badge {
        background: #1a237e;
        border: 2px solid #5c6bc0;
        border-radius: 50px;
        padding: 8px 20px;
        color: #c5cae9;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 10px;
    }
    
    @keyframes pulse {
        0% { box-shadow: 0 4px 15px rgba(255,68,68,0.3); }
        50% { box-shadow: 0 4px 30px rgba(255,68,68,0.7); }
        100% { box-shadow: 0 4px 15px rgba(255,68,68,0.3); }
    }
    
    .sidebar .stButton button {
        width: 100%;
        border-radius: 8px;
        font-weight: 600;
    }
    
    .stProgress > div > div { background: #5c6bc0; }
    
    h1, h2, h3 { color: #c5cae9 !important; }
    
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }
    .status-active { background: #4caf50; box-shadow: 0 0 8px #4caf50; }
    .status-inactive { background: #f44336; }
    .status-warning { background: #ff9800; box-shadow: 0 0 8px #ff9800; }
</style>
""", unsafe_allow_html=True)


def load_violation_log():
    """Load violation log from file."""
    log_path = Path("data/violation_log.json")
    if log_path.exists():
        with open(log_path, "r") as f:
            return json.load(f)
    return []


def save_violation_log(log):
    """Save violation log to file."""
    Path("data").mkdir(exist_ok=True)
    with open("data/violation_log.json", "w") as f:
        json.dump(log, f, indent=2)


def get_registered_students():
    """Get list of registered students."""
    faces_dir = Path("data/registered_faces")
    if not faces_dir.exists():
        return []
    return [d.name for d in faces_dir.iterdir() if d.is_dir()]


def render_header():
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.markdown("""
        <div style='text-align:center; padding: 20px 0;'>
            <h1 style='font-size:2.2em; margin:0;'>🎓 AI Exam Proctoring System</h1>
            <p style='color:#7986cb; margin:5px 0;'>Real-time monitoring powered by Computer Vision & AI</p>
        </div>
        """, unsafe_allow_html=True)


def render_sidebar():
    with st.sidebar:
        st.markdown("## ⚙️ Control Panel")
        
        mode = st.selectbox(
            "Mode",
            ["📹 Live Proctoring", "👤 Student Registration", "📊 Exam Report", "🧪 Noise Test", "⚙️ Settings"],
            index=0
        )
        
        st.markdown("---")
        st.markdown("### 🎓 Session Info")
        
        exam_name = st.text_input("Exam Name", value="Final Exam 2024")
        duration = st.number_input("Duration (min)", min_value=10, max_value=300, value=90)
        
        st.markdown("---")
        st.markdown("### 🔧 Detection Settings")
        
        sensitivity = st.slider("Alert Sensitivity", 1, 10, 7)
        
        st.markdown("**Detection Modules:**")
        detect_gaze = st.checkbox("👁️ Gaze Tracking", value=True)
        detect_face = st.checkbox("🆔 Face Verification", value=True)
        detect_phone = st.checkbox("📱 Phone Detection", value=True)
        detect_multi = st.checkbox("👥 Multiple Persons", value=True)
        
        st.markdown("---")
        
        registered = get_registered_students()
        st.markdown(f"### 👤 Students ({len(registered)})")
        for s in registered[:5]:
            st.markdown(f'<div class="student-badge">👤 {s}</div>', unsafe_allow_html=True)
        if len(registered) > 5:
            st.caption(f"...and {len(registered)-5} more")
        
        settings = {
            "sensitivity": sensitivity,
            "detect_gaze": detect_gaze,
            "detect_face": detect_face,
            "detect_phone": detect_phone,
            "detect_multi": detect_multi,
            "exam_name": exam_name,
            "duration": duration
        }
        
        return mode.split(" ", 1)[1], settings


def render_live_proctoring(settings):
    """Render live proctoring view."""
    try:
        from proctoring_engine import ProctoringEngine
    except ImportError:
        st.error("❌ proctoring_engine.py not found in the project folder.")
        return
    try:
        import mediapipe
    except ImportError:
        st.warning("⚠️ MediaPipe not installed. Run this in PowerShell then restart the app:")
        st.code('python -m pip install mediapipe')
        st.info("Without MediaPipe, basic Haar face detection is still active.")
    
    col_video, col_alerts = st.columns([3, 2])
    
    with col_video:
        st.markdown("### 📹 Live Feed")
        
        status_col1, status_col2, status_col3 = st.columns(3)
        with status_col1:
            st.markdown('<span class="status-indicator status-active"></span>**Camera Active**', unsafe_allow_html=True)
        with status_col2:
            st.markdown('<span class="status-indicator status-active"></span>**AI Running**', unsafe_allow_html=True)
        with status_col3:
            st.markdown(f'<span class="status-indicator status-warning"></span>**Sensitivity: {settings["sensitivity"]}**', unsafe_allow_html=True)
        
        frame_placeholder = st.empty()
        
    with col_alerts:
        st.markdown("### 🚨 Live Alerts")
        alerts_placeholder = st.empty()
        
        st.markdown("### 📊 Session Stats")
        metrics_placeholder = st.empty()
    
    # Initialize engine in session state
    if "engine" not in st.session_state:
        st.session_state.engine = ProctoringEngine(settings)
        st.session_state.violation_count = 0
        st.session_state.violations = []
        st.session_state.session_start = time.time()
    
    # Control buttons
    col_start, col_stop, col_snap = st.columns(3)
    
    with col_start:
        start = st.button("▶️ Start Monitoring", type="primary", use_container_width=True)
    with col_stop:
        stop = st.button("⏹️ Stop", use_container_width=True)
    with col_snap:
        snap = st.button("📸 Snapshot", use_container_width=True)
    
    if stop:
        st.session_state.running = False
        st.info("Monitoring stopped.")
        return
    
    if start:
        st.session_state.running = True
    
    if st.session_state.get("running", False):
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("❌ Cannot access webcam. Please check your camera connection.")
            # Show demo mode
            render_demo_mode(frame_placeholder, alerts_placeholder, metrics_placeholder, settings)
            return
        
        try:
            while st.session_state.get("running", False):
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                processed_frame, violations = st.session_state.engine.process_frame(frame)
                
                # Update violations
                if violations:
                    for v in violations:
                        st.session_state.violations.append({
                            "time": datetime.now().strftime("%H:%M:%S"),
                            "type": v["type"],
                            "severity": v["severity"],
                            "confidence": v.get("confidence", 0.9)
                        })
                        st.session_state.violation_count += 1
                
                # Display frame
                frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                
                # Update alerts
                _render_alerts(alerts_placeholder, st.session_state.violations[-5:])
                
                # Update metrics
                elapsed = int(time.time() - st.session_state.session_start)
                _render_metrics(metrics_placeholder, st.session_state.violation_count, elapsed)
                
                time.sleep(0.033)  # ~30 FPS
        finally:
            cap.release()
    else:
        # Show placeholder when not running
        placeholder_img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(placeholder_img, "Click 'Start Monitoring' to begin", 
                   (80, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)
        frame_placeholder.image(placeholder_img, channels="BGR", use_container_width=True)
        
        _render_alerts(alerts_placeholder, [])
        _render_metrics(metrics_placeholder, 0, 0)


def render_demo_mode(frame_placeholder, alerts_placeholder, metrics_placeholder, settings):
    """Demo mode when no webcam is available."""
    st.warning("🎬 Running in Demo Mode (no webcam detected)")
    
    demo_violations = [
        {"time": "10:23:45", "type": "Looking Away", "severity": "HIGH", "confidence": 0.92},
        {"time": "10:24:12", "type": "Phone Detected", "severity": "CRITICAL", "confidence": 0.88},
        {"time": "10:25:01", "type": "Multiple Faces", "severity": "HIGH", "confidence": 0.95},
    ]
    
    # Create demo frame
    demo_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    demo_frame[:] = (20, 25, 40)
    
    # Draw fake face detection box
    cv2.rectangle(demo_frame, (220, 120), (420, 360), (0, 255, 100), 2)
    cv2.putText(demo_frame, "DEMO MODE", (230, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
    cv2.putText(demo_frame, "Student: Ahmed M.", (225, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 255), 1)
    cv2.putText(demo_frame, "ID: Verified ✓", (225, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 100), 1)
    
    # Add gaze arrows
    cv2.arrowedLine(demo_frame, (320, 200), (360, 200), (255, 150, 0), 2)
    cv2.putText(demo_frame, "GAZE: FORWARD", (240, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    frame_placeholder.image(demo_frame, channels="BGR", use_container_width=True)
    _render_alerts(alerts_placeholder, demo_violations)
    _render_metrics(metrics_placeholder, 3, 305)


def _render_alerts(placeholder, violations):
    """Big red alert cards exactly like the target demo video."""
    with placeholder.container():
        if not violations:
            st.markdown("""
            <div style='background:linear-gradient(135deg,rgba(34,197,94,0.15),rgba(21,128,61,0.25));
                        border:1px solid rgba(34,197,94,0.5);border-left:4px solid #22c55e;
                        border-radius:12px;padding:14px 16px;color:#86efac;'>
                ✅ <strong>No violations detected</strong>
            </div>""", unsafe_allow_html=True)
            return

        ICONS = {"CRITICAL": "🚨", "HIGH": "⚠️", "MEDIUM": "ℹ️"}
        for v in reversed(violations[-5:]):
            severity = v.get("severity", "HIGH")
            icon = ICONS.get(severity, "⚠️")
            conf_pct = int(v.get("confidence", 0.9) * 100)
            t = v.get("time", "")
            vtype = v["type"]
            st.markdown(f"""
            <div style='background:linear-gradient(135deg,rgba(239,68,68,0.18),rgba(185,28,28,0.30));
                        border:1px solid rgba(239,68,68,0.55);border-left:4px solid #ef4444;
                        border-radius:12px;padding:14px 18px;margin:6px 0;
                        box-shadow:0 0 18px rgba(239,68,68,0.18);'>
                <div style='font-size:1.0rem;font-weight:700;color:#fca5a5;margin-bottom:4px;'>
                    {icon} {vtype}
                </div>
                <div style='font-size:0.78rem;color:#fca5a5;opacity:0.8;'>
                    🕐 {t} | Confidence: {conf_pct}%
                </div>
            </div>""", unsafe_allow_html=True)


def _render_metrics(placeholder, violation_count, elapsed_seconds):
    with placeholder.container():
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{violation_count}</div>
                <div class="metric-label">Violations</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            mins = elapsed_seconds // 60
            secs = elapsed_seconds % 60
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{mins:02d}:{secs:02d}</div>
                <div class="metric-label">Elapsed</div>
            </div>
            """, unsafe_allow_html=True)


def render_registration():
    """Render student registration page."""
    st.markdown("## 👤 Student Registration")
    st.markdown("Register students by capturing their face for identity verification.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📝 Student Details")
        student_name = st.text_input("Full Name", placeholder="e.g. Ahmed Mohamed")
        student_id = st.text_input("Student ID", placeholder="e.g. 20210001")
        
        st.markdown("### 📸 Capture Method")
        method = st.radio("", ["Webcam Capture", "Upload Photo"])
        
        if method == "Upload Photo":
            uploaded = st.file_uploader("Upload Student Photo", type=["jpg", "jpeg", "png"])
            if uploaded and student_name and student_id:
                if st.button("✅ Register Student", type="primary"):
                    _register_student_from_upload(student_name, student_id, uploaded)
        else:
            if student_name and student_id:
                if st.button("📸 Capture & Register", type="primary"):
                    _register_student_from_webcam(student_name, student_id)
    
    with col2:
        st.markdown("### 👥 Registered Students")
        students = get_registered_students()
        
        if students:
            for name in students:
                face_dir = Path(f"data/registered_faces/{name}")
                photos = list(face_dir.glob("*.jpg")) + list(face_dir.glob("*.png"))
                
                with st.expander(f"👤 {name}"):
                    if photos:
                        img = cv2.imread(str(photos[0]))
                        if img is not None:
                            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            st.image(img_rgb, width=150)
                    st.caption(f"Photos: {len(photos)}")
                    if st.button(f"🗑️ Remove", key=f"del_{name}"):
                        import shutil
                        shutil.rmtree(face_dir)
                        st.rerun()
        else:
            st.info("No students registered yet.")


def _register_student_from_upload(name, student_id, uploaded_file):
    face_dir = Path(f"data/registered_faces/{name}_{student_id}")
    face_dir.mkdir(parents=True, exist_ok=True)
    
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    if img is not None:
        save_path = face_dir / f"photo_0.jpg"
        cv2.imwrite(str(save_path), img)
        st.success(f"✅ Successfully registered {name} (ID: {student_id})")
        st.balloons()
    else:
        st.error("Failed to process image.")


def _register_student_from_webcam(name, student_id):
    face_dir = Path(f"data/registered_faces/{name}_{student_id}")
    face_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Cannot access webcam.")
        return
    
    frames_captured = 0
    placeholder = st.empty()
    progress = st.progress(0)
    
    for i in range(5):
        ret, frame = cap.read()
        if ret:
            save_path = face_dir / f"photo_{i}.jpg"
            cv2.imwrite(str(save_path), frame)
            frames_captured += 1
            progress.progress((i+1)/5)
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            placeholder.image(frame_rgb, width=300)
            time.sleep(0.5)
    
    cap.release()
    
    if frames_captured > 0:
        st.success(f"✅ Registered {name} with {frames_captured} photos!")
        st.balloons()
    else:
        st.error("Failed to capture photos.")


def render_report():
    """Render exam report."""
    st.markdown("## 📊 Exam Report")
    
    violations = load_violation_log()
    
    # Summary stats
    col1, col2, col3, col4 = st.columns(4)
    
    total = len(violations)
    critical = sum(1 for v in violations if v.get("severity") == "CRITICAL")
    high = sum(1 for v in violations if v.get("severity") == "HIGH")
    students = len(set(v.get("student", "Unknown") for v in violations))
    
    for col, (val, label, color) in zip(
        [col1, col2, col3, col4],
        [(total, "Total Violations", "#5c6bc0"),
         (critical, "Critical", "#f44336"),
         (high, "High Severity", "#ff9800"),
         (students, "Students Flagged", "#4caf50")]
    ):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color:{color}">{val}</div>
                <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Charts
    if violations:
        import pandas as pd
        
        df = pd.DataFrame(violations)
        
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.markdown("### Violations by Type")
            if "type" in df.columns:
                type_counts = df["type"].value_counts()
                st.bar_chart(type_counts)
        
        with col_chart2:
            st.markdown("### Violations by Severity")
            if "severity" in df.columns:
                sev_counts = df["severity"].value_counts()
                st.bar_chart(sev_counts)
        
        st.markdown("### 📋 Violation Log")
        st.dataframe(df, use_container_width=True)
        
        # Export
        csv = df.to_csv(index=False)
        st.download_button(
            "⬇️ Download Report (CSV)",
            csv,
            file_name=f"exam_report_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )
    else:
        st.info("No violations recorded yet. Start a proctoring session to generate data.")
        
        # Demo data
        st.markdown("### 📈 Sample Report Preview")
        import pandas as pd
        demo_data = {
            "time": ["10:23:45", "10:24:12", "10:25:01", "10:27:33", "10:29:10"],
            "type": ["Looking Away", "Phone Detected", "Multiple Faces", "Looking Away", "Head Turn"],
            "severity": ["HIGH", "CRITICAL", "CRITICAL", "MEDIUM", "HIGH"],
            "student": ["Ahmed M.", "Sara K.", "Ahmed M.", "Omar H.", "Sara K."],
            "confidence": [0.92, 0.88, 0.95, 0.75, 0.83]
        }
        st.dataframe(pd.DataFrame(demo_data), use_container_width=True)


def render_settings():
    """Render settings page."""
    st.markdown("## ⚙️ System Settings")
    
    tab1, tab2, tab3 = st.tabs(["🔧 General", "🎯 Detection", "📧 Notifications"])
    
    with tab1:
        st.markdown("### General Configuration")
        st.text_input("Institution Name", value="Cairo University")
        st.text_input("Department", value="Computer Science")
        st.selectbox("Language", ["English", "Arabic", "French"])
        st.selectbox("Camera Resolution", ["640x480", "1280x720", "1920x1080"])
        st.number_input("Frame Rate (FPS)", min_value=10, max_value=60, value=30)
    
    with tab2:
        st.markdown("### Detection Thresholds")
        st.slider("Face Match Threshold", 0.1, 1.0, 0.6, help="Lower = more strict identity matching")
        st.slider("Gaze Deviation Threshold (degrees)", 5, 45, 20)
        st.slider("Phone Detection Confidence", 0.1, 1.0, 0.7)
        st.slider("Multiple Person Confidence", 0.1, 1.0, 0.8)
        
        st.markdown("### Timing")
        st.number_input("Alert Cooldown (seconds)", min_value=1, max_value=30, value=5)
        st.number_input("Violation Buffer Frames", min_value=1, max_value=30, value=10)
    
    with tab3:
        st.markdown("### Notification Settings")
        st.checkbox("Email alerts to supervisor", value=True)
        st.text_input("Supervisor Email", value="supervisor@university.edu")
        st.checkbox("Log all violations to file", value=True)
        st.checkbox("Auto-screenshot on violation", value=True)
        st.text_input("Screenshots directory", value="data/screenshots")
    
    if st.button("💾 Save Settings", type="primary"):
        st.success("✅ Settings saved successfully!")


# ─── Main App ───────────────────────────────────────────────────────────────

def main():
    render_header()
    mode, settings = render_sidebar()
    
    st.markdown("---")
    
    if mode == "Live Proctoring":
        render_live_proctoring(settings)
    elif mode == "Student Registration":
        render_registration()
    elif mode == "Exam Report":
        render_report()
    elif mode == "Noise Test":
        render_noise_test()
    elif mode == "Settings":
        render_settings()


if __name__ == "__main__":
    main()
