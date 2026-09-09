import os
import io
import warnings

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import streamlit as st
import cv2
import numpy as np
import pytesseract
import re
import pandas as pd
import tensorflow as tf
    

# Charts work with plotly if installed, else matplotlib, else built-in Streamlit charts.
CHART_ENGINE = "native"
try:
    import plotly.express as px
    import plotly.graph_objects as go

    CHART_ENGINE = "plotly"
except ModuleNotFoundError:
    try:
        import matplotlib.pyplot as plt

        CHART_ENGINE = "matplotlib"
    except ModuleNotFoundError:
        pass

from datetime import datetime
from PIL import Image, ImageChops, ImageEnhance
from deepface import DeepFace
st.set_page_config(
    page_title="Universal Document Forensic & Verification Engine",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================ UI THEME (CSS)
THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --bg: #f7fbff;
    --panel: #ffffff;
    --panel-2: #f1f7ff;
    --line: #cfe2f8;
    --accent: #1769c2;
    --accent-2: #0d88dc;
    --text: #102a43;
    --muted: #5d7690;
}

html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background: radial-gradient(1200px 600px at 80% -10%, rgba(53, 137, 222, 0.16), transparent 60%),
                radial-gradient(900px 500px at -10% 10%, rgba(144, 205, 255, 0.18), transparent 55%),
                var(--bg) !important;
    color: var(--text);
    font-family: 'Inter', system-ui, sans-serif;
}
[data-testid="stHeader"] { background: transparent; }

h1, h2, h3, h4, p, label, span, div { font-family: 'Inter', system-ui, sans-serif; }
h1, h2, h3 { color: var(--text); letter-spacing: -0.01em; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #ffffff 0%, #eef6ff 100%);
    border-right: 1px solid var(--line);
}
[data-testid="stSidebar"] * { color: var(--text); }

/* Hero banner */
.hero {
    background: linear-gradient(120deg, #e9f4ff, #f8fbff);
    border: 1px solid var(--line);
    border-radius: 18px;
    padding: 26px 30px;
    margin-bottom: 18px;
}
.hero h1 { margin: 0; font-size: 1.9rem; font-weight: 800;
    background: linear-gradient(90deg, #0b5cab, #1686d9);
    -webkit-background-clip: text; background-clip: text; color: transparent; }
.hero p { margin: 6px 0 0 0; color: var(--muted); font-size: 0.95rem; }

/* Section captions */
.section-tag {
    display: inline-block; font-size: 0.72rem; font-weight: 700;
    letter-spacing: 0.12em; text-transform: uppercase; color: var(--accent);
    border: 1px solid var(--line); border-radius: 999px;
    padding: 3px 12px; margin-bottom: 6px;
    background: #edf6ff;
}

/* Cards */
.card {
    background: linear-gradient(180deg, var(--panel) 0%, var(--panel-2) 100%);
    border: 1px solid var(--line);
    border-radius: 16px;
    padding: 18px 20px;
    margin-bottom: 14px;
    box-shadow: 0 8px 24px rgba(44, 105, 165, 0.10);
}
.card-title { font-size: 1.02rem; font-weight: 700; color: #102a43 !important; margin: 0 0 4px 0; }
.card-sub { font-size: 0.85rem; color: #47708f !important; margin: 0 0 10px 0; }

/* Extracted field rows */
.field-row {
    display: flex; justify-content: space-between; gap: 12px;
    padding: 8px 12px; border-radius: 10px;
    background: #f7fbff;
    border: 1px solid #dceafa;
    margin-bottom: 6px; font-size: 0.88rem;
}
.field-key { color: var(--muted); font-weight: 600; }
.field-val { color: #123a60; font-weight: 600; text-align: right;
    font-family: 'JetBrains Mono', ui-monospace, monospace; word-break: break-all; }
.field-val.missing { color: #f59e0b; }

/* Metric chips */
.chip-row { display: flex; gap: 10px; flex-wrap: wrap; margin: 6px 0 12px 0; }
.chip {
    flex: 1 1 120px; text-align: center; border-radius: 12px; padding: 10px 8px;
    background: #f0f7ff; border: 1px solid var(--line);
}
.chip .num { font-size: 1.15rem; font-weight: 800; color: #1268b3; }
.chip .lbl { font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); }

/* Alerts */
[data-testid="stAlert"] { border-radius: 12px; }

/* Uploader */
[data-testid="stFileUploader"] {
    background: #f8fbff; border: 1px dashed #74afe3;
    border-radius: 14px; padding: 6px;
}

/* Make the "Upload Document / ID / Certificate" uploader label white */
[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] label p {
    color: #102a43 !important;
}

/* Buttons */
.stButton > button, [data-testid="stDownloadButton"] > button {
    background: linear-gradient(90deg, #0d67b5, #2196e5);
    color: white; border: none; border-radius: 10px;
    font-weight: 700; padding: 0.5rem 1.2rem;
}
.stButton > button:hover { filter: brightness(1.15); color: white; }

/* Expanders */
[data-testid="stExpander"] {
    background: var(--panel); border: 1px solid var(--line); border-radius: 12px;
}

/* Images */
[data-testid="stImage"] img { border-radius: 12px; border: 1px solid var(--line); }

/* Dataframe */
[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; border: 1px solid var(--line); }

/* Native fallback bars */
.native-bar { background: #deebf7; border-radius: 6px; height: 10px; }

hr { border-color: var(--line); }
</style>
"""
st.markdown(THEME_CSS, unsafe_allow_html=True)

st.markdown(
    """
    <div class="hero">
      <h1>🛡️ Multi-Document Forensic Analyzer</h1>
      <p>AI tamper detection · OCR extraction · QR/barcode decoding · portrait verification — in one pass.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


def card_open(tag, title, subtitle=""):
    sub = f'<p class="card-sub">{subtitle}</p>' if subtitle else ""
    st.markdown(
        f'<span class="section-tag">{tag}</span>'
        f'<div class="card"><p class="card-title">{title}</p>{sub}',
        unsafe_allow_html=True,
    )


def card_close():
    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------- 1. CNN model
@st.cache_resource
def load_tamper_cnn():
    model_path = "models/tamper_detector.h5"
    if os.path.exists(model_path):
        try:
            return tf.keras.models.load_model(model_path, compile=False)
        except Exception as e:
            st.sidebar.warning(f"Tamper model could not be loaded: {e}")
            return None
    return None


tamper_cnn = load_tamper_cnn()


# ------------------------------------------------- 2. Barcode / QR (defensive)
def scan_barcodes_and_qr(img_rgb):
    annotated = img_rgb.copy()
    payloads = []

    try:
        qr_detector = cv2.QRCodeDetector()
        retval, decoded_info, points, _ = qr_detector.detectAndDecodeMulti(img_rgb)
        if retval and points is not None:
            for text, pts in zip(decoded_info, points):
                if text and text.strip():
                    payloads.append({"type": "QR_CODE", "data": text.strip()})
                    cv2.polylines(annotated, [pts.astype(int)], True, (255, 0, 255), 3)
    except Exception:
        pass

    try:
        barcode_detector = cv2.barcode.BarcodeDetector()
        ok, decoded_info, decoded_type, points = barcode_detector.detectAndDecode(img_rgb)
        if ok and points is not None:
            for text, btype, pts in zip(decoded_info, decoded_type, points):
                if text and text.strip():
                    payloads.append({"type": f"BARCODE_{btype}", "data": text.strip()})
                    cv2.polylines(annotated, [pts.astype(int)], True, (0, 255, 255), 3)
    except Exception:
        pass

    return payloads, annotated


# ------------------------------------------------------- 3. ELA + AI inference
def perform_ela_and_ai_predict(pil_img, model, quality=90):
    buffer = io.BytesIO()
    rgb_img = pil_img.convert("RGB")
    rgb_img.save(buffer, "JPEG", quality=quality)
    buffer.seek(0)
    compressed_img = Image.open(buffer).convert("RGB")

    ela_img = ImageChops.difference(rgb_img, compressed_img)
    extrema = ela_img.getextrema()
    max_diff = max([ex[1] for ex in extrema]) or 1
    ela_enhanced = ImageEnhance.Brightness(ela_img).enhance(255.0 / max_diff)

    tamper_probability = None
    if model is not None:
        try:
            # Match whatever input size the trained model expects
            shape = model.input_shape
            th = shape[1] if shape[1] else 128
            tw = shape[2] if shape[2] else 128
            ela_resized = ela_enhanced.resize((tw, th))
            tensor = np.expand_dims(np.asarray(ela_resized, dtype="float32") / 255.0, axis=0)
            pred = model.predict(tensor, verbose=0)
            tamper_probability = float(np.ravel(pred)[0])
        except Exception as e:
            st.sidebar.warning(f"CNN inference failed: {e}")

    if tamper_probability is None:
        # Statistical fallback so the score is never a fake 0%
        ela_np = np.asarray(ela_enhanced.convert("L"), dtype="float32") / 255.0
        tamper_probability = float(np.clip(ela_np.std() * 2.2, 0.0, 1.0))

    return ela_enhanced, tamper_probability, (model is not None)


# ------------------------------------------- 4. Portrait extraction (FIXED)
def _clamp_box(x, y, w, h, W, H):
    x = max(0, min(int(x), W - 1))
    y = max(0, min(int(y), H - 1))
    w = max(1, min(int(w), W - x))
    h = max(1, min(int(h), H - y))
    return x, y, w, h


def extract_portrait(img_rgb, min_confidence=0.30):
    """
    Fixes vs. the original version:
      * DeepFace expects BGR arrays -> convert once (RGB input silently hurt detection).
      * enforce_detection=False returns a whole-image pseudo-face with confidence 0;
        we now filter on confidence instead of accepting it as a portrait.
      * off_x was hard-coded to 0, so the bottom-left quadrant box was misplaced.
      * scaled coordinates are divided by scale BEFORE adding the offset, then clamped.
      * best (highest-confidence, plausible aspect ratio) face wins instead of the first hit.
    """
    H, W = img_rgb.shape[:2]
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    y_split = int(H * 0.45)
    x_split = int(W * 0.55)
    regions = [
        (img_bgr, 0, 0, 1.0),
        (img_bgr[y_split:, :], 0, y_split, 1.0),
        (img_bgr[y_split:, :x_split], 0, y_split, 2.0),
        (img_bgr[:y_split, :x_split], 0, 0, 2.0),
        (img_bgr[:, x_split:], x_split, 0, 2.0),
    ]

    best = None  # (confidence, box)
    for backend in ["retinaface", "mtcnn", "ssd", "opencv"]:
        for region, off_x, off_y, scale in regions:
            if region is None or region.size == 0:
                continue
            test_img = region if scale == 1.0 else cv2.resize(
                region, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
            )
            try:
                results = DeepFace.extract_faces(
                    img_path=test_img,
                    detector_backend=backend,
                    enforce_detection=False,
                    align=False,
                )
            except Exception:
                continue

            for r in results:
                conf = float(r.get("confidence", 0) or 0)
                if conf > 1.0:
                    conf = conf / 100.0
                area = r.get("facial_area", {}) or {}
                fw = int(area.get("w", 0) / scale)
                fh = int(area.get("h", 0) / scale)
                if fw < 20 or fh < 20 or conf < min_confidence:
                    continue
                # reject full-frame "faces" and absurd aspect ratios
                if fw * fh > 0.75 * region.shape[0] * region.shape[1]:
                    continue
                ratio = fh / float(fw)
                if ratio < 0.7 or ratio > 2.0:
                    continue

                fx = int(area.get("x", 0) / scale) + off_x
                fy = int(area.get("y", 0) / scale) + off_y
                fx, fy, fw, fh = _clamp_box(fx, fy, fw, fh, W, H)
                if best is None or conf > best[0]:
                    best = (conf, (fx, fy, fw, fh))

        if best is not None and best[0] >= 0.85:
            break  # confident enough, skip slower backends

    if best is None:
        return None, None, 0.0

    conf, (fx, fy, fw, fh) = best
    pad_x, pad_y = int(fw * 0.15), int(fh * 0.20)
    cx, cy, cw, ch = _clamp_box(fx - pad_x, fy - pad_y, fw + 2 * pad_x, fh + 2 * pad_y, W, H)
    crop = img_rgb[cy:cy + ch, cx:cx + cw]
    if crop.size == 0:
        return None, None, 0.0
    return crop, (fx, fy, fw, fh), conf


# ------------------------------------------ 5. Classifier & regex extraction
def classify_and_extract(raw_text):
    text_upper = raw_text.upper()
    extracted_fields = {}

    if any(k in text_upper for k in ["DEGREE", "CERTIFICATE", "TRANSCRIPT", "GRADE"]):
        doc_type = "Academic Certificate / Transcript"
        requires_face = False
        name_m = re.search(r"(?:NAME|THIS IS TO CERTIFY THAT|STUDENT)\s*[:\-\s]+\s*([A-Za-z\s]+)", raw_text, re.I)
        id_m = re.search(r"(?:REG(?:ISTRATION)?|ROLL|ENROLMENT|ID)\s*(?:NO|NUMBER)?[\s\:\.\=]*([0-9A-Za-z\-]+)", raw_text, re.I)
        inst_m = re.search(r"(?:UNIVERSITY|COLLEGE|INSTITUTE|BOARD)\s*[:\-\s]*([A-Za-z\s]+)", raw_text, re.I)
        extracted_fields["Holder Name"] = name_m.group(1).splitlines()[0].strip() if name_m else "Not detected"
        extracted_fields["Credential / Roll No"] = id_m.group(1) if id_m else "Not detected"
        extracted_fields["Institution"] = inst_m.group(1).splitlines()[0].strip() if inst_m else "Not detected"

    elif any(k in text_upper for k in ["DRIVING", "DRIVER", "LICENSE", "LICENCE", "DL NO"]):
        doc_type = "Driver's License"
        requires_face = True
        dl_m = re.search(r"(?:DL|LICEN[CS]E\s*NO)[\s\:\.\=]*([A-Z0-9\-\s]{8,20})", raw_text, re.I)
        name_m = re.search(r"(?:NAME)[\s\:\.\=]*([A-Za-z\s]+)", raw_text, re.I)
        dob_m = re.search(r"(?:DOB|BIRTH)[\s\:\.\=]*([0-9\/\-\.]{8,10})", raw_text, re.I)
        extracted_fields["License Number"] = dl_m.group(1).strip() if dl_m else "Not detected"
        extracted_fields["Holder Name"] = name_m.group(1).splitlines()[0].strip() if name_m else "Not detected"
        extracted_fields["Date of Birth"] = dob_m.group(1) if dob_m else "Not detected"

    elif any(k in text_upper for k in ["PASSPORT", "REPUBLIC", "NATIONALITY", "SURNAME", "P<"]):
        doc_type = "Passport Document"
        requires_face = True
        pass_m = re.search(r"(?:PASSPORT\s*(?:NO|NUMBER)?)[\s\:\.\=]*([A-Z0-9]{7,10})", raw_text, re.I)
        name_m = re.search(r"(?:SURNAME|GIVEN\s*NAME|NAME)[\s\:\.\=]*([A-Za-z\s]+)", raw_text, re.I)
        extracted_fields["Passport No"] = pass_m.group(1) if pass_m else "Not detected"
        extracted_fields["Holder Name"] = name_m.group(1).splitlines()[0].strip() if name_m else "Not detected"

    elif any(k in text_upper for k in ["UNIQUE IDENTIFICATION", "GOVERNMENT OF INDIA", "AADHAAR", "UIDAI"]):
        doc_type = "National Identity Card"
        requires_face = True
        dob_m = re.search(r"(?:DOB|Birth|Year of Birth)[\s\:\.\=]*([0-9\/\-\.]{4,10})", raw_text, re.I)
        gender_m = re.search(r"\b(FEMALE|MALE|TRANSGENDER)\b", text_upper)
        name_m = re.search(r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s*\n.*(?:DOB|Birth)", raw_text)
        extracted_fields["Document Authority"] = "Unique Identification Authority of India"
        extracted_fields["Date of Birth / Year"] = dob_m.group(1) if dob_m else "Not detected"
        extracted_fields["Gender"] = gender_m.group(1).title() if gender_m else "Not detected"
        extracted_fields["Holder Name"] = name_m.group(1).strip() if name_m else "Not detected"

    else:
        doc_type = "Student / Organization ID Card"
        requires_face = True
        name_m = re.search(r"(?:Name|NAME)\s*[:\-\s]+\s*([A-Za-z\s]+)", raw_text)
        id_m = re.search(r"(?:Roll\s*No|Roll\s*Number|Roll|Reg|ID|No)[\s\:\.\=]*([0-9A-Za-z]{6,20})", raw_text, re.I)
        if not id_m:
            id_m = re.search(r"\b([0-9]{10,14})\b", raw_text)
        dept_m = re.search(r"(?:Course|Department|Dept)[\s\:\.\=]*([A-Za-z\.\s]+)", raw_text, re.I)
        batch_m = re.search(r"(?:Batch)[\s\:\.\=]*([0-9]{4}[\s\-–]+[0-9]{4})", raw_text, re.I)
        extracted_fields["Holder Name"] = name_m.group(1).splitlines()[0].strip() if name_m else "Not detected"
        extracted_fields["Roll / Reg Number"] = id_m.group(1).strip() if id_m else "Not detected"
        extracted_fields["Course / Dept"] = dept_m.group(1).splitlines()[0].strip() if dept_m else "Not detected"
        if batch_m:
            extracted_fields["Batch"] = batch_m.group(1).strip()

    return doc_type, requires_face, extracted_fields


# ------------------------------------------------------------------- charts
PALETTE = {"good": "#16803c", "warn": "#d97706", "bad": "#c62828", "muted": "#7891aa", "info": "#1769c2"}

# Compact, light-themed plotly defaults so charts read as dashboard widgets
PLOTLY_BASE = dict(
    height=210,
    margin=dict(t=34, b=6, l=6, r=6),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#294c6d", size=11),
    title_font=dict(size=13),
    showlegend=False,
)


def draw_pie(container, title, labels, values, colors):
    values = [max(float(v), 0.0001) for v in values]
    if CHART_ENGINE == "plotly":
        fig = px.pie(names=labels, values=values, hole=0.6, color_discrete_sequence=colors)
        fig.update_traces(textinfo="percent", textfont_size=10, sort=False,
                          marker=dict(line=dict(color="#ffffff", width=2)))
        fig.update_layout(title=title, **PLOTLY_BASE)
        container.plotly_chart(fig, use_container_width=True)
    elif CHART_ENGINE == "matplotlib":
        fig, ax = plt.subplots(figsize=(2.6, 2.0))
        fig.patch.set_alpha(0)
        ax.set_facecolor("none")
        ax.pie(values, labels=None, colors=colors, autopct="%1.0f%%",
               textprops={"color": "#294c6d", "fontsize": 8},
               wedgeprops={"width": 0.45, "edgecolor": "#ffffff"})
        ax.set_title(title, color="#102a43", fontsize=9)
        container.pyplot(fig)
        plt.close(fig)
        container.caption(" · ".join(str(l) for l in labels))
    else:
        container.markdown(f"**{title}**")
        total = sum(values)
        for label, value, color in zip(labels, values, colors):
            container.markdown(
                f"<div style='margin-bottom:6px;font-size:0.82rem'>{label} — {value/total*100:.0f}%"
                f"<div class='native-bar'>"
                f"<div style='width:{value/total*100:.0f}%;background:{color};height:10px;"
                f"border-radius:6px'></div></div></div>",
                unsafe_allow_html=True,
            )


def draw_signals(container, ocr_score, face_score, barcode_score, tamper_prob):
    title = "Verification Signals"
    labels = ["OCR quality", "Face confidence", "QR / Barcode", "Integrity (1 - tamper)"]
    values = [ocr_score * 100, face_score * 100, barcode_score * 100, (1 - tamper_prob) * 100]
    colors = [PALETTE["good"] if v >= 60 else PALETTE["warn"] if v >= 30 else PALETTE["bad"] for v in values]
    if CHART_ENGINE == "plotly":
        fig = go.Figure(go.Bar(x=values, y=labels, orientation="h", marker_color=colors,
                               text=[f"{v:.0f}%" for v in values], textposition="outside",
                               textfont=dict(size=10)))
        fig.update_layout(title=title, xaxis_range=[0, 120],
                          xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
                          yaxis=dict(showgrid=False, tickfont=dict(size=10)),
                          **{k: v for k, v in PLOTLY_BASE.items() if k != "showlegend"})
        container.plotly_chart(fig, use_container_width=True)
    elif CHART_ENGINE == "matplotlib":
        fig, ax = plt.subplots(figsize=(2.8, 2.0))
        fig.patch.set_alpha(0)
        ax.set_facecolor("none")
        ax.barh(labels, values, color=colors)
        ax.set_xlim(0, 120)
        ax.tick_params(colors="#294c6d", labelsize=7)
        ax.set_title(title, color="#102a43", fontsize=9)
        for spine in ax.spines.values():
            spine.set_visible(False)
        for i, v in enumerate(values):
            ax.text(v + 2, i, f"{v:.0f}%", va="center", fontsize=7, color="#294c6d")
        fig.tight_layout()
        container.pyplot(fig)
        plt.close(fig)
    else:
        container.markdown(f"**{title}**")
        for label, value in zip(labels, values):
            container.markdown(f"<div style='font-size:0.82rem'>{label} — {value:.0f}%</div>",
                               unsafe_allow_html=True)
            container.progress(min(int(value), 100))


def draw_gauge(container, trust_score):
    title = "Composite Trust Score"
    pct = trust_score * 100
    if CHART_ENGINE == "plotly":
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pct,
            number={"suffix": "%", "font": {"size": 22, "color": "#102a43"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 0, "tickcolor": "rgba(0,0,0,0)"},
                "bar": {"color": PALETTE["info"], "thickness": 0.28},
                "bgcolor": "rgba(0,0,0,0)",
                "steps": [
                    {"range": [0, 40], "color": "rgba(220,38,38,0.30)"},
                    {"range": [40, 70], "color": "rgba(245,158,11,0.30)"},
                    {"range": [70, 100], "color": "rgba(22,163,74,0.30)"},
                ],
            },
        ))
        fig.update_layout(title=title, **PLOTLY_BASE)
        container.plotly_chart(fig, use_container_width=True)
    elif CHART_ENGINE == "matplotlib":
        color = PALETTE["good"] if pct >= 70 else PALETTE["warn"] if pct >= 40 else PALETTE["bad"]
        fig, ax = plt.subplots(figsize=(2.6, 1.7), subplot_kw={"projection": "polar"})
        fig.patch.set_alpha(0)
        ax.set_facecolor("none")
        ax.set_theta_offset(3.14159)
        ax.set_theta_direction(-1)
        ax.set_thetamin(0)
        ax.set_thetamax(180)
        ax.barh(1, 3.14159, color="rgba(148,163,184,0.2)", height=0.5)
        ax.barh(1, 3.14159 * pct / 100, color=color, height=0.5)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_title(f"{title} — {pct:.0f}%", color="#102a43", fontsize=9)
        container.pyplot(fig)
        plt.close(fig)
    else:
        container.markdown(f"**{title}** — {pct:.0f}%")
        container.progress(min(int(pct), 100))


def render_field_rows(extracted_data):
    """Extracted attributes rendered as sleek key/value rows instead of plain markdown."""
    rows = []
    for key, val in extracted_data.items():
        missing = val == "Not detected"
        cls = "field-val missing" if missing else "field-val"
        rows.append(
            f'<div class="field-row"><span class="field-key">{key}</span>'
            f'<span class="{cls}">{val}</span></div>'
        )
    st.markdown("".join(rows), unsafe_allow_html=True)


# --------------------------------------------------------------- app state
if "doc_logs" not in st.session_state:
    st.session_state.doc_logs = []

# ------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### ⚙️ Controls")
    st.caption("Upload a document, then tune preprocessing here.")

uploaded_file = st.file_uploader(
    "📄 Upload Document / ID / Certificate",
    type=["jpg", "jpeg", "png"],
    help="Clear, well-lit scans give the best OCR and face-detection results.",
)

if uploaded_file is not None:
    original_pil = Image.open(uploaded_file)
    original_pil = original_pil.convert("RGBA") if original_pil.mode == "P" else original_pil

    with st.sidebar:
        st.markdown("---")
        st.markdown("#### Image Preprocessing")
        rotation_angle = st.radio("Manual Rotate (Degrees):", [0, 90, 180, 270], index=0, horizontal=True)
        upscale = st.checkbox("Upscale small images (helps face detection)", value=True)

    image = original_pil.rotate(-rotation_angle, expand=True) if rotation_angle else original_pil

    img_array = np.array(image)
    if img_array.ndim == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif img_array.shape[2] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)

    # Small scans are the #1 cause of failed face detection on ID cards
    if upscale and max(img_array.shape[:2]) < 900:
        factor = 900.0 / max(img_array.shape[:2])
        img_array = cv2.resize(img_array, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)

    processed_pil = Image.fromarray(img_array)

    # ---- analysis pipeline (identical to previous version) ----------------
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    resized_gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    with st.spinner("🔍 Parsing document text..."):
        try:
            raw_text = pytesseract.image_to_string(resized_gray, config="--psm 6").strip()
            if len(raw_text) < 15:
                raw_text = pytesseract.image_to_string(resized_gray).strip()
        except Exception as e:
            raw_text = ""
            st.error(f"Tesseract OCR is not available: {e}")

    doc_type, requires_face, extracted_data = classify_and_extract(raw_text)

    barcodes, barcode_overlay = scan_barcodes_and_qr(img_array)

    annotated_img = barcode_overlay.copy()
    with st.spinner("🙂 Locating portrait..."):
        extracted_face, face_box, face_conf = extract_portrait(img_array)
    face_detected = extracted_face is not None

    if face_detected:
        fx, fy, fw, fh = face_box
        cv2.rectangle(annotated_img, (fx, fy), (fx + fw, fy + fh), (0, 200, 0), 3)
        cv2.putText(annotated_img, f"Portrait {face_conf*100:.0f}%", (fx, max(25, fy - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)

    ela_result_img, tamper_prob, model_used = perform_ela_and_ai_predict(processed_pil, tamper_cnn)

    total_fields = max(1, len(extracted_data))
    valid_fields = sum(1 for v in extracted_data.values() if v != "Not detected")
    ocr_score = min(1.0, len(raw_text) / 250.0)
    face_score = face_conf if face_detected else 0.0
    barcode_score = 1.0 if barcodes else 0.0
    field_score = valid_fields / total_fields

    weights = {"field": 0.35, "ocr": 0.15, "face": 0.25 if requires_face else 0.0,
               "barcode": 0.05, "integrity": 0.20}
    total_w = sum(weights.values())
    trust_score = (
        field_score * weights["field"] + ocr_score * weights["ocr"] +
        face_score * weights["face"] + barcode_score * weights["barcode"] +
        (1 - tamper_prob) * weights["integrity"]
    ) / total_w

    if tamper_prob > 0.60:
        verdict, verdict_badge, badge_type = "TAMPER_DETECTED", f"AI FRAUD ALERT: tamper probability {tamper_prob*100:.1f}%", "error"
    elif requires_face and not face_detected:
        verdict, verdict_badge, badge_type = "PHOTO_MISSING", "REJECTED: required portrait not detected", "warning"
    elif valid_fields >= 2:
        verdict, verdict_badge, badge_type = "AUTHENTIC_AND_VERIFIED", f"GENUINE / VERIFIED ({doc_type.upper()})", "success"
    else:
        verdict, verdict_badge, badge_type = "MANUAL_REVIEW", "MANUAL REVIEW: low extraction confidence", "warning"

    if not model_used:
        st.sidebar.info("No trained CNN found at models/tamper_detector.h5 — using ELA statistics fallback.")

    # ---- quick stat chips -------------------------------------------------
    st.markdown(
        f"""
        <div class="chip-row">
          <div class="chip"><div class="num">{trust_score*100:.0f}%</div><div class="lbl">Trust</div></div>
          <div class="chip"><div class="num">{tamper_prob*100:.0f}%</div><div class="lbl">Tamper Risk</div></div>
          <div class="chip"><div class="num">{valid_fields}/{len(extracted_data)}</div><div class="lbl">Fields Found</div></div>
          <div class="chip"><div class="num">{face_score*100:.0f}%</div><div class="lbl">Face Conf.</div></div>
          <div class="chip"><div class="num">{len(barcodes)}</div><div class="lbl">Codes</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---- main work area: canvas | summary --------------------------------
    col_canvas, col_summary = st.columns([3, 2], gap="large")

    with col_canvas:
        card_open("Visual Forensics", "Document Canvas", f"Classified as: {doc_type}")
        st.image(annotated_img, use_container_width=True)
        card_close()

        with st.expander("🧪 ELA Tamper Heatmap", expanded=False):
            st.image(ela_result_img, use_container_width=True,
                     caption=f"Tamper risk: {tamper_prob*100:.1f}% — bright regions indicate resaved/edited areas")

    with col_summary:
        card_open("Verdict", "Extraction & Verification")
        {"success": st.success, "error": st.error, "warning": st.warning}[badge_type](verdict_badge)
        render_field_rows(extracted_data)

        if barcodes:
            st.markdown("**Embedded Data (QR / Barcode)**")
            for b in barcodes:
                st.info(f"**{b['type']}** — `{b['data'][:100]}`")

        if requires_face:
            if face_detected:
                fc1, fc2 = st.columns([1, 2])
                fc1.image(extracted_face, width=110)
                fc2.markdown(f"**Portrait isolated**\n\nConfidence: `{face_conf*100:.0f}%`")
            else:
                st.warning("No valid portrait found. Try rotating the image or uploading a sharper scan.")
        card_close()

        with st.expander("📝 Raw OCR Text"):
            st.text(raw_text if raw_text else "No text extracted.")

    # ---- compact dashboard widgets ---------------------------------------
    card_open("Analytics", "Analysis Dashboard", "Compact forensic signal overview")
    d1, d2, d3, d4 = st.columns(4, gap="medium")
    with d1:
        draw_gauge(st, trust_score)
    with d2:
        draw_pie(st, "Tamper Probability", ["Authentic", "Tamper"],
                 [1 - tamper_prob, tamper_prob], [PALETTE["good"], PALETTE["bad"]])
    with d3:
        draw_pie(st, "Field Coverage", ["Extracted", "Missing"],
                 [valid_fields, max(0, len(extracted_data) - valid_fields)],
                 [PALETTE["info"], PALETTE["muted"]])
    with d4:
        draw_signals(st, ocr_score, face_score, barcode_score, tamper_prob)
    card_close()

    if st.button("📥 Log Document Verification"):
        st.session_state.doc_logs.append({
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Document_Type": doc_type,
            "Fields_Extracted": str(extracted_data),
            "Face_Confidence": f"{face_score*100:.1f}%",
            "AI_Tamper_Probability": f"{tamper_prob*100:.2f}%",
            "Trust_Score": f"{trust_score*100:.1f}%",
            "Verification_Verdict": verdict,
        })
        st.success("Document verification logged successfully.")

else:
    st.markdown(
        """
        <div class="card" style="text-align:center; padding:48px 20px;">
          <p style="font-size:2.4rem; margin:0;">🗂️</p>
          <p class="card-title">No document loaded yet</p>
          <p class="card-sub">Upload an ID card, passport, license or certificate above to start the forensic scan.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ------------------------------------------------------------- audit trail
if st.session_state.doc_logs:
    st.markdown("---")
    card_open("History", "Document Audit Trail", f"{len(st.session_state.doc_logs)} verification(s) recorded this session")
    df_audit = pd.DataFrame(st.session_state.doc_logs)
    st.dataframe(df_audit, use_container_width=True, hide_index=True)

    a1, a2, a3 = st.columns([1, 1, 2], gap="medium")
    cycle = [PALETTE["good"], PALETTE["bad"], PALETTE["warn"], PALETTE["info"], PALETTE["muted"]]

    with a1:
        verdict_counts = df_audit["Verification_Verdict"].value_counts()
        draw_pie(st, "Verdict Mix", list(verdict_counts.index), list(verdict_counts.values),
                 [cycle[i % len(cycle)] for i in range(len(verdict_counts))])

    with a2:
        type_counts = df_audit["Document_Type"].value_counts()
        draw_pie(st, "Doc Type Mix", list(type_counts.index), list(type_counts.values),
                 [cycle[(i + 3) % len(cycle)] for i in range(len(type_counts))])

    with a3:
        st.markdown("<div style='height:70px'></div>", unsafe_allow_html=True)
        st.download_button(
            label="⬇️ Download Universal Audit Log (CSV)",
            data=df_audit.to_csv(index=False).encode("utf-8"),
            file_name="universal_document_verification_log.csv",
            mime="text/csv",
        )
    card_close()
    # ------------------------------------------------------------- final fake-detection result
    st.markdown("---")
    if verdict == "TAMPER_DETECTED":
        final_label = "🚨 FAKE / TAMPERED DOCUMENT DETECTED"
        final_color = "#dc2626"
    elif verdict == "AUTHENTIC_AND_VERIFIED":
        final_label = "✅ GENUINE DOCUMENT"
        final_color = "#16a34a"
    else:
        final_label = "⚠️ UNCERTAIN — MANUAL REVIEW REQUIRED"
        final_color = "#f59e0b"

    st.markdown(
        f"""
        <div style="text-align:center; padding:22px; border-radius:16px;
                    background:{final_color}22; border:2px solid {final_color}; margin-top:10px;">
          <p style="font-size:1.6rem; font-weight:800; color:{final_color}; margin:0;">
            {final_label}
          </p>
          <p style="color:#47708f; margin-top:6px;">
            Final Trust Score: {trust_score*100:.1f}% &nbsp;|&nbsp;
            Tamper Probability: {tamper_prob*100:.1f}%
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Also print to the terminal/console where you ran `streamlit run app.py`
    print(
        f"[FAKE DETECTION RESULT] "
        f"Document Type: {doc_type} | Verdict: {verdict} | "
        f"Trust Score: {trust_score*100:.1f}% | "
        f"Tamper Probability: {tamper_prob*100:.1f}%"
    )
