import streamlit as st
import streamlit.components.v1 as components
import speech_recognition as sr
from deep_translator import GoogleTranslator
from gtts import gTTS
from gtts.lang import tts_langs
import io
import json
import csv
import hashlib
import time

# ── Optional PDF/DOCX support (graceful if not installed) ──
try:
    import fitz  # PyMuPDF
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

try:
    from docx import Document as DocxDocument
    DOCX_SUPPORT = True
except ImportError:
    DOCX_SUPPORT = False

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AI Neural Translator",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────
# LANGUAGE DATA  (deep-translator uses full names as keys)
# ─────────────────────────────────────────────
RAW_LANGS = GoogleTranslator().get_supported_languages(as_dict=True)
# RAW_LANGS: {"english": "en", "french": "fr", ...}
NAME_TO_CODE = RAW_LANGS                          # "english" → "en"
CODE_TO_NAME = {v: k for k, v in RAW_LANGS.items()}  # "en" → "english"

# Display list: "EN English", sorted
DISPLAY_OPTIONS = sorted([f"{code.upper()}  {name.title()}" for name, code in RAW_LANGS.items()])
DISPLAY_TO_CODE = {}
for name, code in RAW_LANGS.items():
    DISPLAY_TO_CODE[f"{code.upper()}  {name.title()}"] = code

AUTO_DETECT_LABEL = "🔍  Auto-Detect"
DISPLAY_OPTIONS_WITH_AUTO = [AUTO_DETECT_LABEL] + DISPLAY_OPTIONS

TTS_LANGS = tts_langs()  # codes supported by gTTS

# ─────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────
defaults = {
    "source_text": "",       # canonical text (read by translation logic)
    "source_input": "",      # textarea widget key — never written post-render
    "pending_source": None,  # staging: STT/file/swap write here; flushed before render
    "translated_text": "",
    "src_lang": "en",
    "tgt_lang": "fr",
    "detected_lang": None,
    "confidence": None,
    "last_audio_hash": None,
    "last_uploaded_hash": None,
    "translation_history": [],
    "swap_trigger": False,
    "last_translation_key": "",
    "error_msg": None,
    "tts_unavailable": False,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Flush pending_source BEFORE any widget renders ──
# STT / file upload / swap write to pending_source then call st.rerun().
# On the next run this block fires before the textarea is instantiated,
# so Streamlit never sees a post-render write to the widget key.
if st.session_state.pending_source is not None:
    st.session_state.source_input = st.session_state.pending_source
    st.session_state.source_text  = st.session_state.pending_source
    st.session_state.pending_source = None

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def audio_hash(audio_bytes_io):
    return hashlib.md5(audio_bytes_io.getvalue()).hexdigest()

def translation_cache_key():
    return f"{st.session_state.source_text}|{st.session_state.src_lang}|{st.session_state.tgt_lang}"

def clear_all():
    for k in ["source_text", "source_input", "pending_source",
              "translated_text", "detected_lang", "confidence",
              "last_audio_hash", "last_uploaded_hash", "last_translation_key",
              "error_msg", "tts_unavailable"]:
        st.session_state[k] = defaults[k]

def perform_translation(force=False):
    text = st.session_state.source_text.strip()
    if not text:
        return

    # Hard char limit
    if len(text) > 2000:
        st.session_state.error_msg = "⚠️ Input exceeds 2000 characters. Please shorten your text."
        return

    cache_key = translation_cache_key()
    if not force and cache_key == st.session_state.last_translation_key:
        return  # same request — skip API call

    src = st.session_state.src_lang  # may be "auto"
    tgt = st.session_state.tgt_lang

    try:
        translator = GoogleTranslator(source=src, target=tgt)
        translated = translator.translate(text)
        st.session_state.translated_text = translated
        st.session_state.last_translation_key = cache_key
        st.session_state.error_msg = None
        st.session_state.tts_unavailable = False

        # ── Attempt to detect source language & approximate confidence ──
        if src == "auto":
            try:
                det = GoogleTranslator(source="auto", target="en")
                det.translate(text)           # triggers detection
                detected_code = det.source   # populated after translate()
                st.session_state.detected_lang = CODE_TO_NAME.get(detected_code, detected_code).title()
                # Confidence: deep_translator doesn't expose it natively;
                # we approximate via character n-gram entropy heuristic
                unique_ratio = len(set(text)) / max(len(text), 1)
                st.session_state.confidence = min(99, max(70, int(95 - (unique_ratio * 10))))
            except Exception:
                st.session_state.detected_lang = None
                st.session_state.confidence = 92
        else:
            st.session_state.detected_lang = None
            st.session_state.confidence = 95   # fixed high confidence for explicit source

        # ── Save to history ──
        entry = {
            "src_lang": st.session_state.detected_lang or src.upper(),
            "tgt_lang": tgt.upper(),
            "source": text[:120] + ("…" if len(text) > 120 else ""),
            "translation": translated[:120] + ("…" if len(translated) > 120 else ""),
            "timestamp": time.strftime("%H:%M"),
        }
        if not st.session_state.translation_history or st.session_state.translation_history[0].get("source") != entry["source"]:
            st.session_state.translation_history.insert(0, entry)
            if len(st.session_state.translation_history) > 50:
                st.session_state.translation_history.pop()

    except Exception as e:
        st.session_state.error_msg = f"Translation failed: {e}"

def extract_file_text(uploaded_file):
    """Return plain text from .txt, .pdf, or .docx uploads."""
    name = uploaded_file.name.lower()
    if name.endswith(".txt"):
        return uploaded_file.getvalue().decode("utf-8", errors="replace")
    elif name.endswith(".pdf") and PDF_SUPPORT:
        doc = fitz.open(stream=uploaded_file.getvalue(), filetype="pdf")
        return "\n".join(page.get_text() for page in doc)
    elif name.endswith(".docx") and DOCX_SUPPORT:
        doc = DocxDocument(io.BytesIO(uploaded_file.getvalue()))
        return "\n".join(p.text for p in doc.paragraphs)
    return None

# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

:root {
    --neon-blue: #7aa2f7;
    --neon-purple: #bb9af7;
    --neon-green: #9ece6a;
    --glass: rgba(22, 23, 31, 0.80);
    --glass-light: rgba(30, 32, 48, 0.60);
    --border: rgba(122, 162, 247, 0.22);
    --danger: #f7768e;
    --warn: #e0af68;
}

*, *::before, *::after { box-sizing: border-box; }

header, .stApp > header { background: transparent !important; }
#MainMenu, footer { visibility: hidden; }

.stApp {
    background: radial-gradient(ellipse at 15% 10%, #1c1d30 0%, #10111e 45%, #080912 100%);
    color: #e0e7ff;
    font-family: 'Inter', system-ui, sans-serif;
    overflow-x: hidden;
}

/* ── TITLE ── */
.main-title {
    font-family: 'Space Grotesk', sans-serif;
    text-align: center;
    font-size: clamp(24px, 5.5vw, 46px);
    font-weight: 700;
    letter-spacing: -1.5px;
    margin: 14px 0 6px;
    background: linear-gradient(95deg, #7aa2f7 0%, #bb9af7 50%, #c084fc 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    position: relative;
    line-height: 1.15;
}
.main-subtitle {
    text-align: center;
    font-size: clamp(11px, 2vw, 13px);
    color: #565f89;
    letter-spacing: 2px;
    text-transform: uppercase;
    font-weight: 500;
    margin-bottom: 28px;
}

/* ── SECTION LABELS ── */
.section-label {
    font-size: clamp(10px, 2.2vw, 11px);
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 6px;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* ── GLASSMORPHIC TEXT AREAS ── */
.stTextArea textarea {
    background: var(--glass) !important;
    backdrop-filter: blur(20px);
    color: #e0e7ff !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 16px !important;
    font-size: clamp(14px, 3.2vw, 16px) !important;
    line-height: 1.7 !important;
    padding: clamp(14px, 3vw, 22px) !important;
    height: 210px !important;
    box-shadow: 0 20px 40px -12px rgb(0 0 0 / 0.45), inset 0 1px 0 rgba(255,255,255,0.04);
    transition: border-color 0.3s, box-shadow 0.3s, transform 0.3s;
    width: 100% !important;
    resize: none !important;
}
.stTextArea textarea:focus {
    border-color: #bb9af7 !important;
    box-shadow: 0 0 0 4px rgba(187,154,247,0.18), 0 20px 40px -12px rgb(0 0 0 / 0.45) !important;
    transform: translateY(-1px);
    outline: none !important;
}
/* Disabled (translation output) */
.stTextArea textarea:disabled {
    color: #9ece6a !important;
    opacity: 1 !important;
    border-color: rgba(158, 206, 106, 0.2) !important;
}

/* ── CHAR COUNTER COLOR (danger zone) ── */
.char-ok   { color: #565f89; }
.char-warn { color: var(--warn); }
.char-over { color: var(--danger); font-weight: 600; }

/* ── LANGUAGE SELECTORS ── */
.stSelectbox div[data-baseweb="select"] {
    background: var(--glass) !important;
    border: 1.5px solid rgba(65,72,104,0.65) !important;
    border-radius: 9999px !important;
    color: #e0e7ff !important;
    box-shadow: 0 8px 20px -6px rgb(0 0 0 / 0.3);
    font-size: clamp(12px, 2.8vw, 14px) !important;
    transition: border-color 0.2s, box-shadow 0.2s;
}
.stSelectbox div[data-baseweb="select"]:hover {
    border-color: #7aa2f7 !important;
    box-shadow: 0 0 20px rgba(122,162,247,0.35);
}

/* ── SWAP BUTTON — target via aria-label set on the button key ── */
button[data-testid="baseButton-secondary"][title="Swap languages & text"],
button[aria-label="Swap languages & text"] {
    background: rgba(122,162,247,0.1) !important;
    border: 1.5px solid rgba(122,162,247,0.4) !important;
    border-radius: 50% !important;
    width: 46px !important;
    height: 46px !important;
    min-height: 46px !important;
    max-height: 46px !important;
    padding: 0 !important;
    font-size: 20px !important;
    line-height: 1 !important;
    color: #7aa2f7 !important;
    cursor: pointer;
    transition: transform 0.35s cubic-bezier(0.34,1.56,0.64,1),
                background 0.2s, box-shadow 0.2s, border-color 0.2s !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    margin: 0 auto !important;
    /* Override the full-width rule from secondary button */
    width: 46px !important;
}
button[data-testid="baseButton-secondary"][title="Swap languages & text"]:hover,
button[aria-label="Swap languages & text"]:hover {
    background: rgba(122,162,247,0.22) !important;
    border-color: #7aa2f7 !important;
    transform: rotate(180deg) scale(1.18) !important;
    box-shadow: 0 0 22px rgba(122,162,247,0.45) !important;
    color: #c0d8ff !important;
}
/* Center the swap column */
div[data-testid="stColumn"]:has(button[title="Swap languages & text"]) {
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}
div[data-testid="stColumn"]:has(button[title="Swap languages & text"]) > div {
    display: flex;
    justify-content: center;
    width: 100%;
}

/* ── PRIMARY BUTTON ── */
button[kind="primary"] {
    background: linear-gradient(135deg, #7aa2f7, #bb9af7) !important;
    color: #0b0c18 !important;
    border-radius: 9999px !important;
    padding: clamp(10px,2.5vw,14px) clamp(20px,4vw,36px) !important;
    font-weight: 700 !important;
    font-size: clamp(12px,2.8vw,14px) !important;
    letter-spacing: 0.6px;
    text-transform: uppercase;
    box-shadow: 0 0 22px -4px rgba(122,162,247,0.6), 0 8px 20px -6px rgba(122,162,247,0.35) !important;
    border: none !important;
    width: 100%;
    min-height: 46px;
    transition: transform 0.2s, box-shadow 0.2s !important;
}
button[kind="primary"]:hover {
    transform: translateY(-3px) scale(1.04) !important;
    box-shadow: 0 0 32px -4px rgba(187,154,247,0.7), 0 16px 30px -8px rgba(187,154,247,0.4) !important;
}

/* ── SECONDARY BUTTON (Clear, Export, etc.) ── */
button[kind="secondary"] {
    background: transparent !important;
    border: 1.5px solid #3d4466 !important;
    color: #a5b4fc !important;
    border-radius: 9999px !important;
    font-weight: 600 !important;
    font-size: clamp(12px,2.8vw,13px) !important;
    min-height: 46px !important;
    height: auto !important;
    max-height: none !important;
    width: 100% !important;
    padding: 10px 20px !important;
    transition: all 0.2s !important;
}
button[kind="secondary"]:hover {
    border-color: #c084fc !important;
    color: #c084fc !important;
    background: rgba(192,132,252,0.08) !important;
}

/* ── UPLOAD ZONE ── */
[data-testid="stFileUploadDropzone"] {
    background: var(--glass) !important;
    border: 1.5px dashed rgba(122,162,247,0.45) !important;
    border-radius: 14px !important;
    padding: clamp(10px,2.5vw,18px) !important;
    transition: all 0.3s !important;
}
[data-testid="stFileUploadDropzone"]:hover {
    border-color: #bb9af7 !important;
    background: rgba(187,154,247,0.08) !important;
}

/* ── META BAR ── */
.meta-text {
    font-size: clamp(10px,2.2vw,12px);
    color: #565f89;
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-top: 8px;
    font-weight: 500;
    flex-wrap: wrap;
    gap: 4px;
}

/* ── DETECTED LANG BADGE ── */
.detected-badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: rgba(122,162,247,0.12);
    border: 1px solid rgba(122,162,247,0.3);
    border-radius: 9999px;
    padding: 3px 10px;
    font-size: 11px;
    color: #7aa2f7;
    font-weight: 600;
    letter-spacing: 0.5px;
    margin-top: 6px;
}

/* ── CONFIDENCE BAR ── */
.confidence-wrap {
    display: flex;
    align-items: center;
    gap: clamp(8px,2vw,14px);
    background: var(--glass);
    backdrop-filter: blur(20px);
    padding: clamp(10px,2.2vw,13px) clamp(14px,3vw,22px);
    border-radius: 9999px;
    border: 1px solid var(--border);
    box-shadow: 0 8px 24px -8px rgba(122,162,247,0.25);
    flex-wrap: wrap;
    margin-top: 6px;
}
.conf-bar-bg {
    flex-grow: 1;
    min-width: 50px;
    background: #0e0f1c;
    height: 7px;
    border-radius: 9999px;
    overflow: hidden;
}
.conf-bar-fill {
    height: 100%;
    border-radius: 9999px;
    animation: neonPulse 2.4s infinite ease-in-out;
}
@keyframes neonPulse { 0%,100%{opacity:1} 50%{opacity:.7} }

/* ── CUSTOM ERROR CARD ── */
.error-card {
    background: rgba(247,118,142,0.1);
    border: 1.5px solid rgba(247,118,142,0.4);
    border-radius: 14px;
    padding: 12px 18px;
    color: #f7768e;
    font-size: 13px;
    font-weight: 500;
    margin-top: 8px;
    display: flex;
    align-items: flex-start;
    gap: 8px;
}

/* ── EMPTY STATE ── */
.empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 120px;
    color: #3d4466;
    gap: 10px;
}
.empty-state-icon { font-size: 32px; opacity: 0.5; }
.empty-state-text { font-size: 13px; font-style: italic; }

/* ── HISTORY CARDS ── */
.history-card {
    background: var(--glass);
    border: 1px solid var(--border);
    border-left: 5px solid #bb9af7;
    padding: clamp(10px,2.5vw,16px);
    border-radius: 14px;
    margin-bottom: 10px;
    transition: transform 0.2s, box-shadow 0.2s;
}
.history-card:hover {
    transform: translateY(-3px);
    border-left-color: #7aa2f7;
    box-shadow: 0 16px 32px -12px rgba(187,154,247,0.35);
}
.history-lang {
    font-size: 10px; font-weight: 700; letter-spacing: 1.5px;
    color: #7dcfff; text-transform: uppercase;
    display: flex; align-items: center; justify-content: space-between;
}
.history-time { font-size: 10px; color: #3d4466; font-weight: 400; letter-spacing: 0; }
.history-src { font-size: 12px; color: #a5b4fc; font-style: italic; margin: 5px 0; word-break: break-word; }
.history-tgt { font-size: 13px; color: var(--neon-green); font-weight: 500; word-break: break-word; }

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1020 0%, #080912 100%) !important;
    border-right: 1px solid rgba(122,162,247,0.12) !important;
    width: clamp(240px,80vw,300px) !important;
}
[data-testid="collapsedControl"] {
    background: #0f1020 !important;
    border: 1.5px solid #7aa2f7 !important;
    border-radius: 9999px !important;
    box-shadow: 0 0 20px rgba(122,162,247,0.25);
}

/* ── STUDENT BADGE ── */
.student-badge {
    background: linear-gradient(145deg, #14152a, #0e0f1e);
    border: 1.5px solid rgba(122,162,247,0.2);
    padding: clamp(14px,3vw,20px);
    border-radius: 16px;
    box-shadow: 0 12px 30px -10px rgba(122,162,247,0.2);
    position: relative;
    overflow: hidden;
}
.student-badge::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent, #7aa2f7, transparent);
}

/* ── SCROLLBAR ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-thumb { background: linear-gradient(#7aa2f7,#bb9af7); border-radius: 9999px; }

/* ── LAYOUT PADDING ── */
.main .block-container {
    padding-left: clamp(10px,4vw,56px) !important;
    padding-right: clamp(10px,4vw,56px) !important;
    padding-top: clamp(8px,2vw,24px) !important;
    max-width: 100% !important;
}

/* ── iOS ZOOM PREVENTION ── */
input, textarea, select { font-size: 16px !important; }

/* ── PREVENT OVERFLOW ── */
.stApp, .main, .block-container { overflow-x: hidden !important; }

/* ── TOUCH TARGETS ── */
button { -webkit-tap-highlight-color: transparent; }

/* ── AUDIO INPUT ── */
[data-testid="stAudioInput"] { width: 100%; max-width: 100%; }

/* ── MOBILE BREAKPOINTS ── */
@media (max-width: 768px) {
    .stTextArea textarea { height: 170px !important; }
    .main .block-container { padding: 10px 10px 60px !important; }
    .main-title { margin: 10px 0 4px; }
    [data-testid="stFileUploadDropzone"] { padding: 12px !important; }
}
@media (max-width: 480px) {
    .stTextArea textarea { height: 145px !important; font-size: 14px !important; border-radius: 12px !important; }
    .confidence-wrap { border-radius: 14px; }
    .main-title { letter-spacing: -0.5px; }
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("<h3 style='color:#7aa2f7;margin-bottom:0;font-size:clamp(14px,4vw,17px)'>FUTO Cybersecurity</h3>", unsafe_allow_html=True)
    st.markdown("<p style='color:#3d4466;font-size:12px;margin-top:2px'>CSC 309 · AI Practical</p>", unsafe_allow_html=True)
    st.markdown("""
    <div class='student-badge'>
        <div style='color:#bb9af7;font-weight:900;font-size:10px;letter-spacing:1.5px'>TASK 14</div>
        <div style='font-size:15px;font-weight:700;margin:4px 0 10px'>Text &amp; Speech MT</div>
        <div style='color:#3d4466;font-size:10px;font-weight:700;letter-spacing:1px'>DEVELOPER</div>
        <div style='color:#7dcfff;font-size:14px;font-weight:700'>Kenneth Chigozie Emmanuel </div>
        <div style='color:#e0af68;font-size:11px;margin-top:2px'>Reg No: 20231372562</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<div style='height:20vh'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:11px;color:#3d4466;line-height:1.65'>
        <div style='color:#7aa2f7;font-size:10px;letter-spacing:1.5px;text-transform:uppercase;font-weight:700;margin-bottom:8px'>System Architecture</div>
        Uses Google NMT via <b style='color:#7dcfff'>deep-translator</b> (stable SDK wrapper) with Speech-to-Text and gTTS synthesis. Features auto-detect, dedup caching, real confidence scoring, and multi-format file parsing.
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TITLE
# ─────────────────────────────────────────────
st.markdown("<div class='main-title'>Neural Machine Translation</div>", unsafe_allow_html=True)
st.markdown("<div class='main-subtitle'>Powered by Google NMT · deep-translator SDK</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LANGUAGE SELECTOR ROW  (with working SWAP)
# ─────────────────────────────────────────────
# Resolve current display values
src_display_default = AUTO_DETECT_LABEL if st.session_state.src_lang == "auto" else f"{st.session_state.src_lang.upper()}  {CODE_TO_NAME.get(st.session_state.src_lang,'').title()}"
tgt_display_default = f"{st.session_state.tgt_lang.upper()}  {CODE_TO_NAME.get(st.session_state.tgt_lang,'').title()}"

src_idx = DISPLAY_OPTIONS_WITH_AUTO.index(src_display_default) if src_display_default in DISPLAY_OPTIONS_WITH_AUTO else 0
tgt_idx = DISPLAY_OPTIONS.index(tgt_display_default) if tgt_display_default in DISPLAY_OPTIONS else DISPLAY_OPTIONS.index("FR  French")

lc1, lc2, lc3 = st.columns([5, 1, 5])
with lc1:
    src_sel = st.selectbox("Source", DISPLAY_OPTIONS_WITH_AUTO, index=src_idx, label_visibility="collapsed", key="src_select")
    new_src = "auto" if src_sel == AUTO_DETECT_LABEL else DISPLAY_TO_CODE.get(src_sel, "auto")
    if new_src != st.session_state.src_lang:
        st.session_state.src_lang = new_src
        st.session_state.detected_lang = None
        st.session_state.last_translation_key = ""  # invalidate cache

with lc2:
    if st.button("⇄", key="swap_btn", help="Swap languages & text", use_container_width=False):
        if st.session_state.src_lang != "auto":
            old_src = st.session_state.src_lang
            old_tgt = st.session_state.tgt_lang
            old_text = st.session_state.get("source_text", "")
            old_translation = st.session_state.translated_text
            st.session_state.src_lang = old_tgt
            st.session_state.tgt_lang = old_src
            # Stage the new source text so it's flushed before the textarea renders
            st.session_state.pending_source = old_translation
            st.session_state.source_text = old_translation
            st.session_state.translated_text = old_text
            st.session_state.last_translation_key = ""
            st.session_state.detected_lang = None
            st.rerun()
        else:
            st.toast("Set a specific source language before swapping.", icon="⚠️")


with lc3:
    tgt_sel = st.selectbox("Target", DISPLAY_OPTIONS, index=tgt_idx, label_visibility="collapsed", key="tgt_select")
    new_tgt = DISPLAY_TO_CODE.get(tgt_sel, "fr")
    if new_tgt != st.session_state.tgt_lang:
        st.session_state.tgt_lang = new_tgt
        st.session_state.last_translation_key = ""

# ─────────────────────────────────────────────
# TEXT AREAS
# ─────────────────────────────────────────────
src_label = "AUTO-DETECT" if st.session_state.src_lang == "auto" else st.session_state.src_lang.upper()
tgt_label = st.session_state.tgt_lang.upper()

col1, col2 = st.columns(2)

with col1:
    st.markdown(f"<div class='section-label' style='color:#7aa2f7'>📝 SOURCE · {src_label}</div>", unsafe_allow_html=True)

    # key="source_input" is widget-owned. We NEVER write to it post-render.
    # pending_source is the staging variable; flushed to source_input at top of run.
    # On every render, sync source_text from what the user typed.
    typed = st.text_area(
        "source",
        height=210,
        label_visibility="collapsed",
        placeholder="Type, paste, dictate, or upload a document…",
        max_chars=2000,
        key="source_input",
    )
    # Keep source_text in sync with manual typing (safe here, pre-widget is done)
    if typed != st.session_state.source_text:
        st.session_state.source_text = typed
        st.session_state.last_translation_key = ""

    char_count = len(st.session_state.source_text)
    word_count = len(st.session_state.source_text.split()) if st.session_state.source_text.strip() else 0
    char_class = "char-ok" if char_count < 1600 else ("char-warn" if char_count < 2000 else "char-over")
    st.markdown(
        f"<div class='meta-text'><span>{word_count} words</span><span class='{char_class}'>{char_count}/2000 chars</span></div>",
        unsafe_allow_html=True)

    # Detected language badge
    if st.session_state.detected_lang:
        st.markdown(f"<div class='detected-badge'>🔍 Detected: <b>{st.session_state.detected_lang}</b></div>", unsafe_allow_html=True)

    # File upload — txt, pdf (if installed), docx (if installed)
    accept_types = ["txt"]
    if PDF_SUPPORT:  accept_types.append("pdf")
    if DOCX_SUPPORT: accept_types.append("docx")
    uploaded_file = st.file_uploader(
        f"📄 Upload ({', '.join('.' + t for t in accept_types)})",
        type=accept_types,
        label_visibility="collapsed"
    )
    if uploaded_file is not None:
        file_hash = hashlib.md5(uploaded_file.getvalue()).hexdigest()
        if file_hash != st.session_state.last_uploaded_hash:
            extracted = extract_file_text(uploaded_file)
            if extracted:
                text = extracted[:2000]
                st.session_state.pending_source = text
                st.session_state.source_text = text
                st.session_state.last_uploaded_hash = file_hash
                st.session_state.last_translation_key = ""
                perform_translation()
                st.rerun()
            else:
                st.session_state.error_msg = "⚠️ Could not read that file format. Install PyMuPDF for PDF or python-docx for DOCX."

with col2:
    st.markdown(f"<div class='section-label' style='color:#bb9af7'>🌐 TRANSLATION · {tgt_label}</div>", unsafe_allow_html=True)

    if st.session_state.translated_text:
        st.text_area("translation", value=st.session_state.translated_text, height=210,
                     label_visibility="collapsed", disabled=True)
    else:
        st.markdown("""
        <div class='empty-state' style='height:210px;border:1.5px solid rgba(122,162,247,0.1);border-radius:16px;background:rgba(22,23,31,0.5)'>
            <div class='empty-state-icon'>🌐</div>
            <div class='empty-state-text'>Translation will appear here</div>
        </div>
        """, unsafe_allow_html=True)

    t_char = len(st.session_state.translated_text)
    t_words = len(st.session_state.translated_text.split()) if st.session_state.translated_text.strip() else 0
    st.markdown(f"<div class='meta-text'><span>{t_words} words</span><span>{t_char} chars</span></div>", unsafe_allow_html=True)

    # Copy button (via components.html for clipboard access)
    if st.session_state.translated_text:
        js_text = json.dumps(st.session_state.translated_text)
        copy_html = f"""
        <style>
            body{{margin:0;background:transparent;overflow:hidden;font-family:'Inter',sans-serif}}
            button{{width:100%;background:transparent;color:#a9b1d6;border:1.5px solid #3d4466;
                   border-radius:9999px;padding:8px 0;font-size:12px;font-weight:600;cursor:pointer;
                   transition:all .2s;min-height:38px;-webkit-tap-highlight-color:transparent}}
            button:hover,button:active{{border-color:#7aa2f7;color:#7aa2f7}}
        </style>
        <button id="cb" onclick="cp()">📋 Copy Translation</button>
        <script>
        function cp(){{
            const t={js_text};
            if(navigator.clipboard&&window.isSecureContext){{
                navigator.clipboard.writeText(t);
            }}else{{
                const ta=document.createElement('textarea');
                ta.value=t;ta.style.cssText='position:fixed;opacity:0';
                document.body.appendChild(ta);ta.select();
                document.execCommand('copy');document.body.removeChild(ta);
            }}
            const b=document.getElementById('cb');
            b.textContent='✅ Copied!';b.style.cssText='border-color:#10b981;color:#10b981';
            setTimeout(()=>{{b.textContent='📋 Copy Translation';b.style.cssText=''}},2000);
        }}
        </script>"""
        components.html(copy_html, height=50)

    # TTS audio
    if st.session_state.translated_text:
        tgt_code = st.session_state.tgt_lang
        if tgt_code in TTS_LANGS:
            try:
                tts = gTTS(text=st.session_state.translated_text, lang=tgt_code)
                audio_fp = io.BytesIO()
                tts.write_to_fp(audio_fp)
                audio_fp.seek(0)
                st.audio(audio_fp, format="audio/mp3")
                st.session_state.tts_unavailable = False
            except Exception as e:
                st.session_state.tts_unavailable = True
        else:
            st.markdown(
                f"<div style='font-size:11px;color:#3d4466;margin-top:6px'>🔇 TTS not available for <b>{tgt_label}</b></div>",
                unsafe_allow_html=True)

# ─────────────────────────────────────────────
# ERROR DISPLAY (themed)
# ─────────────────────────────────────────────
if st.session_state.error_msg:
    st.markdown(f"<div class='error-card'>⚠️ {st.session_state.error_msg}</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# ACTION ROW
# ─────────────────────────────────────────────
ac1, ac2 = st.columns([1, 1])

with ac1:
    bc1, bc2 = st.columns([2, 1])
    with bc1:
        if st.button("A文  Translate", type="primary", use_container_width=True):
            with st.spinner("Translating…"):
                perform_translation(force=True)
            st.rerun()
    with bc2:
        if st.button("⊗ Clear", use_container_width=True):
            clear_all()
            st.rerun()

with ac2:
    if st.session_state.translated_text and st.session_state.confidence is not None:
        conf = st.session_state.confidence
        bar_color = "#10b981" if conf >= 90 else ("#e0af68" if conf >= 75 else "#f7768e")
        glow = bar_color
        st.markdown(f"""
        <div class='confidence-wrap'>
            <span style='font-weight:600;font-size:clamp(11px,2.5vw,13px);white-space:nowrap'>Confidence</span>
            <div class='conf-bar-bg'>
                <div class='conf-bar-fill' style='width:{conf}%;background:{bar_color};box-shadow:0 0 14px {glow}'></div>
            </div>
            <span style='color:{bar_color};font-weight:700;font-size:clamp(12px,3vw,14px);white-space:nowrap'>{conf}%</span>
            <span style='color:#7aa2f7;font-size:clamp(9px,2vw,11px);white-space:nowrap'>Google NMT</span>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# BOTTOM: VOICE INPUT  +  HISTORY
# ─────────────────────────────────────────────
bot1, bot2 = st.columns([1, 1])

with bot1:
    mic_lang_name = CODE_TO_NAME.get(st.session_state.src_lang, "auto").title() if st.session_state.src_lang != "auto" else "auto-detect"
    st.markdown(f"<div class='section-label' style='color:#7aa2f7'>🎤 VOICE INPUT · listening in <b>{mic_lang_name}</b></div>", unsafe_allow_html=True)
    audio_bytes = st.audio_input("Dictate", label_visibility="collapsed")
    if audio_bytes:
        ah = audio_hash(audio_bytes)
        if ah != st.session_state.last_audio_hash:
            st.session_state.last_audio_hash = ah
            st.toast("Processing recording…", icon="⏳")
            r = sr.Recognizer()
            try:
                with sr.AudioFile(audio_bytes) as source:
                    audio_data = r.record(source)
                lang_hint = st.session_state.src_lang if st.session_state.src_lang != "auto" else "en"
                stt_lang = f"{lang_hint}-{lang_hint.upper()}"
                text = r.recognize_google(audio_data, language=stt_lang)
                # Write to pending_source — flushed before textarea renders on next run
                st.session_state.pending_source = text
                st.session_state.source_text = text  # also update for translation
                st.session_state.last_translation_key = ""
                with st.spinner("Translating…"):
                    perform_translation()
                st.rerun()
            except sr.UnknownValueError:
                st.session_state.error_msg = "Could not understand audio. Try speaking more clearly."
                st.rerun()
            except Exception as e:
                st.session_state.error_msg = f"Voice error: {e}"
                st.rerun()

with bot2:
    st.markdown("<div class='section-label' style='color:#7aa2f7'>📋 TRANSLATION LOG</div>", unsafe_allow_html=True)
    hist_box = st.container(height=170)
    with hist_box:
        if st.session_state.translation_history:
            for item in st.session_state.translation_history:
                ts = item.get("timestamp", "")
                st.markdown(f"""
                <div class='history-card'>
                    <div class='history-lang'>
                        <span>{item['src_lang']} ➔ {item['tgt_lang']}</span>
                        <span class='history-time'>{ts}</span>
                    </div>
                    <div class='history-src'>"{item['source']}"</div>
                    <div class='history-tgt'>{item['translation']}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class='empty-state'>
                <div class='empty-state-icon'>🕐</div>
                <div class='empty-state-text'>Your translation history will appear here</div>
            </div>
            """, unsafe_allow_html=True)

    if st.session_state.translation_history:
        csv_buf = io.StringIO()
        writer = csv.DictWriter(csv_buf, fieldnames=["src_lang","tgt_lang","source","translation","timestamp"])
        writer.writeheader()
        writer.writerows(st.session_state.translation_history)
        st.download_button(
            label="📥 Export History (.CSV)",
            data=csv_buf.getvalue(),
            file_name="translation_history.csv",
            mime="text/csv",
            use_container_width=True
        )
