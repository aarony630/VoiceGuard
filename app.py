"""
VoiceGuard — Unified Call Screening System (Gradio)
Pipeline: Robocall Detection → Voice Identity Verification
"""

import numpy as np
import librosa
import joblib
import soundfile as sf
import tempfile
from pathlib import Path
from scipy.spatial.distance import cosine
import gradio as gr

try:
    from resemblyzer import VoiceEncoder, preprocess_wav
    RESEMBLYZER_OK = True
except ImportError:
    RESEMBLYZER_OK = False

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_FILE         = "robocall_detector.pkl"
ENROLLMENT_DIR     = "enrollment"
SAMPLE_RATE        = 16000
N_MFCC             = 40
MAX_DURATION       = 30
VOICE_THRESHOLD    = 0.70
ROBOCALL_THRESHOLD = 0.75   # raised from 0.5 to reduce false positives on mic input

# ── Load models once at startup ───────────────────────────────────────────────
print("Loading voice encoder…")
encoder  = VoiceEncoder() if RESEMBLYZER_OK else None
profiles = {}

def reload_profiles():
    global profiles
    profiles = {}
    root = Path(ENROLLMENT_DIR)
    if not root.exists() or encoder is None:
        return
    for person_dir in sorted(d for d in root.iterdir() if d.is_dir()):
        files = sorted(f for ext in ("*.wav","*.mp3") for f in person_dir.glob(ext))
        if not files:
            continue
        embeddings = [encoder.embed_utterance(preprocess_wav(str(f))) for f in files]
        profiles[person_dir.name] = np.mean(embeddings, axis=0)
    print(f"Enrolled: {list(profiles.keys())}")

reload_profiles()

# ── Feature extraction (matches ML.py exactly) ────────────────────────────────
def extract_features(y, sr):
    max_samples = MAX_DURATION * sr
    if len(y) > max_samples:
        y = y[:max_samples]

    mfccs      = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    mfcc_mean  = np.mean(mfccs, axis=1)
    mfcc_std   = np.std(mfccs, axis=1)
    delta_mean = np.mean(librosa.feature.delta(mfccs), axis=1)

    cent = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    bw   = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    roll = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    zcr  = np.mean(librosa.feature.zero_crossing_rate(y))

    try:
        f0, voiced_flag, _ = librosa.pyin(y, fmin=50, fmax=500, sr=sr)
        voiced_f0    = f0[voiced_flag] if np.any(voiced_flag) else np.array([0.0])
        pitch_mean   = float(np.mean(voiced_f0))
        pitch_std    = float(np.std(voiced_f0))
        voiced_ratio = float(np.mean(voiced_flag))
    except Exception:
        pitch_mean = pitch_std = voiced_ratio = 0.0

    return np.concatenate([
        mfcc_mean, mfcc_std, delta_mean,
        [cent, bw, roll, zcr, pitch_mean, pitch_std, voiced_ratio]
    ])


def check_robocall(audio_path):
    if not Path(MODEL_FILE).exists():
        return False, 0.0, f"⚠️ No model found at {MODEL_FILE}"
    try:
        bundle   = joblib.load(MODEL_FILE)
        clf, scaler = bundle["model"], bundle["scaler"]
        y, sr    = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
        features = scaler.transform([extract_features(y, sr)])
        prob     = clf.predict_proba(features)[0]
        is_robo  = prob[1] >= ROBOCALL_THRESHOLD
        return is_robo, float(prob[1]), None
    except Exception as e:
        return False, 0.0, f"Robo check error: {e}"


def check_voice_match(audio_path):
    if encoder is None:
        return False, "Unknown", 0.0, "resemblyzer not installed"
    if not profiles:
        return False, "Unknown", 0.0, "No enrolled profiles found"
    try:
        test_emb = encoder.embed_utterance(preprocess_wav(audio_path))
        sims     = {name: 1 - cosine(prof, test_emb) for name, prof in profiles.items()}
        best     = max(sims, key=sims.get)
        return sims[best] >= VOICE_THRESHOLD, best, float(sims[best]), sims
    except Exception as e:
        return False, "Unknown", 0.0, f"Voice match error: {e}"


# ── Shared pipeline logic ─────────────────────────────────────────────────────
def run_pipeline(audio_path, log):
    def emit(msg):
        log.append(msg)

    # Step 1: Robocall detection
    emit("🤖 STEP 1: Robocall Detection")
    emit("   Model: Random Forest + MFCC + pitch features")

    is_robo, robo_conf, err = check_robocall(audio_path)

    if err:
        emit(f"   ⚠️  {err}")
        is_robo, robo_conf = False, 0.0
    else:
        emit(f"   Robocall confidence : {robo_conf:.1%}")
        emit(f"   Threshold           : {ROBOCALL_THRESHOLD:.0%}")
        emit(f"   Result              : {'🔴 ROBOCALL' if is_robo else '🟢 HUMAN'}")
    emit("")

    if is_robo:
        emit("🚫 DECISION: BLOCKED — Robocall detected")
        emit("   Call will never reach the customer.")
        return "BLOCKED_ROBO", dict(robo_conf=robo_conf, is_robo=True, sim=0.0, name="", matched=False)

    # Step 2: Voice identity
    emit("🧬 STEP 2: Voice Identity Verification")
    emit("   Model: resemblyzer deep embeddings + cosine similarity")

    matched, name, sim, sims = check_voice_match(audio_path)

    if isinstance(sims, dict):
        for person, s in sorted(sims.items(), key=lambda x: -x[1]):
            marker = " ← best match" if person == name else ""
            emit(f"   {person:20s}: {s:.1%}{marker}")
    else:
        emit(f"   {sims}")

    emit(f"   Threshold           : {VOICE_THRESHOLD:.0%}")
    emit(f"   Result              : {'🟢 VERIFIED' if matched else '🔴 NOT VERIFIED'}")
    emit("")

    details = dict(robo_conf=robo_conf, is_robo=False, sim=sim, name=name, matched=matched)

    if matched:
        emit(f"✅ DECISION: CONNECTED — Verified caller: {name} ({sim:.1%})")
        return "CONNECTED", details
    else:
        emit(f"🚫 DECISION: BLOCKED — Caller not in verified employee database")
        return "BLOCKED_UNKNOWN", details


# ── Gradio handler functions ──────────────────────────────────────────────────
def screen_live_call(audio):
    log = []
    log.append("📞 Incoming call — screening pipeline started")
    log.append("─" * 48)

    if audio is None:
        return verdict_html("ERROR", msg="No audio recorded"), pipeline_html("idle"), "No audio received"

    sr, data = audio
    if data.dtype != np.float32:
        data = data.astype(np.float32)
        if np.abs(data).max() > 1.0:
            data = data / 32768.0
    if data.ndim > 1:
        data = data.mean(axis=1)

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, data, sr)
    log.append(f"🎙️  Recorded: {len(data)/sr:.1f}s at {sr}Hz")
    log.append("")

    verdict, details = run_pipeline(tmp.name, log)
    state = {"BLOCKED_ROBO": "robo", "BLOCKED_UNKNOWN": "unknown", "CONNECTED": "connected"}.get(verdict, "idle")
    return verdict_html(verdict, **details), pipeline_html(state), "\n".join(log)


def screen_robocall_file(robo_file):
    log = []
    log.append("📞 Incoming call — ROBOCALL DEMO MODE")
    log.append("─" * 48)

    if robo_file is None:
        return verdict_html("ERROR", msg="No file uploaded"), pipeline_html("idle"), "No file uploaded"

    log.append(f"🎙️  Pre-recorded file: {Path(robo_file).name}")
    log.append("")

    verdict, details = run_pipeline(robo_file, log)
    state = {"BLOCKED_ROBO": "robo", "BLOCKED_UNKNOWN": "unknown", "CONNECTED": "connected"}.get(verdict, "idle")
    return verdict_html(verdict, **details), pipeline_html(state), "\n".join(log)


# ── HTML helpers ──────────────────────────────────────────────────────────────
# Palette: navy=#0c1a3e  panel=#102044  border=#1e3460  sky=#5bb8d4  orange=#f7941d
NAVY   = "#0c1a3e"
PANEL  = "#f4f8ff"
BORDER = "#ccddf0"
SKY    = "#5bb8d4"
ORANGE = "#f7941d"
MUTED  = "#6a8aaa"
FONT   = "'Open Sans', sans-serif"

def verdict_html(verdict, name="", sim=0.0, robo_conf=0.0, is_robo=False, matched=False, msg=""):
    configs = {
        "BLOCKED_ROBO":    ("#ef4444", "🤖", "CALL BLOCKED",    f"Robocall detected — {robo_conf:.1%} confidence",                    "#fef2f2"),
        "BLOCKED_UNKNOWN": ("#f59e0b", "🚫", "CALL BLOCKED",    f"Caller not in verified database — best match: {name} ({sim:.1%})", "#fef2f2"),
        "CONNECTED":       ("#10b981", "✅", "CALL CONNECTED",   f"Verified: {name} — {sim:.1%} voice similarity",                    "#f0fdf4"),
        "ERROR":           ("#6b7280", "⚠️", "ERROR",           msg,                                                                  PANEL),
    }
    color, icon, title, sub, bg = configs.get(verdict, configs["ERROR"])

    bars = ""
    if verdict != "ERROR":
        robo_pct   = int(robo_conf * 100)
        robo_color = "#3b82f6"
        bars += f"""
        <div style="margin-top:8px;">
            <div style="display:flex;justify-content:space-between;align-items:center;
                        font-size:11px;color:{MUTED};font-weight:600;margin-bottom:5px;">
                <span>ROBOCALL CONFIDENCE</span>
                <span style="color:{robo_color};">{robo_conf:.1%}</span>
            </div>
            <div style="background:{PANEL};border-radius:4px;height:5px;overflow:hidden;">
                <div style="width:{robo_pct}%;height:100%;
                            background:linear-gradient(90deg,{robo_color}99,{robo_color});
                            border-radius:4px;"></div>
            </div>
        </div>"""
        if verdict != "BLOCKED_ROBO":
            sim_pct   = int(sim * 100)
            sim_color = "#f59e0b"
            bars += f"""
        <div style="margin-top:10px;">
            <div style="display:flex;justify-content:space-between;align-items:center;
                        font-size:11px;color:{MUTED};font-weight:600;margin-bottom:5px;">
                <span>VOICE SIMILARITY</span>
                <span style="color:{sim_color};">{sim:.1%}</span>
            </div>
            <div style="background:{PANEL};border-radius:4px;height:5px;overflow:hidden;">
                <div style="width:{sim_pct}%;height:100%;
                            background:linear-gradient(90deg,{sim_color}99,{sim_color});
                            border-radius:4px;"></div>
            </div>
        </div>"""

    metrics_block = f"""
        <div style="border-top:1px solid {color}28;margin-top:20px;padding-top:16px;">{bars}</div>
    """ if bars else ""

    return f"""
    <div class="verdict-card" style="
        background:linear-gradient(135deg,{bg},{color}0a);
        border:1.5px solid {color}66;
        border-radius:14px; padding:28px 28px; text-align:center;
        font-family:{FONT};
        box-shadow: 0 0 36px {color}18, inset 0 0 48px {color}06;
        margin: 4px 0;
    ">
        <div style="font-size:44px; margin-bottom:8px; line-height:1;">{icon}</div>
        <div style="font-size:20px; font-weight:800; color:{color}; letter-spacing:2px;
                    text-shadow:0 0 14px {color}55;">{title}</div>
        <div style="font-size:12px; color:{MUTED}; margin-top:7px; font-weight:400;">{sub}</div>
        {metrics_block}
    </div>"""


def pipeline_html(state="idle"):
    steps = [("📞","INCOMING"), ("🎙️","RECORD"), ("🤖","ROBO CHECK"), ("🧬","VOICE ID"), ("⚖️","DECISION")]
    BASE = "#2e539cff"  # default blue for inactive steps
    color_map = {
        "idle":      [BASE]*5,
        "robo":      ["#10b981", "#10b981", "#ef4444", BASE, "#ef4444"],
        "unknown":   ["#10b981", "#10b981", "#10b981", "#f59e0b", "#f59e0b"],
        "connected": ["#10b981"]*5,
    }
    cols = color_map.get(state, color_map["idle"])

    items = ""
    for i, (icon, label) in enumerate(steps):
        c   = cols[i]
        lit = c != BASE
        glow = f"box-shadow:0 0 12px {c}88;" if lit else ""
        items += f"""
        <div style="text-align:center;flex:1;min-width:0;">
            <div style="font-size:19px;background:{c}16;border:1.5px solid {c};border-radius:50%;
                        width:44px;height:44px;line-height:42px;margin:0 auto 6px;{glow}">{icon}</div>
            <div style="font-size:9px;color:{c};font-family:{FONT};font-weight:700;
                        letter-spacing:1px;white-space:nowrap;">{label}</div>
        </div>
        {"<div style='flex:0.25;text-align:center;color:" + BASE + ";font-size:13px;padding-bottom:18px;'>──▶</div>" if i < len(steps)-1 else ""}
        """
    return f"""
    <div class="pipeline-wrap" style="background:transparent;
                border:1px solid {BORDER};border-radius:12px;
                padding:14px 20px;display:flex;align-items:center;
                font-family:{FONT};gap:2px;">
        {items}
    </div>"""


# ── Gradio UI ─────────────────────────────────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Open+Sans:wght@400;600;700;800&display=swap');

/* ── Base ── */
body, .gradio-container {
    background: #ffffff !important;
    color: #0c1a3e !important;
    font-family: 'Open Sans', sans-serif !important;
}

/* ── Buttons ── */
.gr-button {
    font-family: 'Open Sans', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: 0.5px !important;
    border-radius: 8px !important;
    transition: all 0.2s ease !important;
}
.gr-button-primary {
    background: linear-gradient(135deg, #f7941d, #d97a0a) !important;
    border: none !important;
    color: #0c1a3e !important;
    box-shadow: 0 4px 16px #f7941d44 !important;
}
.gr-button-primary:hover {
    background: linear-gradient(135deg, #ffa535, #f7941d) !important;
    box-shadow: 0 4px 28px #f7941d88 !important;
    transform: translateY(-1px) !important;
}
.gr-button-secondary {
    background: #ffffff !important;
    border: 1px solid #ccddf0 !important;
    color: #5bb8d4 !important;
}
.gr-button-secondary:hover {
    background: #f0f9fc !important;
    box-shadow: 0 0 14px #5bb8d433 !important;
    transform: translateY(-1px) !important;
}

/* ── Labels & headings ── */
h1, h2, h3, label, .label-wrap {
    font-family: 'Open Sans', sans-serif !important;
    color: #0c1a3e !important;
    font-weight: 600 !important;
}

/* ── Text inputs & textareas ── */
textarea, input[type="text"] {
    background: #ffffff !important;
    color: #0c1a3e !important;
    font-family: 'Open Sans', sans-serif !important;
    font-size: 13px !important;
    border: 1px solid #ccddf0 !important;
    border-radius: 8px !important;
    transition: border-color 0.2s !important;
}
textarea:focus, input:focus {
    border-color: #f7941d !important;
    box-shadow: 0 0 0 2px #f7941d22 !important;
    outline: none !important;
}

/* ── Tabs ── */
.tab-nav { border-bottom: 1px solid #ccddf0 !important; background:#ffffff !important; }
.tab-nav button {
    font-family: 'Open Sans', sans-serif !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    color: #6a8aaa !important;
    letter-spacing: 0.3px !important;
    transition: color 0.2s !important;
}
.tab-nav button:hover  { color: #0c1a3e !important; }
.tab-nav button.selected {
    color: #f7941d !important;
    border-bottom: 2px solid #f7941d !important;
    background:#ffffff !important;
}

/* ── Block panels ── */
.gr-block, .gr-panel { background: #ffffff !important; border-color: #ccddf0 !important; }

/* ── Audio component ── */
.gr-audio { background: #ffffff !important; border: 1px solid #ccddf0 !important; border-radius: 8px !important; }

/* ── Force textbox internals to stay white (fix blue fill in Pipeline Log) ── */
.gr-textbox, .gr-textbox textarea, .gr-textbox input, .gr-form, .gr-box {
    background: #ffffff !important;
    color: #0c1a3e !important;
    border-color: #ccddf0 !important;
    box-shadow: none !important;
}

/* ── Equal-height columns ── */
.gradio-row { align-items: stretch !important; }
.gradio-row > .gradio-column {
    display: flex !important;
    flex-direction: column !important;
}
.gradio-row > .gradio-column > div { flex: 1 !important; }

/* ── Hide Gradio footer ── */
footer { display: none !important; }
"""

enrolled_names = ", ".join(profiles.keys()) or "none"
n_enrolled = len(profiles)

HEADER = f"""
<div class="app-header" style="text-align:center;padding:36px 0 18px;font-family:'Open Sans',sans-serif;
            background:linear-gradient(180deg,#ffffff 0%,#f4f8ff 100%);">
    <div style="font-size:11px;letter-spacing:4px;color:{ORANGE};margin-bottom:10px;
                text-transform:uppercase;font-weight:700;">
        AI-Powered Call Screening System
    </div>
    <div style="font-size:48px;font-weight:800;letter-spacing:3px;
                background:linear-gradient(90deg,#17396d 0%,#0f5fa6 55%,#0b2a52 100%);
                -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
                line-height:1.1;">
        VOICEGUARD
    </div>
    <div class="app-header-sub" style="font-size:12px;color:{NAVY};margin-top:12px;letter-spacing:1.5px;font-weight:600;">
        ROBOCALL DETECTION &nbsp;·&nbsp; VOICE IDENTITY VERIFICATION &nbsp;·&nbsp; ACCESS DECISION
    </div>
    <div class="app-header-online" style="margin:18px auto 0;display:inline-flex;align-items:center;gap:10px;
                background:#ffffff;border:1px solid #10b98155;border-radius:24px;
                padding:8px 22px;font-size:12px;color:#10b981;font-family:'Open Sans',sans-serif;font-weight:600;">
        <span style="width:7px;height:7px;border-radius:50%;background:#10b981;
                     display:inline-block;box-shadow:0 0 8px #10b981;flex-shrink:0;"></span>
        SYSTEM ONLINE &nbsp;·&nbsp; {n_enrolled} enrolled caller(s):
        <span class="app-header-online-name" style="color:{NAVY};font-weight:700;">&nbsp;{enrolled_names}</span>
    </div>
</div>
"""

# ── Reusable sub-components ───────────────────────────────────────────────────
HINT_STYLE = f"background:#ffffff;border:1px solid {BORDER};border-radius:8px;" \
             f"padding:12px 18px;margin:10px 0;font-family:'Open Sans',sans-serif;" \
             f"font-size:12px;color:{MUTED};font-weight:400;line-height:1.6;"

IDLE_CARD_LIVE = (
    f'<div class="idle-card" style="background:#ffffff;border:1px solid {BORDER};border-radius:14px;'
    f'box-sizing:border-box;width:100%;height:100%;min-height:160px;'
    f'display:flex;flex-direction:column;align-items:center;justify-content:center;'
    f'font-family:{FONT};color:{MUTED};padding:24px;">'
    f'<div style="font-size:34px;margin-bottom:10px;">📡</div>'
    f'<div style="font-size:12px;letter-spacing:1.5px;font-weight:700;">AWAITING CALL</div>'
    f'</div>'
)

IDLE_CARD_FILE = (
    f'<div class="idle-card" style="background:#ffffff;border:1px solid {BORDER};border-radius:14px;'
    f'box-sizing:border-box;width:100%;height:100%;min-height:160px;'
    f'display:flex;flex-direction:column;align-items:center;justify-content:center;'
    f'font-family:{FONT};color:{MUTED};padding:24px;">'
    f'<div style="font-size:34px;margin-bottom:10px;">📁</div>'
    f'<div style="font-size:11px;letter-spacing:1.5px;font-weight:700;">AWAITING FILE</div>'
    f'</div>'
)

with gr.Blocks(css=CSS, title="VoiceGuard") as demo:
    gr.HTML(HEADER)
    pipeline_display = gr.HTML(value=pipeline_html("idle"), label="Pipeline Status")

    with gr.Tabs():

        # ── Tab 1: Live mic ───────────────────────────────────────────────────
        with gr.Tab("📞  Live Call Screening"):
            gr.HTML(f"""
            <div class="hint-box" style="{HINT_STYLE}border-left:3px solid {SKY};">
                <span style="color:{SKY};font-weight:700;">HOW TO USE</span>
                &nbsp;·&nbsp;
                Click the microphone &rarr; speak your name &amp; purpose &rarr; hit Stop &rarr; press Screen Call
            </div>""")
            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    mic_input  = gr.Audio(sources=["microphone"], type="numpy", label="🎙️ Caller Voice Input")
                    screen_btn = gr.Button("▶  SCREEN THIS CALL", variant="primary", size="lg")
                with gr.Column(scale=1):
                    verdict_out = gr.HTML(value=IDLE_CARD_LIVE)
            log_out = gr.Textbox(
                label="📋 Pipeline Log",
                lines=10, interactive=False,
                placeholder="Detailed screening log will appear here after processing…"
            )
            screen_btn.click(fn=screen_live_call, inputs=[mic_input], outputs=[verdict_out, pipeline_display, log_out])

        # ── Tab 2: Robocall file demo ─────────────────────────────────────────
        with gr.Tab("🤖  Robocall Demo"):
            gr.HTML(f"""
            <div class="hint-box" style="{HINT_STYLE}border-left:3px solid {ORANGE};">
                <span style="color:{ORANGE};font-weight:700;">DEMO MODE</span>
                &nbsp;·&nbsp;
                Upload a pre-recorded robocall .wav/.mp3 &rarr; run through the detection pipeline &rarr; expect BLOCKED
            </div>""")
            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    robo_file  = gr.Audio(sources=["upload"], type="filepath", label="📁 Upload Robocall Audio")
                    robo_btn   = gr.Button("🤖  RUN ROBOCALL THROUGH PIPELINE", variant="secondary", size="lg")
                with gr.Column(scale=1):
                    robo_verdict = gr.HTML(value=IDLE_CARD_FILE)
            robo_log = gr.Textbox(
                label="📋 Pipeline Log",
                lines=10, interactive=False,
                placeholder="Detailed screening log will appear here after processing…"
            )
            robo_btn.click(fn=screen_robocall_file, inputs=[robo_file], outputs=[robo_verdict, pipeline_display, robo_log])

        # ── Tab 3: Enrolled callers ───────────────────────────────────────────
        with gr.Tab("👤  Enrolled Callers"):
            enrolled_text = "\n".join([f"✅  {n}" for n in profiles]) or \
                "⚠️  No enrolled callers found.\n\nAdd folders to the enrollment/ directory:\n  enrollment/PersonName/sample.wav"
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Textbox(
                        value=enrolled_text,
                        label="Verified callers in database",
                        lines=9, interactive=False
                    )
                with gr.Column(scale=1):
                    gr.HTML(f"""
                    <div class="enroll-card" style="background:#ffffff;border:1px solid {BORDER};border-radius:10px;
                                padding:20px 24px;font-family:'Open Sans',sans-serif;
                                font-size:12px;color:{MUTED};line-height:1.7;">
                        <div style="color:{SKY};font-size:13px;font-weight:700;
                                    letter-spacing:0.5px;margin-bottom:14px;">
                            ENROLLING A NEW CALLER
                        </div>
                        <div style="margin-bottom:10px;">
                            <span style="color:{ORANGE};font-weight:700;">①</span>
                            &ensp;Create a directory:<br>
                            &emsp;<code style="color:#10b981;font-size:12px;">enrollment/PersonName/</code>
                        </div>
                        <div style="margin-bottom:10px;">
                            <span style="color:{ORANGE};font-weight:700;">②</span>
                            &ensp;Add voice samples:<br>
                            &emsp;<code style="color:#10b981;font-size:12px;">.wav</code>
                            or <code style="color:#10b981;font-size:12px;">.mp3</code>
                            &nbsp;(3+ recommended)
                        </div>
                        <div style="margin-bottom:20px;">
                            <span style="color:{ORANGE};font-weight:700;">③</span>
                            &ensp;Restart the app
                        </div>
                        <div style="border-top:1px solid {BORDER};padding-top:16px;">
                            <div style="color:{SKY};font-size:13px;font-weight:700;
                                        letter-spacing:0.5px;margin-bottom:10px;">
                                DETECTION THRESHOLDS
                            </div>
                            <div style="margin-bottom:6px;">
                                Robocall block:
                                <span style="color:#ef4444;font-weight:700;">&nbsp;≥ 75% confidence</span>
                            </div>
                            <div>
                                Voice match:
                                <span style="color:#10b981;font-weight:700;">&nbsp;≥ 75% similarity</span>
                            </div>
                        </div>
                    </div>""")

if __name__ == "__main__":
    demo.launch()
