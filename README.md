# VoiceGuard 🛡️
### AI-Powered Call Screening & Voice Authentication System

VoiceGuard is an AI-powered call screening system that protects customers by automatically verifying the identity of every incoming caller before connecting them. Acting as an invisible layer between callers and customers, the system first detects whether a call is from an automated robocall, then confirms whether the caller is a registered and trusted employee. Only verified callers are connected — everyone else is blocked.

---

## How It Works

```
Incoming Call
     ↓
🎙️  Live audio captured — caller prompted to speak
     ↓
🤖  Stage 1: Robocall Detection
     → Robocall detected?  ❌ BLOCKED
     → Human?              ↓ continue
     ↓
🧬  Stage 2: Voice Identity Verification
     → Verified employee?  ✅ CALL CONNECTED
     → Unknown caller?     ❌ BLOCKED
```

---

## Features

- **Two-stage ML pipeline** — robocall detection followed by voice identity verification
- **Real-time screening** — live microphone input processed on the fly
- **Enrolled employee database** — enroll verified callers with a few voice samples
- **Robocall demo mode** — test the system with a pre-recorded robocall file
- **Clean Gradio UI** — browser-based interface, no installation complexity

---

## Project Structure

```
voice-biometrics-call/
│
├── app.py                    # Main Gradio app — unified screening pipeline
├── ML.py                     # Robocall detector — training & inference
├── voice_match.py            # Voice identity matching (standalone)
├── robocall_detector.pkl     # Pre-trained robocall detection model
├── requirements.txt          # Python dependencies
│
├── enrollment/               # Enrolled verified callers
│   └── PersonName/
│       ├── sample1.wav
│       └── sample2.wav
│
└── data/                     # Training data (not included)
    ├── robocalls/
    └── normal_calls/
```

---

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/JzZ404/voice-biometrics-call.git
cd voice-biometrics-call
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Enroll verified callers
Create a subfolder per person inside `enrollment/` and add `.wav` or `.mp3` voice samples:
```
enrollment/
  Joyce/
    sample1.wav
    sample2.wav
  Aaron/
    sample1.wav
```
More samples per person = better accuracy. Three or more is recommended.

### 4. Run the app
```bash
python app.py
```
The Gradio interface will open in your browser at `http://localhost:7860`

---

## Usage

| Tab | Description |
|-----|-------------|
| 📞 Live Call Screening | Record live mic input and run through the full pipeline |
| 🤖 Robocall Demo | Upload a pre-recorded robocall file to demonstrate blocking |
| 👤 Enrolled Callers | View currently enrolled verified employees |

---

## Training the Robocall Detector

A pre-trained model (`robocall_detector.pkl`) is included. To retrain from scratch:

```bash
# Clone the robocall dataset
git clone https://github.com/wspr-ncsu/robocall-audio-dataset data/robocalls

# Train the model
python ML.py train

# Test on an audio file
python ML.py test path/to/audio.wav
```

---

## Configuration

Key parameters can be adjusted in `app.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ROBOCALL_THRESHOLD` | `0.75` | Confidence required to flag as robocall |
| `VOICE_THRESHOLD` | `0.70` | Similarity score required to verify a caller |
| `SAMPLE_RATE` | `16000` | Audio sample rate (Hz) |

---

## Tech Stack

- **Robocall Detection** — Random Forest classifier (scikit-learn)
- **Feature Extraction** — MFCC, pitch, spectral features (librosa)
- **Voice Embeddings** — GE2E encoder (resemblyzer)
- **Similarity Matching** — Cosine distance (scipy)
- **Interface** — Gradio + Python

---

## Team

Built as a capstone project at the **University of Washington Global Innovation Exchange (GIX)**.

---

## License

MIT License
