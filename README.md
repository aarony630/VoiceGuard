# VoiceGuard 
### AI-Powered Call Screening & Voice Authentication System

VoiceGuard is an AI-powered call screening system that protects customers by automatically verifying the identity of every incoming caller before connecting them. Acting as an invisible layer between callers and customers, the system first detects whether a call is from an automated robocall, then confirms whether the caller is a registered and trusted employee. Only verified callers are connected — everyone else is blocked.

---

## How It Works

<img width="541" height="691" alt="Group 1 (3)" src="https://github.com/user-attachments/assets/a641deb9-717b-476c-a935-c65ee12d7cf9" />


---

## Dataset
The model was trained using a combination of specialized datasets to ensure accurate classification between malicious robocalls and legitimate human speech.

### 1. Robocall & Scam Data
* **[Robocall Audio Dataset](https://github.com/wspr-ncsu/robocall-audio-dataset.git):** Provided by WSPR-NCSU, containing a wide variety of automated telemarketing and informational broadcasts.

### 2. Human Speech (Normal Calls)
* **[Normal Human Call Dataset](https://www.kaggle.com/datasets/divyanshsharma23/genrate-scam-call-dataset/data):** Used as the baseline for "Safe" calls to ensure the model can distinguish natural conversation flow from automated scripts.


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

