"""
Robocall Detector - ML.py

Setup:
  1. Clone the robocall dataset:
       git clone https://github.com/wspr-ncsu/robocall-audio-dataset data/robocalls
  2. Install dependencies:
       pip install scikit-learn joblib datasets librosa tqdm
  3. Normal speech (negative class) — pick ONE:
       a) Add your own normal call recordings to data/normal_calls/
       b) The script will auto-download LibriSpeech from HuggingFace if the folder is empty

Usage:
  Train:   python ML.py train
  Test:    python ML.py test path/to/audio.mp3
"""

import sys
import subprocess
import io
import imageio_ffmpeg
import numpy as np
import librosa
import librosa.display
import joblib
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from pathlib import Path
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# --- Config ---
ROBOCALL_DIR     = "data/robocalls/audio-wav-16khz"  # cloned GitHub repo audio folder
NORMAL_DIR       = "data/normal_calls"               # your own normal call recordings
MODEL_FILE       = "robocall_detector.pkl"
SAMPLE_RATE      = 16000
N_MFCC           = 40
MAX_DURATION     = 30   # seconds to use per clip
MAX_TRAIN_SAMPLES = None  # set to None to use all samples


def load_audio(audio_path, sr=SAMPLE_RATE):
    """Load audio, piping m4a through ffmpeg in memory if needed."""
    if Path(audio_path).suffix.lower() == ".m4a":
        result = subprocess.run(
            [imageio_ffmpeg.get_ffmpeg_exe(), "-i", audio_path, "-f", "wav", "-ar", str(sr), "-ac", "1", "pipe:1"],
            capture_output=True, check=True
        )
        return librosa.load(io.BytesIO(result.stdout), sr=sr, mono=True)
    return librosa.load(audio_path, sr=sr, mono=True)


def plot_fingerprint(y, sr, audio_path, label, prob_robo):
    """Save an MFCC heatmap (audio fingerprint) for a given audio clip."""
    max_samples = MAX_DURATION * sr
    if len(y) > max_samples:
        y = y[:max_samples]

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)

    fig, ax = plt.subplots(figsize=(12, 5))
    img = librosa.display.specshow(mfccs, x_axis="time", sr=sr, ax=ax, cmap="magma")
    fig.colorbar(img, ax=ax, label="MFCC Amplitude")
    ax.set_title(f"Audio Fingerprint — {Path(audio_path).name}\n{label}  |  Robocall: {prob_robo:.1%}", fontsize=13)
    ax.set_ylabel("MFCC Coefficient")
    ax.set_xlabel("Time (s)")

    out = Path(audio_path).stem + "_fingerprint.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Fingerprint saved -> {out}")


def plot_confusion(y_test, y_pred):
    """Save a confusion matrix heatmap after training."""
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set(xticks=[0, 1], yticks=[0, 1],
           xticklabels=["Normal", "Robocall"],
           yticklabels=["Normal", "Robocall"],
           xlabel="Predicted", ylabel="Actual",
           title="Confusion Matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    fig.tight_layout()
    fig.savefig("confusion_matrix.png", dpi=150)
    plt.close(fig)
    print("Confusion matrix saved -> confusion_matrix.png")


def extract_features(y, sr):
    """Extract MFCC + spectral + pitch features from an audio array."""
    # Trim to max duration
    max_samples = MAX_DURATION * sr
    if len(y) > max_samples:
        y = y[:max_samples]

    # MFCCs — mean, std, and delta
    mfccs      = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    mfcc_mean  = np.mean(mfccs, axis=1)
    mfcc_std   = np.std(mfccs, axis=1)
    delta_mean = np.mean(librosa.feature.delta(mfccs), axis=1)

    # Spectral features
    cent = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    bw   = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    roll = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    zcr  = np.mean(librosa.feature.zero_crossing_rate(y))

    # Pitch features — TTS voices have unnaturally low pitch variation
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


def load_audio_folder(folder, label, limit=None):
    """Load all audio files from a folder and extract features."""
    folder = Path(folder)
    files  = sorted(list(folder.glob("*.wav")) + list(folder.glob("*.mp3")))
    if limit:
        files = files[:limit]

    X, y = [], []
    tag = "Robocall" if label == 1 else "Normal"
    for f in tqdm(files, desc=f"{tag} features"):
        try:
            audio, sr = librosa.load(str(f), sr=SAMPLE_RATE, mono=True)
            if len(audio) < sr:  # skip clips under 1 second
                continue
            X.append(extract_features(audio, sr))
            y.append(label)
        except Exception as e:
            print(f"  Skipped {f.name}: {e}")
    return X, y


def load_normal_from_huggingface(n_samples):
    """Download normal speech from LibriSpeech via HuggingFace as negative class."""
    print("Downloading LibriSpeech test-clean from HuggingFace...")
    try:
        import soundfile as sf
        import io
        from datasets import load_dataset, Audio
        ds = load_dataset("librispeech_asr", "clean", split="test", trust_remote_code=True, streaming=True)
        ds = ds.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE, decode=False))
        X, y = [], []
        for i, sample in enumerate(tqdm(ds, desc="Normal features", total=n_samples)):
            if i >= n_samples:
                break
            try:
                audio_entry = sample["audio"]
                # Handle decoded array or raw bytes
                if isinstance(audio_entry, dict) and "array" in audio_entry:
                    audio = np.array(audio_entry["array"], dtype=np.float32)
                    sr    = audio_entry["sampling_rate"]
                elif isinstance(audio_entry, dict) and "bytes" in audio_entry and audio_entry["bytes"]:
                    audio, sr = sf.read(io.BytesIO(audio_entry["bytes"]))
                    audio = audio.astype(np.float32)
                elif isinstance(audio_entry, dict) and "path" in audio_entry and audio_entry["path"]:
                    audio, sr = librosa.load(audio_entry["path"], sr=SAMPLE_RATE, mono=True)
                else:
                    continue
                if sr != SAMPLE_RATE:
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
                X.append(extract_features(audio, SAMPLE_RATE))
                y.append(0)
            except Exception as e:
                print(f"  Skipped sample {i}: {e}")
        return X, y
    except Exception as e:
        print(f"HuggingFace download failed: {e}")
        return [], []


def train():
    # --- Positive class: robocalls ---
    if not Path(ROBOCALL_DIR).exists():
        print(f"\nERROR: Robocall audio folder not found: {ROBOCALL_DIR}")
        print("Clone the dataset first:")
        print("  git clone https://github.com/wspr-ncsu/robocall-audio-dataset data/robocalls")
        return

    X_rob, y_rob = load_audio_folder(ROBOCALL_DIR, label=1, limit=MAX_TRAIN_SAMPLES)
    n_robocall = len(X_rob)
    if n_robocall == 0:
        print("ERROR: No robocall audio files found.")
        return
    print(f"Loaded {n_robocall} robocall samples")

    # --- Negative class: normal speech ---
    normal_path  = Path(NORMAL_DIR)
    normal_path.mkdir(parents=True, exist_ok=True)
    normal_files = sorted(list(normal_path.glob("*.wav")) + list(normal_path.glob("*.mp3")))

    if normal_files:
        X_norm, y_norm = load_audio_folder(NORMAL_DIR, label=0, limit=n_robocall)
    else:
        print(f"No files in {NORMAL_DIR}/ — downloading from LibriSpeech...")
        X_norm, y_norm = load_normal_from_huggingface(n_robocall)

    if not X_norm:
        print("\nERROR: Could not load normal speech samples.")
        print(f"Add normal call recordings (.wav/.mp3) to {NORMAL_DIR}/ and try again.")
        return
    print(f"Loaded {len(X_norm)} normal speech samples")

    # --- Train ---
    X = np.array(X_rob + X_norm)
    y = np.array(y_rob + y_norm)

    print(f"\nTraining on {len(X)} samples  ({sum(y==1)} robocall  /  {sum(y==0)} normal)")

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    print("\nTest set results:")
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=["Normal", "Robocall"]))
    plot_confusion(y_test, y_pred)

    joblib.dump({"model": clf, "scaler": scaler}, MODEL_FILE)
    print(f"Model saved -> {MODEL_FILE}")


def predict(audio_path):
    if not Path(MODEL_FILE).exists():
        print(f"No trained model found. Run:  python ML.py train")
        return

    bundle  = joblib.load(MODEL_FILE)
    clf     = bundle["model"]
    scaler  = bundle["scaler"]

    y, sr    = load_audio(audio_path, sr=SAMPLE_RATE)
    features = scaler.transform([extract_features(y, sr)])
    prob     = clf.predict_proba(features)[0]

    is_robo = prob[1] >= 0.5
    label   = "ROBOCALL" if is_robo else "NORMAL CALL"

    print(f"\nFile     : {audio_path}")
    print(f"Result   : {label}")
    print(f"Robocall : {prob[1]:.1%}")
    print(f"Normal   : {prob[0]:.1%}")
    plot_fingerprint(y, sr, audio_path, label, prob[1])


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] == "train":
        train()
    elif sys.argv[1] == "test" and len(sys.argv) >= 3:
        target = Path(sys.argv[2])
        if target.is_dir():
            files = sorted([f for ext in ("*.wav", "*.mp3", "*.m4a") for f in target.glob(ext)])
            if not files:
                print(f"No audio files found in {target}")
            for f in files:
                predict(str(f))
        else:
            predict(sys.argv[2])
    else:
        print("Usage:")
        print("  python ML.py train")
        print("  python ML.py test path/to/audio.mp3")
        print("  python ML.py test path/to/folder/")
