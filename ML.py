"""
Robocall Detector - ML.py (Leakage-Safe)

Setup:
  1. Clone the robocall dataset:
       git clone https://github.com/wspr-ncsu/robocall-audio-dataset data/robocalls
  2. Install dependencies:
       pip install scikit-learn joblib datasets librosa tqdm imageio-ffmpeg matplotlib soundfile
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
import hashlib
from collections import Counter
from pathlib import Path

import imageio_ffmpeg
import joblib
import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler

# --- Config ---
BASE_DIR          = Path(__file__).resolve().parent
ROBOCALL_DIR      = "data/robocalls"
NORMAL_DIR        = "data/normal_calls"
MODEL_FILE        = BASE_DIR / "robocall_detector.pkl"
SAMPLE_RATE       = 16000
N_MFCC            = 40
MAX_DURATION      = 30
MAX_TRAIN_SAMPLES = 1000
ROBOCALL_THRESHOLD = 0.60

# If True, split by group instead of random file-level split.
# This helps reduce leakage from near-duplicate files in the same folder/source.
USE_GROUP_SPLIT = False


def load_audio(audio_path, sr=SAMPLE_RATE):
    """Load audio. Route .m4a through ffmpeg in memory."""
    ext = Path(audio_path).suffix.lower()
    if ext == ".m4a":
        result = subprocess.run(
            [
                imageio_ffmpeg.get_ffmpeg_exe(),
                "-i", str(audio_path),
                "-f", "wav",
                "-ar", str(sr),
                "-ac", "1",
                "pipe:1"
            ],
            capture_output=True,
            check=True
        )
        return librosa.load(io.BytesIO(result.stdout), sr=sr, mono=True)
    return librosa.load(str(audio_path), sr=sr, mono=True)


def plot_fingerprint(y, sr, audio_path, label, prob_robo):
    """Save MFCC heatmap for a given audio clip."""
    max_samples = MAX_DURATION * sr
    if len(y) > max_samples:
        y = y[:max_samples]

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)

    fig, ax = plt.subplots(figsize=(12, 5))
    img = librosa.display.specshow(mfccs, x_axis="time", sr=sr, ax=ax, cmap="magma")
    fig.colorbar(img, ax=ax, label="MFCC Amplitude")
    ax.set_title(
        f"Audio Fingerprint — {Path(audio_path).name}\n{label}  |  Robocall: {prob_robo:.1%}",
        fontsize=13
    )
    ax.set_ylabel("MFCC Coefficient")
    ax.set_xlabel("Time (s)")

    out = Path(audio_path).stem + "_fingerprint.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Fingerprint saved -> {out}")


def plot_confusion(y_true, y_pred):
    """Save confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["Normal", "Robocall"],
        yticklabels=["Normal", "Robocall"],
        xlabel="Predicted",
        ylabel="Actual",
        title="Confusion Matrix"
    )
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
                fontsize=14
            )
    fig.tight_layout()
    fig.savefig(str(Path("confusion_matrix.png").resolve()), dpi=150)
    plt.close(fig)
    print("Confusion matrix saved -> confusion_matrix.png")


def extract_features(y, sr):
    """Extract MFCC + spectral + pitch features."""
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


def get_group_id(file_path, root_dir):
    """
    Build a coarse group id to reduce leakage.
    Idea: files from the same immediate parent folders are treated as related.

    Example group:
      robocalls/sourceA/campaign1/file.wav  -> "sourceA/campaign1"
      normal_calls/speaker3/file.wav        -> "speaker3"
    """
    file_path = Path(file_path)
    root_dir = Path(root_dir)

    try:
        rel = file_path.relative_to(root_dir)
    except ValueError:
        rel = file_path

    parts = rel.parts

    if len(parts) >= 3:
        return "/".join(parts[:2])
    if len(parts) >= 2:
        return parts[0]
    return file_path.stem


def file_sha1(file_path, block_size=65536):
    """Optional duplicate check by file bytes."""
    h = hashlib.sha1()
    with open(file_path, "rb") as f:
        while True:
            block = f.read(block_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def load_audio_folder(folder, label, limit=None):
    """
    Load audio files from a folder and extract features.
    Returns:
      X_list, y_list, file_paths, groups
    """
    folder = Path(folder)
    files = sorted(
        list(folder.rglob("*.wav")) +
        list(folder.rglob("*.mp3")) +
        list(folder.rglob("*.m4a"))
    )
    if limit:
        files = files[:limit]

    X, y, file_paths, groups = [], [], [], []
    tag = "Robocall" if label == 1 else "Normal"

    for f in tqdm(files, desc=f"{tag} features"):
        try:
            audio, sr = load_audio(str(f), sr=SAMPLE_RATE)
            if len(audio) < sr:   # skip clips under 1 second
                continue

            X.append(extract_features(audio, sr))
            y.append(label)
            file_paths.append(str(f))
            groups.append(get_group_id(f, folder))
        except Exception as e:
            print(f"  Skipped {f.name}: {e}")

    return X, y, file_paths, groups


def load_normal_from_huggingface(n_samples):
    """
    Download normal speech from LibriSpeech via HuggingFace.
    Returns:
      X_list, y_list, file_paths, groups
    """
    print("Downloading LibriSpeech test-clean from HuggingFace...")
    try:
        import soundfile as sf
        from datasets import load_dataset, Audio

        ds = load_dataset(
            "librispeech_asr",
            "clean",
            split="test",
            trust_remote_code=True,
            streaming=True
        )
        ds = ds.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE, decode=False))

        X, y, file_paths, groups = [], [], [], []

        for i, sample in enumerate(tqdm(ds, desc="Normal features", total=n_samples)):
            if i >= n_samples:
                break
            try:
                audio_entry = sample["audio"]

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

                # synthetic path/group to support group splitting
                speaker_id = str(sample.get("speaker_id", f"speaker_{i}"))
                chapter_id = str(sample.get("chapter_id", "chapter"))
                file_paths.append(f"hf://librispeech/{speaker_id}/{chapter_id}/{i}")
                groups.append(f"{speaker_id}/{chapter_id}")

            except Exception as e:
                print(f"  Skipped sample {i}: {e}")

        return X, y, file_paths, groups

    except Exception as e:
        print(f"HuggingFace download failed: {e}")
        return [], [], [], []


def print_split_stats(name, y_split, groups_split):
    print(f"{name}: n={len(y_split)} | class_counts={dict(Counter(y_split))}")
    print(f"{name}: unique_groups={len(set(groups_split))}")


def group_train_val_test_split(X, y, groups, train_size=0.70, val_size=0.15, test_size=0.15, random_state=42):
    """
    Group-based split:
      1) train vs temp
      2) val vs test
    """
    assert abs(train_size + val_size + test_size - 1.0) < 1e-8

    X = np.array(X)
    y = np.array(y)
    groups = np.array(groups)

    gss1 = GroupShuffleSplit(n_splits=1, train_size=train_size, random_state=random_state)
    train_idx, temp_idx = next(gss1.split(X, y, groups=groups))

    X_train, y_train, g_train = X[train_idx], y[train_idx], groups[train_idx]
    X_temp, y_temp, g_temp    = X[temp_idx], y[temp_idx], groups[temp_idx]

    # temp should be split into val/test equally
    relative_val_size = val_size / (val_size + test_size)

    gss2 = GroupShuffleSplit(n_splits=1, train_size=relative_val_size, random_state=random_state)
    val_rel_idx, test_rel_idx = next(gss2.split(X_temp, y_temp, groups=g_temp))

    X_val, y_val, g_val   = X_temp[val_rel_idx], y_temp[val_rel_idx], g_temp[val_rel_idx]
    X_test, y_test, g_test = X_temp[test_rel_idx], y_temp[test_rel_idx], g_temp[test_rel_idx]

    return X_train, X_val, X_test, y_train, y_val, y_test, g_train, g_val, g_test


def random_train_val_test_split(X, y, groups, random_state=42):
    """
    Fallback random split.
    Still leakage-safe with respect to scaling,
    but weaker than group split for audio datasets.
    """
    X = np.array(X)
    y = np.array(y)
    groups = np.array(groups)

    X_train, X_temp, y_train, y_temp, g_train, g_temp = train_test_split(
        X, y, groups, test_size=0.30, random_state=random_state, stratify=y
    )
    X_val, X_test, y_val, y_test, g_val, g_test = train_test_split(
        X_temp, y_temp, g_temp, test_size=0.50, random_state=random_state, stratify=y_temp
    )

    return X_train, X_val, X_test, y_train, y_val, y_test, g_train, g_val, g_test


def train():
    # --- Positive class: robocalls ---
    if not Path(ROBOCALL_DIR).exists():
        print(f"\nERROR: Robocall audio folder not found: {ROBOCALL_DIR}")
        print("Clone the dataset first:")
        print("  git clone https://github.com/wspr-ncsu/robocall-audio-dataset data/robocalls")
        return

    X_rob, y_rob, paths_rob, groups_rob = load_audio_folder(
        ROBOCALL_DIR, label=1, limit=MAX_TRAIN_SAMPLES
    )
    n_robocall = len(X_rob)
    if n_robocall == 0:
        print("ERROR: No robocall audio files found.")
        return
    print(f"Loaded {n_robocall} robocall samples")

    # --- Negative class: normal speech ---
    normal_path = Path(NORMAL_DIR)
    normal_path.mkdir(parents=True, exist_ok=True)
    normal_files = sorted(
        list(normal_path.rglob("*.wav")) +
        list(normal_path.rglob("*.mp3")) +
        list(normal_path.rglob("*.m4a"))
    )

    if normal_files:
        X_norm, y_norm, paths_norm, groups_norm = load_audio_folder(
            NORMAL_DIR, label=0, limit=MAX_TRAIN_SAMPLES
        )
    else:
        print(f"No files in {NORMAL_DIR}/ — downloading from LibriSpeech...")
        X_norm, y_norm, paths_norm, groups_norm = load_normal_from_huggingface(n_robocall)

    if not X_norm:
        print("\nERROR: Could not load normal speech samples.")
        print(f"Add normal call recordings (.wav/.mp3/.m4a) to {NORMAL_DIR}/ and try again.")
        return
    print(f"Loaded {len(X_norm)} normal speech samples")

    # --- Cap both classes at the smaller count ---
    cap = min(len(X_rob), len(X_norm))
    X_rob, y_rob, paths_rob, groups_rob = X_rob[:cap], y_rob[:cap], paths_rob[:cap], groups_rob[:cap]
    X_norm, y_norm, paths_norm, groups_norm = X_norm[:cap], y_norm[:cap], paths_norm[:cap], groups_norm[:cap]
    print(f"Capped at {cap} per class")

    # --- Merge dataset ---
    X = X_rob + X_norm
    y = y_rob + y_norm
    file_paths = paths_rob + paths_norm
    groups = groups_rob + groups_norm

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    file_paths = np.array(file_paths)
    groups = np.array(groups)

    print(f"\nTraining on {len(X)} samples ({sum(y==1)} robocall / {sum(y==0)} normal)")

    # --- Split BEFORE scaling ---
    if USE_GROUP_SPLIT:
        print("\nUsing GROUP-BASED split")
        X_train, X_val, X_test, y_train, y_val, y_test, g_train, g_val, g_test = group_train_val_test_split(
            X, y, groups, train_size=0.70, val_size=0.15, test_size=0.15, random_state=42
        )
    else:
        print("\nUsing RANDOM stratified split")
        X_train, X_val, X_test, y_train, y_val, y_test, g_train, g_val, g_test = random_train_val_test_split(
            X, y, groups, random_state=42
        )

    print_split_stats("Train", y_train, g_train)
    print_split_stats("Val", y_val, g_val)
    print_split_stats("Test", y_test, g_test)

    # overlap check
    overlap_train_val = set(g_train).intersection(set(g_val))
    overlap_train_test = set(g_train).intersection(set(g_test))
    overlap_val_test = set(g_val).intersection(set(g_test))
    print(f"Group overlap train/val  : {len(overlap_train_val)}")
    print(f"Group overlap train/test : {len(overlap_train_test)}")
    print(f"Group overlap val/test   : {len(overlap_val_test)}")

    # --- Scale using TRAIN ONLY ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled   = scaler.transform(X_val)
    X_test_scaled  = scaler.transform(X_test)

    # --- Model ---
    clf = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train_scaled, y_train)

    # --- Validation ---
    print("\nValidation set results:")
    y_val_pred = clf.predict(X_val_scaled)
    print(classification_report(y_val, y_val_pred, target_names=["Normal", "Robocall"], digits=4))
    print("Validation accuracy:", f"{accuracy_score(y_val, y_val_pred):.4f}")

    # --- Test ---
    print("\nTest set results:")
    y_test_pred = clf.predict(X_test_scaled)
    print(classification_report(y_test, y_test_pred, target_names=["Normal", "Robocall"], digits=4))
    print("Test accuracy:", f"{accuracy_score(y_test, y_test_pred):.4f}")
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_test_pred))
    plot_confusion(y_test, y_test_pred)

    # --- Save ---
    joblib.dump(
        {
            "model": clf,
            "scaler": scaler,
            "config": {
                "sample_rate": SAMPLE_RATE,
                "n_mfcc": N_MFCC,
                "max_duration": MAX_DURATION,
                "use_group_split": USE_GROUP_SPLIT,
                "robocall_threshold": ROBOCALL_THRESHOLD
            }
        },
        MODEL_FILE
    )
    print(f"Model saved -> {MODEL_FILE}")


def predict(audio_path, save_fingerprint=True):
    if not Path(MODEL_FILE).exists():
        print("No trained model found. Run: python ML.py train")
        return

    bundle = joblib.load(MODEL_FILE)
    clf = bundle["model"]
    scaler = bundle["scaler"]

    y, sr = load_audio(audio_path, sr=SAMPLE_RATE)
    features = extract_features(y, sr).reshape(1, -1)
    features_scaled = scaler.transform(features)

    prob = clf.predict_proba(features_scaled)[0]
    is_robo = prob[1] >= ROBOCALL_THRESHOLD
    label = "ROBOCALL" if is_robo else "NORMAL CALL"

    print(f"\nFile     : {audio_path}")
    print(f"Result   : {label}")
    print(f"Robocall : {prob[1]:.1%}")
    print(f"Normal   : {prob[0]:.1%}")

    if save_fingerprint:
        import random
        plot_fingerprint(y, sr, audio_path, label, prob[1])

        match_dir = Path(ROBOCALL_DIR) if is_robo else Path(NORMAL_DIR)
        match_files = (
            list(match_dir.rglob("*.wav")) +
            list(match_dir.rglob("*.mp3")) +
            list(match_dir.rglob("*.m4a"))
        )

        if match_files:
            ref_path = random.choice(match_files)
            ref_y, ref_sr = load_audio(str(ref_path), sr=SAMPLE_RATE)
            ref_label = "ROBOCALL (reference)" if is_robo else "NORMAL CALL (reference)"
            plot_fingerprint(ref_y, ref_sr, str(ref_path), ref_label, prob[1])
            print(f"Reference: {ref_path.name}")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] == "train":
        train()
    elif sys.argv[1] == "test" and len(sys.argv) >= 3:
        target = Path(sys.argv[2])
        if target.is_dir():
            files = sorted([
                f for ext in ("*.wav", "*.mp3", "*.m4a")
                for f in target.glob(ext)
            ])
            if not files:
                print(f"No audio files found in {target}")
            for f in files:
                predict(str(f), save_fingerprint=False)
        else:
            predict(sys.argv[2])
    else:
        print("Usage:")
        print("  python ML.py train")
        print("  python ML.py test path/to/audio.mp3")
        print("  python ML.py test path/to/folder/")
