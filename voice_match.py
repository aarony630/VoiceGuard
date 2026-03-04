import subprocess
import tempfile
import imageio_ffmpeg
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from resemblyzer import VoiceEncoder, preprocess_wav
from pathlib import Path
from scipy.spatial.distance import cosine

# --- Config ---
ENROLLMENT_DIR = "enrollment"   # subfolders = one person each (e.g. enrollment/Joyce/)
TEST_DIR       = "test"
THRESHOLD      = 0.82           # cosine similarity cutoff (tune 0.75-0.90)
MIN_MARGIN     = 0.08           # winner must beat 2nd place by at least this much


def plot_mfcc(ax, audio_path, title):
    wav_path, is_tmp = to_wav_if_needed(audio_path)
    y, sr = librosa.load(wav_path, sr=None)
    if is_tmp:
        Path(wav_path).unlink(missing_ok=True)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    librosa.display.specshow(mfccs, x_axis="time", ax=ax, sr=sr)
    ax.set_title(title, fontsize=9)
    ax.set_ylabel("Coeff")


def to_wav_if_needed(path):
    """Convert m4a to a temp wav file via ffmpeg; return path to use."""
    if Path(path).suffix.lower() != ".m4a":
        return path, False
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", path, "-ar", "16000", "-ac", "1", tmp.name],
        check=True, capture_output=True
    )
    return tmp.name, True


def get_embedding(encoder, path):
    wav_path, is_tmp = to_wav_if_needed(path)
    wav = preprocess_wav(wav_path)
    if is_tmp:
        Path(wav_path).unlink(missing_ok=True)
    return encoder.embed_utterance(wav)


def build_profiles(encoder, name_filter=None):
    """Build a voice profile per person from enrollment subfolders."""
    enrollment_root = Path(ENROLLMENT_DIR)
    person_dirs = sorted([d for d in enrollment_root.iterdir() if d.is_dir()])
    if name_filter:
        person_dirs = [d for d in person_dirs if d.name.lower() == name_filter.lower()]

    if not person_dirs:
        raise FileNotFoundError(
            f"No subfolders found in '{ENROLLMENT_DIR}/'. "
            f"Create one folder per person, e.g. enrollment/Joyce/"
        )

    profiles = {}
    for person_dir in person_dirs:
        files = sorted([f for ext in ("*.wav", "*.mp3") for f in person_dir.glob(ext)])
        if not files:
            print(f"  Warning: no audio files found in {person_dir}, skipping.")
            continue
        print(f"\n  {person_dir.name} ({len(files)} file(s)):")
        embeddings = []
        for f in files:
            emb = get_embedding(encoder, str(f))
            embeddings.append(emb)
            print(f"    {f.name}")
        profiles[person_dir.name] = {
            "profile": np.mean(embeddings, axis=0),
            "files": files,
        }

    return profiles


def main(name_filter=None):
    print("Loading voice encoder (first run downloads weights)...")
    encoder = VoiceEncoder()

    print("\nBuilding enrollment profiles...")
    profiles = build_profiles(encoder, name_filter)

    # --- Test sample ---
    test_files = sorted([
        f for ext in ("*.wav", "*.mp3", "*.m4a")
        for f in Path(TEST_DIR).glob(ext)
    ])
    if not test_files:
        raise FileNotFoundError(f"No audio file found in '{TEST_DIR}/'")
    TEST_FILE = str(test_files[0])

    print(f"\nTest sample: {TEST_FILE}")
    test_embedding = get_embedding(encoder, TEST_FILE)

    # --- Compare against each person ---
    print("\nResults:")
    results = {}
    for name, data in profiles.items():
        sim = 1 - cosine(data["profile"], test_embedding)
        results[name] = {"sim": sim, "matched": False}

    # Determine winner: must exceed threshold AND have a clear margin over 2nd place
    sorted_scores = sorted(results.items(), key=lambda x: x[1]["sim"], reverse=True)
    if len(sorted_scores) >= 2:
        top_name, top_data = sorted_scores[0]
        second_sim = sorted_scores[1][1]["sim"]
        margin = top_data["sim"] - second_sim
        if top_data["sim"] >= THRESHOLD and margin >= MIN_MARGIN:
            results[top_name]["matched"] = True
    elif sorted_scores and sorted_scores[0][1]["sim"] >= THRESHOLD:
        results[sorted_scores[0][0]]["matched"] = True

    for name, r in results.items():
        status = "MATCH" if r["matched"] else "NO MATCH"
        print(f"  {name:20s}  similarity={r['sim']:.4f}  [{status}]")

    # --- Plot ---
    n_people = len(profiles)
    fig, axes = plt.subplots(2, max(n_people, 2), figsize=(6 * max(n_people, 2), 7))
    fig.suptitle("Voice Match — Per-Person Profiles vs Test Sample", fontsize=13, fontweight="bold")

    # Top row: one enrollment sample per person
    for col, (name, data) in enumerate(profiles.items()):
        plot_mfcc(axes[0][col], str(data["files"][0]),
                  f"{name}  ·  {data['files'][0].name}")

    # Hide unused top-row axes
    for col in range(n_people, axes.shape[1]):
        axes[0][col].axis("off")

    # Bottom-left: test sample MFCC
    plot_mfcc(axes[1][0], TEST_FILE, f"Test  ·  {Path(TEST_FILE).name}")

    # Bottom-right: result panel
    ax_r = axes[1][1] if axes.shape[1] > 1 else axes[1][0]
    ax_r.axis("off")
    best_name = max(results, key=lambda n: results[n]["sim"])
    best = results[best_name]
    color = "#2ecc71" if best["matched"] else "#e74c3c"
    label = f"MATCH: {best_name}" if best["matched"] else "NO MATCH"
    ax_r.text(0.5, 0.62, label,
              transform=ax_r.transAxes, fontsize=18, fontweight="bold",
              color=color, ha="center", va="center")
    lines = [f"{n}: {r['sim']:.2%}" for n, r in results.items()]
    ax_r.text(0.5, 0.38, "\n".join(lines),
              transform=ax_r.transAxes, fontsize=11,
              color="black", ha="center", va="center")
    ax_r.text(0.5, 0.18, f"Threshold: {THRESHOLD:.2%}  |  Min margin: {MIN_MARGIN:.0%}",
              transform=ax_r.transAxes, fontsize=9,
              color="grey", ha="center", va="center")
    ax_r.set_title("Verification Result", fontsize=9)
    rect_color = "#d5f5e3" if best["matched"] else "#fadbd8"
    ax_r.add_patch(plt.Rectangle((0.05, 0.08), 0.9, 0.82,
                                  transform=ax_r.transAxes,
                                  color=rect_color, zorder=0))

    # Hide any extra bottom axes
    for col in range(2, axes.shape[1]):
        axes[1][col].axis("off")

    plt.tight_layout()
    out = "voice_fingerprint_result.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out}")
    plt.show()


if __name__ == "__main__":
    import sys
    name_filter = None
    if len(sys.argv) >= 2:
        name_filter = sys.argv[1]
        available = [d.name for d in Path(ENROLLMENT_DIR).iterdir() if d.is_dir()]
        if not any(d.lower() == name_filter.lower() for d in available):
            print(f"ERROR: No enrollment folder found for '{name_filter}'")
            print(f"Available: {available}")
            sys.exit(1)
        print(f"Comparing against: {name_filter}")
    main(name_filter)
