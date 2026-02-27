#!/usr/bin/env python3
"""
Scenario 1: CLI Audio Transcription with Speaker Diarization
Uses NVIDIA Parakeet for ASR and NeMo ClusteringDiarizer (TitaNet + VAD)
to identify who spoke when.

Usage:
    python scenario1/transcribe-diarize.py
    python scenario1/transcribe-diarize.py audio.mp3
    python scenario1/transcribe-diarize.py audio.mp3 --speakers 2
    python scenario1/transcribe-diarize.py audio.mp3 --voiceprints voiceprints/
"""

import json
import os
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf

# Fix Windows temp directory issue with NeMo/lhotse (avoid paths with spaces)
# Use a short, space-free temp path to prevent WinError 267 in NeMo's manifest handling
_local_temp = Path("C:/nemo_temp")
_local_temp.mkdir(exist_ok=True)
os.environ["TEMP"] = str(_local_temp)
os.environ["TMP"] = str(_local_temp)
tempfile.tempdir = str(_local_temp)

# Constants
AUDIO_EXTENSIONS = {'.wav', '.flac', '.mp3'}
ASR_MODEL_NAME = "nvidia/parakeet-tdt-0.6b-v2"
SPEAKER_MODEL = "titanet_large"          # Speaker embedding model
VAD_MODEL = "vad_marblenet"              # Voice activity detection model (English)
TARGET_SAMPLE_RATE = 16000
DEFAULT_INPUT_DIR = "input"              # Default directory to scan for audio files


# ─── Audio conversion ────────────────────────────────────────────────

def convert_to_wav(audio_path: Path, output_path: Path | None = None) -> Path:
    """Convert audio file to 16kHz mono WAV for model compatibility."""
    print(f"Converting {audio_path.name} to 16kHz WAV...")
    audio, _ = librosa.load(str(audio_path), sr=TARGET_SAMPLE_RATE, mono=True)
    if output_path is None:
        temp_dir = Path(tempfile.gettempdir())
        # Sanitize filename: replace spaces and special chars to avoid
        # WinError 267 in NeMo/lhotse path handling
        safe_stem = audio_path.stem.replace(' ', '_')
        safe_stem = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in safe_stem)
        output_path = temp_dir / f"diar_temp_{safe_stem}.wav"
    sf.write(str(output_path), audio, TARGET_SAMPLE_RATE)
    return output_path


# ─── Voiceprint support ───────────────────────────────────────────────

def load_voiceprints(voiceprint_dir: Path) -> dict[str, np.ndarray]:
    """
    Load reference audio files from a directory and extract speaker embeddings.
    File names (without extension) become speaker names.
    E.g., voiceprints/Mike_Collins.wav → "Mike_Collins"

    Supports .wav, .flac, .mp3 files. Non-WAV files are converted first.
    """
    from nemo.collections.asr.models import EncDecSpeakerLabelModel

    if not voiceprint_dir.exists():
        print(f"Warning: Voiceprint directory not found: {voiceprint_dir}")
        return {}

    audio_files = [
        f for f in voiceprint_dir.iterdir()
        if f.suffix.lower() in AUDIO_EXTENSIONS
    ]
    if not audio_files:
        print(f"Warning: No audio files found in {voiceprint_dir}")
        return {}

    print(f"\nLoading voiceprints from {voiceprint_dir}...")
    print(f"  Found {len(audio_files)} reference file(s)")

    # Load speaker model for embedding extraction
    speaker_model = EncDecSpeakerLabelModel.from_pretrained(model_name=SPEAKER_MODEL)

    voiceprints = {}
    for audio_file in sorted(audio_files):
        speaker_name = audio_file.stem.replace('_', ' ')
        # Always normalize to 16kHz mono WAV via librosa (even .wav files)
        # to avoid shape mismatches like (batch, time, 1) vs expected (batch, time)
        wav_file = Path(tempfile.gettempdir()) / f"vp_{audio_file.stem}.wav"
        audio, _ = librosa.load(str(audio_file), sr=TARGET_SAMPLE_RATE, mono=True)
        sf.write(str(wav_file), audio, TARGET_SAMPLE_RATE)

        try:
            emb = speaker_model.get_embedding(str(wav_file))
            # Flatten to 1D numpy array
            if hasattr(emb, 'cpu'):
                emb = emb.cpu().numpy().flatten()
            voiceprints[speaker_name] = emb
            print(f"  ✓ {speaker_name} ({audio_file.name})")
        except Exception as e:
            print(f"  ✗ {speaker_name}: {e}")
        finally:
            # Clean up temp WAV
            if wav_file.exists():
                wav_file.unlink()

    return voiceprints


def match_speakers_to_voiceprints(
    diar_segments: list[dict],
    wav_path: Path,
    voiceprints: dict[str, np.ndarray],
    threshold: float = 0.5,
) -> dict[str, str]:
    """
    Match diarization cluster IDs to known voiceprints.

    For each unique speaker cluster, extract an embedding from a representative
    audio segment, then compare via cosine similarity to all voiceprints.

    Returns a mapping: {cluster_id: speaker_name}.
    Unmatched clusters keep their original ID.
    """
    from nemo.collections.asr.models import EncDecSpeakerLabelModel
    import torch

    if not voiceprints:
        return {}

    unique_speakers = sorted(set(s['speaker'] for s in diar_segments))
    if not unique_speakers:
        return {}

    # Load speaker model
    speaker_model = EncDecSpeakerLabelModel.from_pretrained(model_name=SPEAKER_MODEL)

    # Load the full audio
    audio, _ = librosa.load(str(wav_path), sr=TARGET_SAMPLE_RATE, mono=True)

    speaker_map = {}
    used_names = set()

    print(f"\nMatching clusters to voiceprints (threshold={threshold})...")

    for cluster_id in unique_speakers:
        # Gather segments for this cluster and pick longest ones for a robust embedding
        cluster_segs = [s for s in diar_segments if s['speaker'] == cluster_id]
        cluster_segs.sort(key=lambda s: s['end'] - s['start'], reverse=True)

        # Use up to 30 seconds of the longest segments
        combined_audio = []
        total_dur = 0.0
        for seg in cluster_segs:
            start_sample = int(seg['start'] * TARGET_SAMPLE_RATE)
            end_sample = int(seg['end'] * TARGET_SAMPLE_RATE)
            chunk = audio[start_sample:end_sample]
            combined_audio.append(chunk)
            total_dur += seg['end'] - seg['start']
            if total_dur >= 30.0:
                break

        # Write combined audio to temp file for embedding extraction
        combined = np.concatenate(combined_audio)
        # Sanitize cluster_id for safe filenames (e.g. '<NA>' has illegal chars on Windows)
        safe_id = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in str(cluster_id))
        tmp_path = Path(tempfile.gettempdir()) / f"cluster_{safe_id}.wav"
        sf.write(str(tmp_path), combined, TARGET_SAMPLE_RATE)

        try:
            cluster_emb = speaker_model.get_embedding(str(tmp_path))
            if hasattr(cluster_emb, 'cpu'):
                cluster_emb = cluster_emb.cpu().numpy().flatten()

            # Compare to all voiceprints via cosine similarity
            best_name = None
            best_score = -1.0
            for name, ref_emb in voiceprints.items():
                if name in used_names:
                    continue
                cos_sim = np.dot(cluster_emb, ref_emb) / (
                    np.linalg.norm(cluster_emb) * np.linalg.norm(ref_emb) + 1e-8
                )
                if cos_sim > best_score:
                    best_score = cos_sim
                    best_name = name

            if best_name and best_score >= threshold:
                speaker_map[cluster_id] = best_name
                used_names.add(best_name)
                print(f"  {cluster_id} → {best_name} (similarity: {best_score:.3f})")
            else:
                speaker_map[cluster_id] = cluster_id
                print(f"  {cluster_id} → {cluster_id} (no match, best: {best_score:.3f})")
        except Exception as e:
            print(f"  {cluster_id} → {cluster_id} (error: {e})")
            speaker_map[cluster_id] = cluster_id
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    return speaker_map


def apply_speaker_names(
    segments: list[dict],
    speaker_map: dict[str, str],
) -> list[dict]:
    """Replace cluster IDs with speaker names in segment list."""
    return [
        {**seg, 'speaker': speaker_map.get(seg['speaker'], seg['speaker'])}
        for seg in segments
    ]


# ─── RTTM parsing ────────────────────────────────────────────────────

def parse_rttm(rttm_path: str) -> list[dict]:
    """
    Parse an RTTM file into a list of speaker segments.
    RTTM format: SPEAKER <file> 1 <start> <dur> <NA> <NA> <speaker> <NA> <NA>
    """
    segments = []
    with open(rttm_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 8 and parts[0] == "SPEAKER":
                start = float(parts[3])
                duration = float(parts[4])
                speaker = parts[7]
                # Skip invalid/placeholder entries
                if speaker in ('<NA>', 'NA', ''):
                    continue
                segments.append({
                    'start': start,
                    'end': start + duration,
                    'speaker': speaker,
                })
    # Sort by start time
    segments.sort(key=lambda s: s['start'])
    return segments


# ─── Speaker diarization ─────────────────────────────────────────────

def run_diarization(
    wav_path: Path,
    out_dir: Path,
    num_speakers: int | None = None,
    max_speakers: int = 8,
) -> list[dict]:
    """
    Run NeMo ClusteringDiarizer on a WAV file.
    Returns a list of dicts: [{start, end, speaker}, ...]
    """
    from omegaconf import OmegaConf
    from nemo.collections.asr.models import ClusteringDiarizer

    diar_out = out_dir / "diar_work"
    diar_out.mkdir(parents=True, exist_ok=True)

    # Create NeMo manifest (JSON-lines) pointing to the audio file
    manifest_path = diar_out / "input_manifest.json"
    meta = {
        "audio_filepath": str(wav_path),
        "offset": 0,
        "duration": None,
        "label": "infer",
        "text": "-",
        "num_speakers": num_speakers,
        "rttm_filepath": None,
        "uem_filepath": None,
    }
    manifest_path.write_text(json.dumps(meta) + "\n", encoding="utf-8")

    # Auto-detect device
    import torch
    _device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build OmegaConf config for ClusteringDiarizer
    cfg = OmegaConf.create({
        "device": _device,
        "verbose": True,
        "sample_rate": TARGET_SAMPLE_RATE,
        "batch_size": 64,
        "num_workers": 0,  # Must be 0 on Windows to avoid multiprocessing pickle errors
        "diarizer": {
            "manifest_filepath": str(manifest_path),
            "out_dir": str(diar_out),
            "oracle_vad": False,
            "collar": 0.25,
            "ignore_overlap": True,
            "vad": {
                "model_path": VAD_MODEL,
                "external_vad_manifest": None,
                "parameters": {
                    "window_length_in_sec": 0.15,
                    "shift_length_in_sec": 0.01,
                    "smoothing": "median",
                    "overlap": 0.5,
                    "onset": 0.1,
                    "offset": 0.1,
                    "pad_onset": 0.1,
                    "pad_offset": 0,
                    "min_duration_on": 0,
                    "min_duration_off": 0.2,
                    "filter_speech_first": True,
                },
            },
            "speaker_embeddings": {
                "model_path": SPEAKER_MODEL,
                "parameters": {
                    "window_length_in_sec": [1.5, 0.75],
                    "shift_length_in_sec": [0.75, 0.375],
                    "multiscale_weights": [1, 1],
                    "save_embeddings": False,
                },
            },
            "clustering": {
                "parameters": {
                    "oracle_num_speakers": num_speakers is not None,
                    "max_num_speakers": max_speakers,
                    "enhanced_count_thres": 80,
                    "max_rp_threshold": 0.25,
                    "sparse_search_volume": 30,
                    "maj_vote_spk_count": False,
                },
            },
        },
    })

    # If user specified num_speakers, set oracle
    if num_speakers is not None:
        cfg.diarizer.clustering.parameters.oracle_num_speakers = True

    print("\nRunning speaker diarization...")
    print(f"  Speaker model : {SPEAKER_MODEL}")
    print(f"  VAD model     : {VAD_MODEL}")
    if num_speakers:
        print(f"  Speakers      : {num_speakers} (user-specified)")
    else:
        print(f"  Speakers      : auto-detect (max {max_speakers})")

    diarizer = ClusteringDiarizer(cfg=cfg)
    diarizer.diarize()

    # Find the generated RTTM file
    rttm_dir = diar_out / "pred_rttms"
    rttm_files = list(rttm_dir.glob("*.rttm"))
    if not rttm_files:
        print("Warning: No RTTM output produced by diarizer.")
        return []

    segments = parse_rttm(str(rttm_files[0]))
    print(f"  Found {len(set(s['speaker'] for s in segments))} speaker(s), "
          f"{len(segments)} segment(s)")

    # Clean up working directory
    shutil.rmtree(diar_out, ignore_errors=True)

    return segments


# ─── ASR transcription ───────────────────────────────────────────────

def run_transcription(wav_path: Path) -> tuple[str, list[dict]]:
    """
    Transcribe audio with Parakeet and return (full_text, word_timestamps).
    word_timestamps: [{word, start, end}, ...]

    For long audio files, splits into chunks to avoid CUDA OOM on GPUs
    with limited VRAM (e.g. 10GB RTX 3080).
    """
    import nemo.collections.asr as nemo_asr
    import torch
    import gc

    # Free GPU memory from diarization models before loading ASR
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()

    print("\nLoading Parakeet ASR model...")
    print("(First run will download ~1.2GB model)")
    asr_model = nemo_asr.models.ASRModel.from_pretrained(ASR_MODEL_NAME)
    # Ensure model is on the correct device
    if not torch.cuda.is_available():
        asr_model = asr_model.cpu()
    print("Model loaded successfully!")

    print(f"\nTranscribing audio...")

    # Load audio as numpy array to bypass NeMo/lhotse manifest temp directory
    # issues on Windows (WinError 267)
    audio_np, _ = librosa.load(str(wav_path), sr=TARGET_SAMPLE_RATE, mono=True)
    duration_sec = len(audio_np) / TARGET_SAMPLE_RATE
    print(f"  Duration: {duration_sec:.1f}s ({duration_sec/60:.1f} min)")

    # Split long audio into chunks to avoid CUDA OOM
    # Parakeet can handle ~5 min comfortably on 10GB VRAM
    MAX_CHUNK_SEC = 300  # 5 minutes per chunk
    all_words = []
    all_segments = []
    all_text_parts = []

    if duration_sec <= MAX_CHUNK_SEC:
        chunks = [(audio_np, 0.0)]
    else:
        chunk_samples = MAX_CHUNK_SEC * TARGET_SAMPLE_RATE
        chunks = []
        for start_sample in range(0, len(audio_np), chunk_samples):
            chunk = audio_np[start_sample:start_sample + chunk_samples]
            offset_sec = start_sample / TARGET_SAMPLE_RATE
            chunks.append((chunk, offset_sec))
        print(f"  Splitting into {len(chunks)} chunks ({MAX_CHUNK_SEC}s each) to fit GPU memory")

    for chunk_idx, (chunk_audio, offset_sec) in enumerate(chunks):
        if len(chunks) > 1:
            chunk_dur = len(chunk_audio) / TARGET_SAMPLE_RATE
            print(f"\n  Chunk {chunk_idx + 1}/{len(chunks)} "
                  f"({offset_sec:.0f}s – {offset_sec + chunk_dur:.0f}s)")

        try:
            output = asr_model.transcribe(chunk_audio, timestamps=True)
            all_text_parts.append(output[0].text)
            # Offset timestamps to account for chunk position
            for w in output[0].timestamp.get('word', []):
                w['start'] += offset_sec
                w['end'] += offset_sec
                all_words.append(w)
            for s in output[0].timestamp.get('segment', []):
                s['start'] += offset_sec
                s['end'] += offset_sec
                all_segments.append(s)
        except Exception as e:
            print(f"  Timestamp extraction failed: {e}")
            print("  Retrying without timestamps...")
            if torch.cuda.is_available():
                gc.collect()
                torch.cuda.empty_cache()
            try:
                output = asr_model.transcribe(chunk_audio)
                t = output[0] if isinstance(output[0], str) else output[0].text
                all_text_parts.append(t)
            except Exception as e2:
                print(f"  Transcription error: {e2}")
                sys.exit(1)

    text = " ".join(all_text_parts)
    return text, all_words, all_segments


# ─── Alignment: map words → speakers ─────────────────────────────────

def assign_speakers_to_words(
    words: list[dict],
    diar_segments: list[dict],
) -> list[dict]:
    """
    For each ASR word, find which diarization speaker segment it falls into.
    Uses the word midpoint to decide assignment. If no segment contains the
    midpoint, assigns to the nearest segment's speaker.
    Returns words with added 'speaker' key.
    """
    labeled = []
    for w in words:
        mid = (w['start'] + w['end']) / 2
        speaker = None
        # First try: exact containment
        for seg in diar_segments:
            if seg['start'] <= mid <= seg['end']:
                speaker = seg['speaker']
                break
        # Fallback: nearest segment by distance to midpoint
        if speaker is None and diar_segments:
            best_dist = float('inf')
            for seg in diar_segments:
                # Distance from midpoint to nearest edge of segment
                dist = min(abs(mid - seg['start']), abs(mid - seg['end']))
                if dist < best_dist:
                    best_dist = dist
                    speaker = seg['speaker']
        labeled.append({**w, 'speaker': speaker or 'unknown'})
    return labeled


def assign_speakers_to_segments(
    asr_segments: list[dict],
    diar_segments: list[dict],
) -> list[dict]:
    """
    For each ASR segment, determine the dominant speaker based on overlap.
    Returns ASR segments with added 'speaker' key.
    """
    labeled = []
    for seg in asr_segments:
        seg_start = seg['start']
        seg_end = seg['end']
        # Calculate overlap with each diarization speaker segment
        speaker_overlap: dict[str, float] = {}
        for d in diar_segments:
            overlap_start = max(seg_start, d['start'])
            overlap_end = min(seg_end, d['end'])
            overlap = max(0, overlap_end - overlap_start)
            if overlap > 0:
                speaker_overlap[d['speaker']] = (
                    speaker_overlap.get(d['speaker'], 0) + overlap
                )
        # Pick speaker with most overlap
        if speaker_overlap:
            speaker = max(speaker_overlap, key=speaker_overlap.get)
        else:
            speaker = "unknown"
        labeled.append({**seg, 'speaker': speaker})
    return labeled


# ─── Build speaker-labeled transcript ─────────────────────────────────

def build_speaker_transcript(labeled_words: list[dict]) -> str:
    """Build a readable transcript grouped by speaker turns."""
    if not labeled_words:
        return ""

    lines = []
    current_speaker = None
    current_words = []
    turn_start = 0.0

    for w in labeled_words:
        if w['speaker'] != current_speaker:
            # Flush previous turn
            if current_words:
                text = " ".join(current_words)
                ts = seconds_to_srt_time(turn_start).replace(',', '.')
                lines.append(f"[{ts}] {current_speaker}: {text}")
            current_speaker = w['speaker']
            current_words = [w['word']]
            turn_start = w['start']
        else:
            current_words.append(w['word'])

    # Flush last turn
    if current_words:
        text = " ".join(current_words)
        ts = seconds_to_srt_time(turn_start).replace(',', '.')
        lines.append(f"[{ts}] {current_speaker}: {text}")

    return "\n".join(lines)


# ─── Output formatting ───────────────────────────────────────────────

def seconds_to_srt_time(seconds: float) -> str:
    """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def generate_diarized_srt(labeled_segments: list[dict]) -> str:
    """Generate SRT with speaker labels on each subtitle."""
    srt_lines = []
    for i, seg in enumerate(labeled_segments, 1):
        start = seconds_to_srt_time(seg['start'])
        end = seconds_to_srt_time(seg['end'])
        speaker = seg.get('speaker', 'unknown')
        text = seg['segment'].strip()
        srt_lines.append(f"{i}\n{start} --> {end}\n[{speaker}] {text}\n")
    return "\n".join(srt_lines)


def generate_diarized_txt(
    full_text: str,
    speaker_transcript: str,
    labeled_segments: list[dict],
    num_speakers: int,
) -> str:
    """Generate TXT with speaker-labeled transcript."""
    lines = [
        "TRANSCRIPTION (with speaker diarization)",
        "=" * 50,
        f"Speakers detected: {num_speakers}",
        "",
        "FULL TEXT",
        "-" * 50,
        full_text,
        "",
        "SPEAKER-LABELED TRANSCRIPT",
        "-" * 50,
        speaker_transcript,
        "",
        "TIMESTAMPED SEGMENTS",
        "-" * 50,
    ]
    for seg in labeled_segments:
        start = seconds_to_srt_time(seg['start']).replace(',', '.')
        end = seconds_to_srt_time(seg['end']).replace(',', '.')
        speaker = seg.get('speaker', 'unknown')
        lines.append(f"[{start} - {end}] [{speaker}] {seg['segment'].strip()}")
    return "\n".join(lines)


def save_outputs(
    full_text: str,
    speaker_transcript: str,
    labeled_segments: list[dict],
    num_speakers: int,
    audio_file: Path,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Save diarized transcription to .txt and .srt files."""
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = audio_file.stem

    txt_path = output_dir / f"{timestamp}_{base_name}_diarized.txt"
    srt_path = output_dir / f"{timestamp}_{base_name}_diarized.srt"

    txt_content = generate_diarized_txt(
        full_text, speaker_transcript, labeled_segments, num_speakers
    )
    srt_content = generate_diarized_srt(labeled_segments)

    txt_path.write_text(txt_content, encoding="utf-8")
    srt_path.write_text(srt_content, encoding="utf-8")

    return txt_path, srt_path


# ─── CLI ──────────────────────────────────────────────────────────────

def print_help():
    """Print usage instructions."""
    print(f"""
Usage: python scenario1/transcribe-diarize.py [audio_file ...] [options]

Transcription with speaker diarization using NVIDIA NeMo models.

Arguments:
  audio_file          Path to audio file(s) (.wav, .flac, or .mp3)
                      If omitted, processes all audio files in {DEFAULT_INPUT_DIR}/

Options:
  --input-dir DIR     Directory to scan for audio files (default: {DEFAULT_INPUT_DIR}/)
  --speakers N        Number of speakers (default: auto-detect)
  --max-speakers N    Maximum speakers for auto-detect (default: 8)
  --voiceprints DIR   Directory with reference audio files for known speakers.
                      File names become speaker labels (e.g., John_Smith.wav).
                      Underscores in names are replaced with spaces.
  --threshold N       Voiceprint match threshold 0.0-1.0 (default: 0.5)

Examples:
  python scenario1/transcribe-diarize.py                              # Process all files in {DEFAULT_INPUT_DIR}/
  python scenario1/transcribe-diarize.py meeting.mp3                  # Process a specific file
  python scenario1/transcribe-diarize.py --input-dir recordings/      # Scan a custom directory
  python scenario1/transcribe-diarize.py meeting.mp3 --speakers 2
  python scenario1/transcribe-diarize.py --voiceprints voiceprints/    # With speaker recognition

Voiceprint Setup:
  1. Create a directory (e.g., voiceprints/)
  2. Add short audio clips (5-30s) of each known speaker
  3. Name files after speakers: Mike_Collins.wav, Jane_Doe.mp3
  4. Run with --voiceprints voiceprints/

Output:
  Generates in the 'output/' directory:
  - {{timestamp}}_{{filename}}_diarized.txt - Transcript with speaker labels
  - {{timestamp}}_{{filename}}_diarized.srt - Subtitles with speaker labels

Models:
  ASR     : nvidia/parakeet-tdt-0.6b-v2 (English transcription)
  Speaker : titanet_large (speaker embeddings)
  VAD     : vad_marblenet (voice activity detection, English)
""")


def parse_args(argv: list[str]) -> tuple[list[Path], int | None, int, Path | None, float, Path | None]:
    """Parse CLI arguments. Returns (audio_paths, num_speakers, max_speakers, voiceprints_dir, threshold, input_dir)."""
    if argv and argv[0] in ['-h', '--help', 'help']:
        print_help()
        sys.exit(0)

    audio_paths = []
    num_speakers = None
    max_speakers = 8
    voiceprints_dir = None
    threshold = 0.5
    input_dir = None

    i = 0
    while i < len(argv):
        if argv[i] == '--speakers' and i + 1 < len(argv):
            num_speakers = int(argv[i + 1])
            i += 2
        elif argv[i] == '--max-speakers' and i + 1 < len(argv):
            max_speakers = int(argv[i + 1])
            i += 2
        elif argv[i] == '--voiceprints' and i + 1 < len(argv):
            voiceprints_dir = Path(argv[i + 1])
            i += 2
        elif argv[i] == '--threshold' and i + 1 < len(argv):
            threshold = float(argv[i + 1])
            i += 2
        elif argv[i] == '--input-dir' and i + 1 < len(argv):
            input_dir = Path(argv[i + 1])
            i += 2
        elif argv[i].startswith('--'):
            print(f"Unknown argument: {argv[i]}")
            print_help()
            sys.exit(1)
        else:
            # Positional argument = audio file
            audio_paths.append(Path(argv[i]))
            i += 1

    return audio_paths, num_speakers, max_speakers, voiceprints_dir, threshold, input_dir


def discover_audio_files(input_dir: Path) -> list[Path]:
    """Find all supported audio files in a directory."""
    files = sorted(
        f for f in input_dir.iterdir()
        if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS
    )
    return files


def process_file(
    audio_path: Path,
    output_dir: Path,
    num_speakers: int | None,
    max_speakers: int,
    voiceprints: dict,
    threshold: float,
):
    """Process a single audio file: diarize, transcribe, save outputs."""
    print(f"\nAudio file: {audio_path}")

    # Convert to WAV (always needed for diarizer, ensure 16kHz mono)
    temp_wav = convert_to_wav(audio_path)
    audio_for_processing = temp_wav

    try:
        # Step 1: Speaker diarization
        diar_segments = run_diarization(
            audio_for_processing, output_dir,
            num_speakers=num_speakers, max_speakers=max_speakers,
        )

        # Step 1b: Match clusters to voiceprints
        speaker_map = {}
        if voiceprints and diar_segments:
            speaker_map = match_speakers_to_voiceprints(
                diar_segments, audio_for_processing, voiceprints, threshold
            )
            diar_segments = apply_speaker_names(diar_segments, speaker_map)

        unique_speakers = sorted(set(s['speaker'] for s in diar_segments))
        detected_count = len(unique_speakers)

        # Step 2: ASR transcription with timestamps
        full_text, words, asr_segments = run_transcription(audio_for_processing)

        # Step 3: Align speakers to ASR output
        if words and diar_segments:
            labeled_words = assign_speakers_to_words(words, diar_segments)
            speaker_transcript = build_speaker_transcript(labeled_words)
        else:
            speaker_transcript = full_text

        if asr_segments and diar_segments:
            labeled_segments = assign_speakers_to_segments(
                asr_segments, diar_segments
            )
        else:
            # Fallback: use diarization segments with text from full transcript
            labeled_segments = [{
                'start': s['start'],
                'end': s['end'],
                'segment': '',
                'speaker': s['speaker'],
            } for s in diar_segments]

        # Step 4: Save outputs
        txt_path, srt_path = save_outputs(
            full_text, speaker_transcript, labeled_segments,
            detected_count, audio_path, output_dir,
        )

        # Summary
        print("\n" + "=" * 60)
        print("  Transcription + Diarization Complete!")
        print("=" * 60)
        print(f"\n  Speakers found: {detected_count}")
        for spk in unique_speakers:
            count = sum(1 for s in diar_segments if s['speaker'] == spk)
            total = sum(s['end'] - s['start']
                        for s in diar_segments if s['speaker'] == spk)
            print(f"    {spk}: {count} segments, {total:.1f}s total")
        print(f"\nOutput files:")
        print(f"  TXT: {txt_path}")
        print(f"  SRT: {srt_path}")
        print(f"\nPreview:")
        print("-" * 40)
        preview_lines = speaker_transcript.split('\n')[:10]
        print('\n'.join(preview_lines))
        if len(speaker_transcript.split('\n')) > 10:
            print("...")
        print("-" * 40)

    finally:
        # Clean up temp WAV
        if temp_wav and temp_wav.exists():
            temp_wav.unlink()


def main():
    audio_paths, num_speakers, max_speakers, voiceprints_dir, threshold, input_dir = parse_args(sys.argv[1:])

    # Directories
    script_dir = Path(__file__).parent.resolve()
    repo_root = script_dir.parent
    output_dir = repo_root / "output"

    # Determine files to process
    if not audio_paths:
        # No files specified — scan input directory
        scan_dir = input_dir or (repo_root / DEFAULT_INPUT_DIR)
        if not scan_dir.exists():
            scan_dir.mkdir(parents=True, exist_ok=True)
            print(f"Created input directory: {scan_dir}")
            print(f"Place audio files (.wav, .mp3, .flac) in '{scan_dir}' and run again.")
            print(f"Or specify a file directly: python scenario1/transcribe-diarize.py <audio_file>")
            sys.exit(0)
        audio_paths = discover_audio_files(scan_dir)
        if not audio_paths:
            print(f"No audio files found in {scan_dir}")
            print(f"Supported formats: {', '.join(sorted(AUDIO_EXTENSIONS))}")
            print(f"\nPlace audio files there and run again, or specify a file directly:")
            print(f"  python scenario1/transcribe-diarize.py <audio_file>")
            sys.exit(0)
        print(f"Found {len(audio_paths)} audio file(s) in {scan_dir}")
    else:
        # Validate specified files
        for p in audio_paths:
            if not p.exists():
                print(f"Error: File not found: {p}")
                sys.exit(1)
            if p.suffix.lower() not in AUDIO_EXTENSIONS:
                print(f"Error: Unsupported format: {p.suffix}")
                print(f"Supported: {', '.join(AUDIO_EXTENSIONS)}")
                sys.exit(1)

    print("=" * 60)
    print("  Scenario 1: Transcription + Speaker Diarization")
    print("=" * 60)
    print(f"\nFiles to process: {len(audio_paths)}")
    for p in audio_paths:
        print(f"  - {p.name}")

    # Load voiceprints once (shared across all files)
    voiceprints = {}
    if voiceprints_dir:
        voiceprints = load_voiceprints(voiceprints_dir)

    # Process each file
    for i, audio_path in enumerate(audio_paths, 1):
        if len(audio_paths) > 1:
            print(f"\n{'#' * 60}")
            print(f"  File {i}/{len(audio_paths)}")
            print(f"{'#' * 60}")

        process_file(
            audio_path, output_dir,
            num_speakers, max_speakers,
            voiceprints, threshold,
        )

    if len(audio_paths) > 1:
        print(f"\n{'=' * 60}")
        print(f"  All {len(audio_paths)} files processed!")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
