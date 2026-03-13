# Scenario 1: Simple CLI Transcription

Command-line interface for quick transcription of a single audio file.

## Model

- **Name**: nvidia/parakeet-tdt-0.6b-v2
- **Language**: English only
- **License**: CC-BY-4.0 (commercial use allowed)

## Usage

```bash
# From repository root
python scenario1/transcribe.py <audio_file>

# Examples
python scenario1/transcribe.py my_audio.mp3
python scenario1/transcribe.py /path/to/audio.wav

# Help
python scenario1/transcribe.py --help
```

## Supported Formats

- `.wav` - Native format (16kHz mono recommended)
- `.flac` - Auto-converted to 16kHz WAV
- `.mp3` - Auto-converted to 16kHz WAV

## Output

Generates two files in the `output/` directory (at repo root):
- `{timestamp}_{filename}.txt` - Full transcription with timestamps
- `{timestamp}_{filename}.srt` - Subtitle file for video editors

## Speaker Diarization

Use `transcribe-diarize.py` to transcribe audio **with speaker identification** (who said what):

```bash
# Auto-detect speakers
python scenario1/transcribe-diarize.py meeting.mp3

# Specify number of speakers
python scenario1/transcribe-diarize.py meeting.mp3 --speakers 2

# Limit auto-detection range
python scenario1/transcribe-diarize.py meeting.mp3 --max-speakers 4

# Use voiceprints to label known speakers by name
python scenario1/transcribe-diarize.py meeting.mp3 --voiceprints voiceprints/

#mike's quick copy paste
python scenario1/transcribe-diarize.py --voiceprints voiceprints/


# Adjust match sensitivity (default: 0.5, higher = stricter)
python scenario1/transcribe-diarize.py meeting.mp3 --voiceprints voiceprints/ --threshold 0.6

# Reduce speaker bleed at turn boundaries (default: 2, higher = more smoothing, 0 = off)
python scenario1/transcribe-diarize.py meeting.mp3 --min-words 3
```

### Voiceprints (Speaker Recognition)

Label speakers by name using reference audio clips:

1. Create a `voiceprints/` directory
2. Add short audio clips (5–30s) of each known speaker
3. Name files after speakers: `Mike_Collins.wav`, `Jane_Doe.mp3` (underscores → spaces)
4. Run with `--voiceprints voiceprints/`

Unmatched speakers keep their cluster IDs (e.g., `speaker_0`).

### Diarization Models

| Model | Purpose | Size |
|-------|---------|------|
| `titanet_large` | Speaker embeddings | ~90MB |
| `vad_marblenet` | Voice activity detection (English) | ~20MB |

### Diarization Output

Generates speaker-labeled files in `output/`:
- `{timestamp}_{filename}_diarized.txt` - Transcript with speaker turns
- `{timestamp}_{filename}_diarized.srt` - Subtitles with speaker labels

Example output:
```
[00:00:00.000] speaker_0: Welcome to the meeting.
[00:00:02.500] speaker_1: Thanks for having me.
[00:00:05.100] speaker_0: Let's get started with the agenda.
```

## Best For

- Command-line workflows
- Batch processing and automation
- CI/CD pipelines
- Quick single-file transcription
- Meeting transcription with speaker identification
