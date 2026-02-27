# Voiceprints

Place short audio clips (5-30 seconds) of known speakers in this directory.

## Naming Convention

File names become speaker labels. Underscores are replaced with spaces:
- `Mike_Collins.wav`  **Mike Collins**
- `Jane_Doe.mp3`  **Jane Doe**

## Supported Formats

- `.wav` (preferred - no conversion needed)
- `.mp3`
- `.flac`

## Tips

- Use clear speech with minimal background noise
- 10-30 seconds of speech works best
- One file per speaker
- Multiple files per speaker are not yet supported (use the best sample)

## Usage

```bash
python scenario1/transcribe-diarize.py meeting.mp3 --voiceprints voiceprints/
```
