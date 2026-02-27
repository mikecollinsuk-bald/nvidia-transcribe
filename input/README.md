# Input Directory

Place audio files here for automatic processing by `transcribe-diarize.py`.

## Supported Formats

- `.wav`
- `.mp3`
- `.flac`

## Usage

```bash
# Process all files in this directory
python scenario1/transcribe-diarize.py

# Or specify a custom input directory
python scenario1/transcribe-diarize.py --input-dir /path/to/recordings/
```

Files are processed alphabetically. Output goes to `output/`.
