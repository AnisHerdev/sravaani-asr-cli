# SraVaani ASR CLI

Command-line interface for running speech-to-text inference with the **SraVaani-1.0** model from ARTPARK-IISc. Transcribes audio files with word-level timestamps and displays performance metrics (latency, real-time factor) in a clean terminal UI.

## Features

- **Multi-file transcription** — process one or more audio files in a single run
- **Word-level timestamps** — precise timing for each transcribed word
- **Performance metrics** — audio duration, transcription time, and real-time factor (RTF)
- **Rich terminal output** — responsive tables and panels that adapt to terminal width
- **Offline mode** — works without internet after initial model download
- **GPU/CPU auto-detection** — uses CUDA when available, falls back to CPU

## Installation

```bash
git clone https://github.com/yourusername/sravaani-asr-cli.git
cd sravaani-asr-cli
pip install -r requirements.txt
```

## Configuration

Create a `.env` file with your Hugging Face token:

```bash
cp .env.example .env
# Edit .env and add your HF_TOKEN
```

Get a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) with access to `ARTPARK-IISc/SraVaani-1.0`.

## Usage

```bash
# Transcribe default audio file
python main.py

# Transcribe specific files
python main.py audio1.wav audio2.mp3

# Run in offline mode (requires cached model)
OFFLINE_MODE=1 python main.py audio.wav
```

## Output

The CLI displays two tables:

1. **Metrics table** — per-file audio duration, transcription time, and RTF
2. **Transcripts table** — file name and transcribed text (or panels on narrow terminals)

Example:
```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                    SraVaani Transcription Results                            ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ File                    │ Audio (s) │ Transcribe (s) │ Real-time Factor │
├─────────────────────────┼───────────┼────────────────┼──────────────────┤
│ Alaakaa_loovaa...mp3    │    45.32  │         2.14   │           0.05x  │
└─────────────────────────┴───────────┴────────────────┴──────────────────┘
```

## Requirements

- Python 3.10+
- PyTorch 2.12+
- Transformers 5.13+
- See `requirements.txt` for full list

## Model

**SraVaani-1.0** by ARTPARK-IISc — an automatic speech recognition model for Indian languages.

- Hugging Face: [`ARTPARK-IISc/SraVaani-1.0`](https://huggingface.co/ARTPARK-IISc/SraVaani-1.0)
- License: Check model card for details

## License

MIT License — see [LICENSE](LICENSE) for details.