import sys, os, torch, argparse, time
from transformers import AutoModel
from dotenv import load_dotenv
import soundfile as sf
from rich.console import Console
from rich.table import Table

load_dotenv()

access_token = os.environ.get("HF_TOKEN")
OFFLINE_MODE = os.getenv("OFFLINE_MODE", "0").strip().lower() in ("1", "true", "yes")

REPO = "ARTPARK-IISc/SraVaani-1.0"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModel.from_pretrained(
    REPO,
    trust_remote_code=True,
    token=access_token,
    local_files_only=OFFLINE_MODE,
).to(DEV).eval()

DEFAULT_AUDIO = "Alaakaa_loovaa_unplugged_herdev.mp3"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="SraVaani speech-to-text")
    p.add_argument("audio", nargs="*", default=[DEFAULT_AUDIO],
                   help=f"audio files (default: {DEFAULT_AUDIO})")
    return p.parse_args(argv)


def get_audio_duration(path: str) -> float:
    """Return audio duration in seconds using soundfile (fast, no full decode)."""
    info = sf.info(path)
    return info.frames / info.samplerate


def main(argv=None):
    args = parse_args(argv)
    console = Console()

    # Get durations for all files first
    durations = [get_audio_duration(path) for path in args.audio]

    # Transcribe with timestamps to get word-level timing
    t0 = time.perf_counter()
    hyps = model.transcribe(args.audio, return_hypotheses=True, timestamps=True)
    elapsed = time.perf_counter() - t0

    # Build results table
    table = Table(title="SraVaani Transcription Results", show_header=True, header_style="bold cyan")
    table.add_column("File", style="white", no_wrap=False)
    table.add_column("Audio (s)", justify="right", style="green")
    table.add_column("Transcribe (s)", justify="right", style="yellow")
    table.add_column("Real-time Factor", justify="right", style="magenta")

    total_audio = 0.0
    for path, dur, hyp in zip(args.audio, durations, hyps):
        total_audio += dur
        # Use last word's end time if timestamps available, else fall back to soundfile duration
        audio_s = dur
        if hyp.timestamp and hyp.timestamp.get("word"):
            last_word = hyp.timestamp["word"][-1]
            audio_s = last_word["end"]
        rtf = elapsed / audio_s if audio_s > 0 else 0.0
        # Truncate filename for display
        display_name = os.path.basename(path)
        if len(display_name) > 40:
            display_name = display_name[:37] + "..."
        table.add_row(display_name, f"{audio_s:.2f}", f"{elapsed:.2f}", f"{rtf:.2f}x")

    console.print(table)

    # Print transcripts
    console.print()  # blank line
    for path, hyp in zip(args.audio, hyps):
        console.print(f"[bold]{os.path.basename(path)}[/bold]")
        console.print(hyp.text)
        console.print()


if __name__ == "__main__":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    main()