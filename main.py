import sys, os, torch, argparse, time
from transformers import AutoModel
from dotenv import load_dotenv
import soundfile as sf
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

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


def _render_transcripts_table(console: Console, args, hyps):
    """Render transcriptions as a responsive table, or Panels if terminal is too narrow."""
    term_width = console.size.width
    # Use Panels for very narrow terminals (< 70 chars) - table borders break badly
    if term_width < 70:
        console.print()
        for path, hyp in zip(args.audio, hyps):
            console.print(Panel(
                hyp.text,
                title=f"[bold cyan]{os.path.basename(path)}[/bold cyan]",
                border_style="green",
                expand=False,
                padding=(1, 2)
            ))
        return

    # Normal: use table with responsive column sizing
    transcript_table = Table(
        title="Transcriptions",
        show_header=True,
        header_style="bold green",
        expand=True,  # Use full terminal width
        min_width=60,  # Don't shrink below this
    )
    transcript_table.add_column(
        "File",
        style="cyan",
        no_wrap=True,
        min_width=20,
        max_width=30,
    )
    transcript_table.add_column(
        "Transcript",
        style="white",
        max_width=120,
        min_width=30,
        overflow="fold",
        no_wrap=False,
        ratio=1,  # Take remaining space
    )

    for path, hyp in zip(args.audio, hyps):
        transcript_table.add_row(os.path.basename(path), hyp.text)

    console.print()
    console.print(transcript_table)


def _render_metrics_table(console: Console, args, durations, hyps, elapsed):
    """Render metrics table with responsive sizing."""
    term_width = console.size.width

    table = Table(
        title="SraVaani Transcription Results",
        show_header=True,
        header_style="bold cyan",
        expand=True,
        min_width=50,
    )
    table.add_column("File", style="white", min_width=15, max_width=35, overflow="ellipsis", no_wrap=False)
    table.add_column("Audio (s)", justify="right", style="green", min_width=8)
    table.add_column("Transcribe (s)", justify="right", style="yellow", min_width=10)
    table.add_column("Real-time Factor", justify="right", style="magenta", min_width=12)

    for path, dur, hyp in zip(args.audio, durations, hyps):
        audio_s = dur
        if hyp.timestamp and hyp.timestamp.get("word"):
            last_word = hyp.timestamp["word"][-1]
            audio_s = last_word["end"]
        rtf = elapsed / audio_s if audio_s > 0 else 0.0
        display_name = os.path.basename(path)
        if len(display_name) > 40:
            display_name = display_name[:37] + "..."
        table.add_row(display_name, f"{audio_s:.2f}", f"{elapsed:.2f}", f"{rtf:.2f}x")

    console.print(table)


def main(argv=None):
    args = parse_args(argv)
    console = Console()

    # Get durations for all files first
    durations = [get_audio_duration(path) for path in args.audio]

    # Transcribe with timestamps to get word-level timing
    t0 = time.perf_counter()
    hyps = model.transcribe(args.audio, return_hypotheses=True, timestamps=True)
    elapsed = time.perf_counter() - t0

    _render_metrics_table(console, args, durations, hyps, elapsed)
    _render_transcripts_table(console, args, hyps)


if __name__ == "__main__":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    main()