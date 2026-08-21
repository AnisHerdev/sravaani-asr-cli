import sys, os, torch, argparse, time, tempfile
from transformers import AutoModel
from dotenv import load_dotenv
import soundfile as sf
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box

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
# Model traced with max ~7000 frames; at 16kHz/10ms hop = ~70s. Use 30s chunks for safety.
DEFAULT_CHUNK_SECONDS = 30
DEFAULT_STRIDE_SECONDS = 5


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="SraVaani speech-to-text")
    p.add_argument("audio", nargs="*", default=[DEFAULT_AUDIO],
                   help=f"audio files (default: {DEFAULT_AUDIO})")
    p.add_argument("--chunk-seconds", type=int, default=DEFAULT_CHUNK_SECONDS,
                   help=f"Split long audio into chunks of this many seconds (default: {DEFAULT_CHUNK_SECONDS})")
    p.add_argument("--stride-seconds", type=int, default=DEFAULT_STRIDE_SECONDS,
                   help=f"Overlap between chunks in seconds (default: {DEFAULT_STRIDE_SECONDS})")
    p.add_argument("--no-chunk", action="store_true",
                   help="Disable chunking (may fail on long audio)")
    return p.parse_args(argv)


def get_audio_duration(path: str) -> float:
    """Return audio duration in seconds using soundfile (fast, no full decode)."""
    info = sf.info(path)
    return info.frames / info.samplerate


def chunk_audio(path: str, chunk_seconds: int, stride_seconds: int = 0) -> list[str]:
    """Split audio file into chunks with optional overlap (stride). Returns list of temp file paths."""
    info = sf.info(path)
    samplerate = info.samplerate
    total_frames = info.frames
    chunk_frames = chunk_seconds * samplerate
    stride_frames = stride_seconds * samplerate
    
    if total_frames <= chunk_frames:
        return [path]  # No chunking needed
    
    # Step size = chunk - stride (so chunks overlap by stride_frames)
    step_frames = chunk_frames - stride_frames
    
    temp_files = []
    with sf.SoundFile(path) as f:
        for start in range(0, total_frames, step_frames):
            end = min(start + chunk_frames, total_frames)
            frames_to_read = end - start
            f.seek(start)
            chunk_data = f.read(frames_to_read)
            
            # Create temp file
            tf = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tf.close()
            sf.write(tf.name, chunk_data, samplerate)
            temp_files.append((tf.name, start / samplerate, end / samplerate))
    
    return temp_files


def transcribe_with_chunking(audio_paths: list[str], chunk_seconds: int, stride_seconds: int, no_chunk: bool):
    """Transcribe audio files, chunking long ones if needed with optional stride."""
    all_hyps = []
    all_durations = []
    temp_files_to_cleanup = []
    
    for path in audio_paths:
        dur = get_audio_duration(path)
        
        if no_chunk or dur <= chunk_seconds:
            # Transcribe directly
            hyps = model.transcribe([path], return_hypotheses=True, timestamps=True)
            all_hyps.extend(hyps)
            all_durations.append(dur)
        else:
            # Chunk and transcribe each chunk
            console = Console()
            console.print(f"[yellow]Audio {os.path.basename(path)} is {dur:.1f}s - chunking into {chunk_seconds}s segments with {stride_seconds}s stride...[/yellow]")
            
            chunks = chunk_audio(path, chunk_seconds, stride_seconds)
            temp_files_to_cleanup.extend([c[0] for c in chunks[1:]])
            
            chunk_hyps = []
            chunk_infos = []  # (hyp, chunk_start_time, chunk_end_time)
            for i, (chunk_path, chunk_start, chunk_end) in enumerate(chunks):
                chunk_dur = chunk_end - chunk_start
                hyps = model.transcribe([chunk_path], return_hypotheses=True, timestamps=True)
                chunk_hyps.extend(hyps)
                chunk_infos.append((hyps[0], chunk_start, chunk_end))
                console.print(f"  Chunk {i+1}/{len(chunks)} done ({chunk_dur:.1f}s, {chunk_start:.1f}-{chunk_end:.1f}s)")
            
            # Merge hypotheses with stride deduplication
            # For each chunk after the first, drop words that fall in the overlap region
            merged_words = []
            merged_text_parts = []
            
            for idx, (hyp, chunk_start, chunk_end) in enumerate(chunk_infos):
                if not hyp.timestamp or not hyp.timestamp.get("word"):
                    # Fallback: just concatenate text
                    merged_text_parts.append(hyp.text)
                    continue
                
                words = hyp.timestamp["word"]
                
                if idx == 0:
                    # First chunk: keep all words
                    for w in words:
                        merged_words.append({
                            "start": w["start"] + chunk_start,
                            "end": w["end"] + chunk_start,
                            "word": w["word"]
                        })
                    merged_text_parts.append(hyp.text)
                else:
                    # Subsequent chunks: drop words in the overlap region (stride_seconds before chunk_end of prev)
                    overlap_start = chunk_start  # chunk_start already accounts for stride overlap
                    # Keep only words that start after the overlap region
                    kept_words = [w for w in words if w["start"] >= stride_seconds]
                    
                    for w in kept_words:
                        merged_words.append({
                            "start": w["start"] + chunk_start - stride_seconds,
                            "end": w["end"] + chunk_start - stride_seconds,
                            "word": w["word"]
                        })
                    # Rebuild text from kept words
                    merged_text_parts.append(" ".join(w["word"] for w in kept_words))
            
            merged_text = " ".join(merged_text_parts)
            
            # Create merged hypothesis
            class MergedHyp:
                def __init__(self, text, timestamp=None):
                    self.text = text
                    self.timestamp = timestamp
            
            merged_timestamp = {"word": merged_words} if merged_words else None
            all_hyps.append(MergedHyp(merged_text, merged_timestamp))
            all_durations.append(dur)
    
    # Cleanup temp files
    for tf in temp_files_to_cleanup:
        try:
            os.unlink(tf)
        except Exception:
            pass
    
    return all_hyps, all_durations

def _render_transcripts_table(console: Console, args, hyps):
    """Render transcriptions with file name as title and transcript spanning full width."""
    term_width = console.size.width
    MARGIN = 4
    content_width = max(60, term_width - 2 * MARGIN)

    console.print()
    for path, hyp in zip(args.audio, hyps):
        filename = os.path.basename(path)
        # Title row
        title_table = Table(
            show_header=False,
            box=box.HORIZONTALS,
            width=content_width,
            min_width=60,
        )
        title_table.add_column("Title", style="bold cyan", no_wrap=True, overflow="ellipsis")
        title_table.add_row(filename)
        console.print(title_table)

        # Transcript row - full width
        transcript_table = Table(
            show_header=False,
            box=box.HORIZONTALS,
            width=content_width,
            min_width=60,
        )
        transcript_table.add_column("Transcript", style="white", overflow="fold", no_wrap=False, ratio=1)
        transcript_table.add_row(hyp.text)
        console.print(transcript_table)
        console.print()  # spacing between files


def _render_metrics_table(console: Console, args, durations, hyps, elapsed):
    """Render metrics table with responsive sizing."""
    term_width = console.size.width
    MARGIN = 4
    table_width = max(50, term_width - 2 * MARGIN)

    table = Table(
        title="SraVaani Transcription Results",
        show_header=True,
        header_style="bold cyan",
        expand=False,
        width=table_width,
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

    # Transcribe with optional chunking
    t0 = time.perf_counter()
    hyps, durations = transcribe_with_chunking(args.audio, args.chunk_seconds, args.stride_seconds, args.no_chunk)
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
