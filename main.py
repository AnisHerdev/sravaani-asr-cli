import sys, os, torch, argparse
from transformers import AutoModel
from dotenv import load_dotenv

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


def main(argv=None):
    args = parse_args(argv)
    for path in args.audio:
        print(f"Transcribing file: {path}...")
    hyps = model.transcribe(args.audio, return_hypotheses=True)
    for path, h in zip(args.audio, hyps):
        print(f"{path}\t{h.text}")


if __name__ == "__main__":
    try:
        import sys
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    main()