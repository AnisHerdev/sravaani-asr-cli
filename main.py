import sys, os, torch
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

fpath = "Alaakaa_loovaa_unplugged_herdev.mp3"
print(f"Transcribing file: {fpath}...")
hyps = model.transcribe(fpath, return_hypotheses=True)
for path, h in zip(sys.argv[1:], hyps):
    print(f"{path}\t{h.text}")

# --- lower-level alternative (explicit processor) ---
# from transformers import AutoProcessor
# import soundfile as sf   # or: import wave (stdlib) for PCM WAV
# proc = AutoProcessor.from_pretrained(REPO, trust_remote_code=True)
# wav, sr = sf.read(path, dtype="float32")   # average channels if stereo
# inputs = proc(wav, sampling_rate=sr, return_tensors="pt").to(DEV)
# text = proc.batch_decode(model.generate(**inputs))[0]