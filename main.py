import sys, torch
from transformers import AutoModel
import os
from dotenv import load_dotenv

load_dotenv()

DOWNLOAD_MODEL=True

if DOWNLOAD_MODEL:
access_token=os.environ.get("HF_TOKEN")
REPO = "ARTPARK-IISc/SraVaani-1.0"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModel.from_pretrained(REPO, trust_remote_code=True,token=access_token).to(DEV).eval()

fpath="Alaakaa_loovaa_unplugged_herdev.mp3"
hyps = model.transcribe(fpath, return_hypotheses=True)
for path, h in zip(sys.argv[1:], hyps):
    print(f"{path}\t{h.text}")