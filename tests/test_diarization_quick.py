from pyannote.audio import Pipeline
import torch
import os
from dotenv import load_dotenv

load_dotenv()
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    token = os.getenv("HF_TOKEN")
)
# Use GPU if available
if torch.cuda.is_available():
    pipeline = pipeline.to(torch.device("cuda"))

# Running diarizaion on processed audio
audio_path = 'data/processed_audio/meeting_01_processed.wav'
diarization = pipeline(audio_path)

for turn, _, speaker in diarization.speaker_diarization.itertracks(yield_label=True):
    print(f"[{turn.start:7.1f}s -> {turn.end:7.1f}s] {speaker}")


"""
It gives the list of time ranges, each labeled with a speaker ID like 
"Speaker_00", "Speaker_01", etc. The speakers didn't have real names and 
that's what we need to figure out to align with out whisper model. The print 
statement gives output like:
[    1.1s ->     3.2s] SPEAKER_02  ---> Means Speaker 2 speaks from 1.1 to 3.2 seconds.
[    3.5s ->     6.1s] SPEAKER_02
[    6.6s ->     7.5s] SPEAKER_02
[    7.9s ->    11.1s] SPEAKER_02
.
.
.
. so on]
"""