"""
Speaker diarization using pyannote.audio (open-source speaker diarization library)
This file identifies distinct speakers in an audio file and produces time-stamped 
speaker segments. These segments are later aligned with Whisper's transcription to 
produce a "who said what's transcript." because pyannote assigns arbitary IDs like
"Speaker_00". They didn't give real names which is our task to align that.
"""

import os
from dataclasses import dataclass, asdict
from pyannote.audio import Pipeline
import torch

