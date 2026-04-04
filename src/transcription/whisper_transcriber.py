# 1. The transcriber adds structure, cleanliness, and abstraction, not better transcription.
# 2. The raw output carries thousands of token IDs, seek positions, temperature, and compression_ratio — internal Whisper bookkeeping we    'll never need downstream. Our transcriber drops all of that, cutting file size by 55%.
# 3. Whisper outputs text with leading spaces like " So I was having dinner...". Our transcriber strips those, giving cleaner segments for any NLP pipeline downstream.
'''Raw Whisper returns a dict. Our code wraps it in typed dataclasses with: 
    Computed properties (.duration)
    Named fields (start_time vs start)
    .to_dict() for serialization
# This means any Python code that uses our results gets autocomplete, type checking, and will crash immediately on wrong field names instead of silently producing bugs.
'''

"""Whisper transcription engine — converts audio to timestamped text."""

from ast import If
import logging
import time
from dataclasses import dataclass, asdict
from typing import Any

import torch
import whisper

logger = logging.getLogger(__name__) #extra 
#getLogger(__name__) creates a logger named after this file. If this file is src/transcription/whisper_transcriber.py,
# the logger is named that — so when a log message appears you know exactly which file it came from

VALID_MODEL_SIZES = frozenset({"tiny", "base", "small", "medium", "large"})
#frozenset is like a set but immutable — nobody can accidentally add or remove values from it at runtime. 


'''@dataclass is a decorator that tells Python: auto-generate __init__, __repr__, and __eq__ methods for free based on the fields you list.
Without it you'd have to write def __init__(self, text, start_time, ...) manually. Each field is just a typed variable declaration — Python
enforces the types at the IDE level.'''
@dataclass
class TranscriptSegment:
    """One continuous chunk of transcribed speech with timing and quality signals."""
    text: str
    start_time: float       # seconds from start of audio
    end_time: float         # seconds from end of audio 
    avg_logprob: float      # Whisper's raw metric: closer to 0.0 = more confident
    is_likely_speech: bool  # False if Whisper thinks this is noise/silence
    
    '''The @property called duration is a computed value — it looks like a plain attribute when you call segment.
    duration but it actually runs the subtraction every time you access it. No need to store it separately.'''
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time ## Calculates the difference between end and start times


@dataclass
class TranscriptionResult: #"""Complete transcription output for one audio file."""
    #summary: TranscriptSegment is for individual chunks of transcribed speech with timing info, while TranscriptionResult is the overall
    # output for an entire audio file, containing a list of segments and summary info like full text and language.
    segments: list[TranscriptSegment] #a list of all the segments in the audio, each with its own text and timing info
    full_text: str #the full concatenated text of all segments, without timestamps
    language: str #e.g. "english"
    model_size: str #e.g. "base"

    def to_dict(self) -> dict:
        return asdict(self)


class WhisperTranscriber:    #Transcribes/converts audio to text using OpenAI's Whisper model.
    def __init__(self, model_size: str = "medium"): #The __init__ method is the constructor for the class WhisperTranscriber. It initializes the transcriber with a specified model size (defaulting to "base") and loads the Whisper model onto the appropriate device (GPU if available, otherwise CPU).
        if model_size not in VALID_MODEL_SIZES:
            raise ValueError(f"Model size must be one of {VALID_MODEL_SIZES}")

        self.model_size = model_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu" #Checks if a CUDA-compatible GPU is available. If so, it sets the device to "cuda" for faster processing; otherwise, it falls back to "cpu".

        logger.info("Loading Whisper '%s' on %s...", model_size, self.device)
        self.model = whisper.load_model(model_size, device=self.device)

    def transcribe(self, audio_path: str, language: str | None = None) -> TranscriptionResult:
        """Transcribe an audio file and return structured, timestamped results.

        Args:
            audio_path: Path to audio file (WAV 16kHz mono preferred).
            language: Force a language code (e.g. "en"), or None to auto-detect.
        """
        start = time.perf_counter() # time counter for how long transcription process takes.

        # fp16(half-precision) is TRUE for GPU
        options: dict[str, Any] = {"fp16": self.device == "cuda"}
        # why? Whisper's transcribe() accepts mixed-type kwargs (bool, str, int, etc.). Any accurately reflects that this is a heterogeneous options bag.
        if language:
            options["language"] = language # language auto-detect, but if the caller specified a language, we pass that to Whisper to skip detection and speed things up.

        raw: dict[str, Any] = self.model.transcribe(audio_path, **options) # **options is a placeholder for any additional keyword arguments we want to pass to the transcribe method. In this case, it could include fp16 and language if they were set. This allows us to keep the code flexible and easily add more options in the future without changing the method signature.
        # Calls the transcribe method of the loaded Whisper model, passing in the audio file path and
        # any options (like fp16/use GPU and language). This returns a raw dictionary containing the transcription results, including segments, full text,and detected language.
        elapsed = time.perf_counter() - start #

        segments = [
            TranscriptSegment(
                text=s["text"].strip(),
                start_time=s["start"],
                end_time=s["end"],
                avg_logprob=s.get("avg_logprob", -1.0),
                is_likely_speech=s.get("no_speech_prob", 0) < 0.5,
            )
            for s in raw.get("segments", [])
        ]

        audio_duration = segments[-1].end_time if segments else 0
        realtime_factor = elapsed / audio_duration if audio_duration > 0 else 0

        logger.info(
            "Transcribed %d segments in %.1fs (%.2fx realtime)",
            len(segments), elapsed, realtime_factor,
        )

        return TranscriptionResult(
            segments=segments,
            full_text=raw.get("text", "").strip(),
            language=raw.get("language", "unknown"),
            model_size=self.model_size, 
        )