import os
import time
import json
import logging
import gc
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Any, Optional

import whisperx
from whisperx.diarize import DiarizationPipeline
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)      # __name__ = name of this file like whisperx_pipeline

@dataclass
class TranscriptionConfig:
    """Configuration for the WhisperX pipeline.
    
    All the knobs you can turn. Default are tuned for 
    Apple Slicon CPU development workflow.
    """
    model_size: str = "base"    # tiny, base, small, medium, large-v2, large-v3
    device: str = "cpu"         # cpu or cuda
    compute_type: str = "int8"  # int8 (fast), float32 (accurate)
    batch_size: int = 4         # 4 for CPU, 16 for GPU
    language: Optional[str] = None      # None = auto-detect by AI model
    min_speakers: Optional[int] = None  # set if you know the speaker count
    max_speakers: Optional[int] = None
    enable_diarization: bool = True     # Set False for single-speker recordings

    def to_dict(self) -> dict:
        return asdict(self)
    
@dataclass
class PipelineResult:
    """Complete output from the WhisperX pipeline."""
    segments: list[dict]            # The main output - aligned, diarized segments
    language: str                   # Detected or forced language
    num_speakers: int               # How many speakers were identified
    word_count: int                 # Total words transcribed
    audio_duration: float           # Duration of audio in seconds
    processing_time: float          # How long processing took
    config: dict                    # The config used (for reproducibility)

    @property
    def full_text(self) -> str:
        """Concatenate all segment text into a single string."""
        return " ".join(
            seg.get("text", "").strip()
            for seg in self.segments
            if seg.get("text", "").strip()
        )
        
    @property
    def speakers(self) -> list[str]:
        """List of unique speaker IDs found."""
        return sorted({seg.get("speaker", "UNKNOWN") for seg in self.segments})
        
    @property
    def realtime_factor(self) -> float:
        """How much faster/slower than realtime we processed.
        < 1.0 means faster than realtime (good)
        > 1.0 means slower than realtime (expected on CPU)
        """
        if self.audio_duration > 0:
            return self.processing_time / self.audio_duration
        return 0
    
    def to_dict(self) -> dict:
        return {
            'language': self.language,
            'num_speakers': self.num_speakers,
            'word_count': self.word_count,
            'audio_duration_seconds': round(self.audio_duration, 2),
            'processing_time_seconds': round(self.processing_time, 2),
            'realtime_factor': round(self.realtime_factor, 2),
            'num_segments': len(self.segments),
            'speakers': self.speakers,
            'config': self.config,
            'segments': self.segments
        }
    
    def save(self, output_path: str) -> str:
        """Save results to JSON file."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info(f'Results saved to {path}')
        return str(path)
    
    def format_readable(self, max_segments: int | None = None) -> str:
        """Format as a human readable speaker labeled transcript."""
        lines = []
        current_speaker = None

        segments = self.segments[:max_segments] if max_segments else self.segments

        for seg in segments:
            speaker = seg.get("speaker", "UNKNOWN")
            text = seg.get("text", "").strip()
            start = seg.get("start", 0)

            if speaker != current_speaker:
                current_speaker = speaker
                lines.append(f"\n[{self._format_time(start)}] {speaker}:")

            lines.append(f"  {text}")

        return "\n".join(lines)
    
    def _format_time(self, seconds: float) -> str:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins:02d}:{secs:02d}"
    
class WhisperXPipeline:
    """Complete audio processing pipeline using WhisperX.
    
    Handles the full flow: audio -> transcription -> alignment -> diarization.
    Manages model loading/unloading for memory efficiency on CPU.
    """

    def __init__(self, config: TranscriptionConfig | None = None):
        self.config = config or TranscriptionConfig()
        self.hf_token = os.getenv("HF_TOKEN")

        if self.config.enable_diarization and not self.hf_token:
            logger.warning(
                "HF_token not set - diarization will be disabled. "
                "Set HF_TOKEN in .env to enable speaker identification."
            )
            self.config.enable_diarization = False

    def process(self, audio_path: str) -> PipelineResult:
        """Process an audio file through the complete WhisperX pipeline.
        
        Steps:
        1. Load audio
        2. Transcribe (with VAD filtering and batched inference)
        3. Align (word-level timestamps via Wav2Vec2)
        4. Diarize (speaker identification via pyannote)
        5. Assign speakers to words

        Memory Management: On CPU/mac, we load and unload models
        between steps to prevent memory issues with larger models.

        Args:
        audio_path: Path to audio file (wav, mp3, m4a, mp4, etc)

        Returns:
        PipelineResult with aligned, diarized segments.
        """
        audio_path = str(audio_path)

        if not Path(audio_path).exists():
            raise FileNotFoundError(f'Audio file not found: {audio_path}')
        
        logger.info(f'Processing: {audio_path}')
        logger.info(f'Config: {self.config.to_dict()}')

        pipeline_start = time.time()

        # Step 1: Load Audio
        logger.info("Loading Audio...")
        audio = whisperx.load_audio(audio_path)
        audio_duration = len(audio) / 16000     # WhisperX loads at 16KHz
        logger.info(f'Audio duration: {audio_duration:.1f}s'
                    f'({audio_duration/60:.1f} minutes)')
        
        # Step 2: Transcribe
        logger.info(f"Transcribing with Whisper '{self.config.model_size}'")
        step_start = time.time()

        model = whisperx.load_model(
            self.config.model_size,
            self.config.device,
            compute_type = self.config.compute_type,
            )
        
        transcribe_kwargs: dict[str, Any] = {"batch_size": self.config.batch_size}
        if self.config.language:
            transcribe_kwargs["language"] = self.config.language

        result = model.transcribe(audio, **transcribe_kwargs)
        detected_language = result.get("language", "en")
        logger.info(f"  Transcribed in {time.time() - step_start:.1f}s, "
                    f"language = {detected_language}, "
                    f"segments = {len(result['segments'])}")
        
        # Free transcription model memory
        del model
        gc.collect()

        # Step 3: Align
        logger.info(f"Aligning word timestamps(Wav2Vec2)")
        step_start = time.time()

        model_a, metadata = whisperx.load_align_model(
            language_code = detected_language,
            device = self.config.device,
        )

        result = whisperx.align(
            result['segments'],
            model_a,
            metadata,
            audio,
            self.config.device,
            return_char_alignments = False,
        )

        logger.info(f"  Aligned in {time.time() - step_start:.1f}s")

        # Free alignment model
        del model_a
        gc.collect()

        # Step 4 and 5: Diarize and Assign Speakers
        num_speakers = 1        # Default if diarization is disabled

        if self.config.enable_diarization:
            logger.info(f"Diarizing speakers(pyannote)")
            step_start = time.time()
            
            diarize_model = DiarizationPipeline(
                token = self.hf_token,
                device = self.config.device,
            )

            diarize_kwargs = {}
            if self.config.min_speakers is not None:
                diarize_kwargs['min_speakers'] = self.config.min_speakers
            if self.config.max_speakers is not None:
                diarize_kwargs['max_speakers'] = self.config.max_speakers

            diarize_segments = diarize_model(
                audio,
                **diarize_kwargs,
            )
            logger.info(f" Diarized in {time.time() - step_start:.1f}s")

            # Step 5: Assign speakers to words
            logger.info(f"Assigning speakers to words")
            result = whisperx.assign_word_speakers(diarize_segments, result)

            # Count unique speakers
            speaker_ids = set()
            for seg in result.get("segments", []):
                if "speaker" in seg:
                    speaker_ids.add(seg["speaker"])
            num_speakers = len(speaker_ids)
            logger.info(f" Found {num_speakers} speakers")

            del diarize_model
            gc.collect()
        else:
            logger.info("Diarization disabled (single speaker mode)")
            # Adding default speaker label
            for seg in result.get("segments", []):
                seg["speaker"] = "SPEAKER_00"

        # Conclude all the result that we have been building till now
        processing_time = time.time() - pipeline_start
        segments = result.get("segments", [])
        word_count = sum(
            len(seg.get("text","").split())
            for seg in segments
        )
        pipeline_result = PipelineResult(
            segments = segments,
            language = detected_language,
            num_speakers = num_speakers,
            word_count = word_count,
            audio_duration = audio_duration,
            processing_time = processing_time,
            config = self.config.to_dict(),
        )
        logger.info(
            f"Pipeline complete: {word_count} words, "
            f"{num_speakers} speakers, "
            f"{len(segments)} segments, "
            f"{processing_time:.1f}s total "
            f"({pipeline_result.realtime_factor:.1f}x realtime)"
        )
        return pipeline_result


        
        
