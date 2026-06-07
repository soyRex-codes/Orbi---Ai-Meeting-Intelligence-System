"""End-to-end meeting pipeline.

Audio -> WhisperX transcript -> cleaned transcript -> topic segments -> JSON.
"""

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

from src.processing.text_cleaner import NlpTranscriptCleaner
from src.processing.topic_segmenter import TopicSegmenter
from src.transcription.whisperx_pipeline import TranscriptionConfig, WhisperXPipeline

logger = logging.getLogger(__name__)


@dataclass
class MeetingPipelineConfig:
    """Configuration for the full meeting processing pipeline."""

    # WhisperX transcription settings.
    whisperx_model: str = "base"
    language: str = "en"
    enable_diarization: bool = True
    min_speakers: int | None = None
    max_speakers: int | None = None

    # Topic segmentation settings.
    topic_threshold: float = 0.40
    topic_min_size: int = 3
    topic_window_size: int = 3

    # Output location.
    output_dir: str = "data/processed_meetings"


class MeetingPipeline:
    """Run the full meeting intelligence pipeline for one audio file."""

    def __init__(self, config: MeetingPipelineConfig | None = None):
        self.config = config or MeetingPipelineConfig()

    def process(self, audio_path: str, meeting_id: str | None = None) -> dict:
        """Process one audio file and return a JSON-serializable result."""
        if meeting_id is None:
            meeting_id = Path(audio_path).stem

        pipeline_start = time.time()

        logger.info("=" * 50)
        logger.info("STEP 1: WhisperX pipeline")
        logger.info("=" * 50)

        whisperx_config = TranscriptionConfig(
            model_size=self.config.whisperx_model,
            language=self.config.language,
            enable_diarization=self.config.enable_diarization,
            min_speakers=self.config.min_speakers,
            max_speakers=self.config.max_speakers,
        )
        transcript_result = WhisperXPipeline(whisperx_config).process(audio_path)

        logger.info("=" * 50)
        logger.info("STEP 2: Text cleaning")
        logger.info("=" * 50)

        cleaner = NlpTranscriptCleaner()
        cleaned_segments = cleaner.clean_segments_batch(transcript_result.segments)
        cleaning_stats = self._build_cleaning_stats(
            original_segments=transcript_result.segments,
            cleaned_segments=cleaned_segments,
        )

        logger.info("=" * 50)
        logger.info("STEP 3: Topic segmentation")
        logger.info("=" * 50)

        segmenter = TopicSegmenter()
        topics = segmenter.segment(
            cleaned_segments,
            threshold=self.config.topic_threshold,
            min_segment_size=self.config.topic_min_size,
            window_size=self.config.topic_window_size,
        )

        total_time = time.time() - pipeline_start
        result = {
            "meeting_id": meeting_id,
            "transcript": transcript_result.to_dict(),
            "cleaned_segments": cleaned_segments,
            "cleaning_stats": cleaning_stats,
            "topics": [topic.to_dict() for topic in topics],
            "metadata": {
                "total_processing_seconds": round(total_time, 2),
                "audio_duration_seconds": round(transcript_result.audio_duration, 2),
                "num_speakers": transcript_result.num_speakers,
                "speakers": transcript_result.speakers,
                "num_topics": len(topics),
                "num_cleaned_segments": len(cleaned_segments),
            },
        }

        output_path = self._save_result(meeting_id, result)
        logger.info(f"Pipeline complete: {meeting_id}")
        logger.info(f"Saved: {output_path}")

        return result

    def _build_cleaning_stats(
        self,
        original_segments: list[dict],
        cleaned_segments: list[dict],
    ) -> dict:
        """Summarize how much text was removed during cleaning."""
        original_words = sum(len(seg.get("text", "").split()) for seg in original_segments)
        cleaned_words = sum(len(seg.get("text", "").split()) for seg in cleaned_segments)

        return {
            "original_segments": len(original_segments),
            "cleaned_segments": len(cleaned_segments),
            "removed_empty_segments": len(original_segments) - len(cleaned_segments),
            "original_words": original_words,
            "cleaned_words": cleaned_words,
            "removed_words": original_words - cleaned_words,
        }

    def _save_result(self, meeting_id: str, result: dict) -> Path:
        """Save the processed meeting result to disk."""
        output_path = Path(self.config.output_dir) / f"{meeting_id}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        return output_path
