import logging
from pathlib import Path

import pytest

from src.pipeline import MeetingPipeline, MeetingPipelineConfig


@pytest.mark.slow
def test_pipeline_e2e_meeting_01():
    """Run the full meeting pipeline on the sample audio file."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    audio_path = Path("data/sample_audio/meeting_01.webm")
    assert audio_path.exists(), f"Sample audio missing: {audio_path}"

    config = MeetingPipelineConfig(
        whisperx_model="base",
        language="en",
        topic_threshold=0.40,
    )

    pipeline = MeetingPipeline(config)
    result = pipeline.process(str(audio_path), "test-meeting-01")

    assert result["meeting_id"] == "test-meeting-01"
    assert result["metadata"]["num_topics"] >= 1
    assert result["metadata"]["num_cleaned_segments"] >= 1
    assert result["topics"]

    print(f"\nTopics found: {result['metadata']['num_topics']}")
    for topic in result["topics"]:
        print(
            f"Topic {topic['topic_id']}: {topic['num_utterances']} segments, "
            f"{topic['duration_seconds']:.0f}s - {topic['text_preview'][:80]}..."
        )
