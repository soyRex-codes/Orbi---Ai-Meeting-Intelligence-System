# End-to-end test of the WhisperX pipeline

import json
import logging
import pytest
from src.transcription.whisperx_pipeline import WhisperXPipeline, TranscriptionConfig

# Enable logging so we can see progress
logging.basicConfig(level = logging.INFO, format = "%(asctime)s [%(levelname)s] %(message)s")

@pytest.mark.slow
def test_full_pipeline():
    """Run the complete pipeline on a sample meeting."""

    config = TranscriptionConfig(
        model_size = "base",
        device = "cpu",
        compute_type = "int8",
        batch_size = 4,
        language = "en",
        enable_diarization = True,
    )
    pipeline = WhisperXPipeline(config)
    result = pipeline.process("data/sample_audio/meeting_01.webm")

    # Print Summary
    print('Pipeline Result Summary')
    print()
    print(f"Audio duration:   {result.audio_duration:.1f}s ({result.audio_duration/60:.1f} min)")
    print(f"Processing time:  {result.processing_time:.1f}s")
    print(f"Realtime factor:  {result.realtime_factor:.1f}x")
    print(f"Words:            {result.word_count}")
    print(f"Segments:         {len(result.segments)}")
    print(f"Speakers:         {result.num_speakers} → {result.speakers}")
    print(f"Language:         {result.language}")

    # Print first 15 Segments
    print('Transcript (first 15 segments)')
    print()
    print(result.format_readable(max_segments = 15))

    # Show word level detail for first segment
    print('Word-level detail (first segment)')
    print()
    first_seg = result.segments[0] if result.segments else {}
    words = first_seg.get('words', [])
    for w in words[:10]:
        score = w.get("score", 0)
        speaker = w.get("speaker", "?")
        confidence = "yes" if score > 0.5 else "?"
        print(f"[{confidence}] {w.get('start', 0):.2f}s"
              f"\"{w.get('word','')}\" "
              f"(score: {score:.2f}, speaker: {speaker})")
        
    # Speaker Talk Time
    print("Speaker Analysis")
    print()
    talk_time = {}
    for seg in result.segments:
        speaker = seg.get("speaker", "Unknown")
        duration = seg.get("end", 0) - seg.get("start", 0)
        talk_time[speaker] = talk_time.get(speaker, 0) + duration

    total = sum(talk_time.values())
    for speaker, seconds in sorted(talk_time.items()):
        pct = seconds / total * 100 if total > 0 else 0
        print(f"{speaker}: {seconds:.1f}s ({pct:.0f}%)")

    # Save results
    output_path = result.save("data/outputs/meeting_01_whisperx.json")
    print(f"\nFull results saved to: {output_path}")

    # Basic Assertions
    assert len(result.segments) > 0, "Should have transcribed segments"
    assert result.word_count > 0, "Should have transcribed words"
    if config.enable_diarization:
        assert result.num_speakers >=1, "Should detect at least one speaker"

    print(f"\nFull pipeline test passed")
    return result

@pytest.mark.slow
def test_compare_model_sizes():
    """Compare base vs small model on the same audio."""

    audio = "data/sample_audio/meeting_01.webm"
    
    for model_size in ["base", "small"]:
        print(f"Testing model: {model_size}")

        config = TranscriptionConfig(
            model_size=model_size,
            device="cpu",
            compute_type="int8",
            batch_size=4,
            language="en",
            enable_diarization=False,  # Skip to make comparison faster
        )
        
        pipeline = WhisperXPipeline(config)
        result = pipeline.process(audio)
        
        print(f"  Time: {result.processing_time:.1f}s "
              f"({result.realtime_factor:.1f}x realtime)")
        print(f"  Words: {result.word_count}")
        print(f"  First sentence: {result.segments[0]['text'][:80]}...")
        
        result.save(f"data/outputs/comparison_{model_size}.json")

if __name__ == "__main__":
    result = test_full_pipeline()
    # Uncomment the next line to run model comparison:
    # test_compare_model_sizes()
