from pathlib import Path

import pytest


@pytest.mark.slow
def test_whisperx_quick_transcription():
    """Verify WhisperX can load, transcribe, and return segments for sample audio."""
    import whisperx

    device = "cpu"
    compute_type = "int8"
    batch_size = 4
    model_size = "base"
    audio_file = Path("data/sample_audio/meeting_01.webm")

    assert audio_file.exists(), f"Sample audio missing: {audio_file}"

    print(f"Device: {device}, Compute: {compute_type}, Model: {model_size}")
    print(f"Audio: {audio_file}")

    model = whisperx.load_model(model_size, device, compute_type=compute_type)
    audio = whisperx.load_audio(str(audio_file))
    result = model.transcribe(audio, batch_size=batch_size)

    assert result["language"]
    assert result["segments"]

    print("\nTranscription complete")
    print(f"Language detected: {result['language']}")
    print(f"Segments: {len(result['segments'])}")
    for seg in result["segments"][:3]:
        print(f" [{seg['start']:.1f}s -- {seg['end']:.1f}s] {seg['text']}")
