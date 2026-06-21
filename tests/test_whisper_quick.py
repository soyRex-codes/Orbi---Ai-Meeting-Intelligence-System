import json
from pathlib import Path

import pytest


@pytest.mark.slow
def test_whisper_quick_transcription():
    """Verify OpenAI Whisper can transcribe the sample audio."""
    import whisper

    audio_path = Path("data/sample_audio/meeting_01.webm")
    output_path = Path("data/outputs/test_transcript_raw.json")

    assert audio_path.exists(), f"Sample audio missing: {audio_path}"

    model = whisper.load_model("base")
    result = model.transcribe(str(audio_path))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    assert result["text"].strip()
    assert result["segments"]

    print(result["text"][:500])
    print(f"\nNumber of segments: {len(result['segments'])}")
    print(f"\nFirst segment: {result['segments'][0]}")
