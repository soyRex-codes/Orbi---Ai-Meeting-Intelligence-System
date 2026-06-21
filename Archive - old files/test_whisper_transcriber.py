import json
from pathlib import Path

import pytest

from src.transcription.audio_processor import AudioProcessor
from src.transcription.whisper_transcriber import WhisperTranscriber


@pytest.mark.slow
def test_whisper_transcriber_quick():
    """Prepare sample audio and transcribe it with the local Whisper wrapper."""
    audio_path = Path("data/sample_audio/meeting_01.webm")
    output_path = Path("data/outputs/whisper_transcriper_result.json")

    assert audio_path.exists(), f"Sample audio missing: {audio_path}"

    processor = AudioProcessor()
    clean_audio = processor.prepare_audio(str(audio_path))[1]

    transcriber = WhisperTranscriber(model_size="base")
    result = transcriber.transcribe(clean_audio)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)

    assert result.language
    assert result.segments

    print(f"\nlanguage detected: {result.language}")
    print(f"segments: {result.segments}")
    print(f"Transcription result saved to {output_path}")
