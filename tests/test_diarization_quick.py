import os

import pytest


@pytest.mark.slow
def test_diarization_quick():
    """Run pyannote diarization on the processed sample audio."""
    token = os.getenv("HF_TOKEN")
    if not token:
        pytest.skip("HF_TOKEN is required for gated pyannote diarization model")

    from pyannote.audio import Pipeline
    from huggingface_hub.errors import GatedRepoError
    import torch

    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=token,
        )
    except GatedRepoError:
        pytest.skip(
            "HF_TOKEN is valid, but this Hugging Face account does not have "
            "access to pyannote/speaker-diarization-3.1 yet"
        )

    if torch.cuda.is_available():
        pipeline = pipeline.to(torch.device("cuda"))

    diarization = pipeline("data/processed_audio/meeting_01_processed.wav")
    tracks = list(diarization.speaker_diarization.itertracks(yield_label=True))

    assert tracks
    for turn, _, speaker in tracks[:10]:
        print(f"[{turn.start:7.1f}s -> {turn.end:7.1f}s] {speaker}")
