from pathlib import Path

from src.pipeline import MeetingPipeline, MeetingPipelineConfig
from src.processing.text_cleaner import NlpTranscriptCleaner
from src.processing.topic_segmenter import TopicSegmenter


class FakeToken:
    def __init__(
        self,
        text: str,
        *,
        pos: str = "NOUN",
        is_punct: bool = False,
        trailing_ws: bool = False,
    ):
        self.text = text
        self.pos_ = pos
        self.is_punct = is_punct
        self.text_with_ws = text + (" " if trailing_ws else "")


class FakeNlp:
    def __call__(self, text: str):
        return self._tokenize(text)

    def pipe(self, texts):
        for text in texts:
            yield self._tokenize(text)

    def _tokenize(self, text: str):
        words = text.split()
        return [
            FakeToken(word, trailing_ws=index < len(words) - 1)
            for index, word in enumerate(words)
        ]


class FakeEmbeddingModel:
    def encode(self, texts, show_progress_bar=False, normalize_embeddings=True):
        embeddings = []
        for text in texts:
            if "budget" in text.lower():
                embeddings.append([0.0, 1.0])
            else:
                embeddings.append([1.0, 0.0])
        return embeddings


def build_cleaner_without_spacy() -> NlpTranscriptCleaner:
    cleaner = object.__new__(NlpTranscriptCleaner)
    cleaner.nlp = FakeNlp()
    return cleaner


def build_segmenter_without_model() -> TopicSegmenter:
    segmenter = object.__new__(TopicSegmenter)
    segmenter.model = FakeEmbeddingModel()
    return segmenter


def test_text_cleaner_removes_fillers_duplicates_and_empty_segments():
    cleaner = build_cleaner_without_spacy()
    segments = [
        {"text": " um you know roadmap roadmap looks good ", "speaker": "A", "start": 0, "end": 2},
        {"text": "uh", "speaker": "B", "start": 3, "end": 4},
    ]

    cleaned = cleaner.clean_segments_batch(segments)

    assert cleaned == [
        {"text": "Roadmap looks good", "speaker": "A", "start": 0, "end": 2}
    ]


def test_topic_segmenter_splits_when_similarity_drops():
    segmenter = build_segmenter_without_model()
    segments = [
        {"text": "Roadmap intro", "speaker": "A", "start": 0, "end": 1},
        {"text": "Roadmap milestones", "speaker": "B", "start": 1, "end": 2},
        {"text": "Roadmap risks", "speaker": "A", "start": 2, "end": 3},
        {"text": "Budget review", "speaker": "B", "start": 3, "end": 4},
        {"text": "Budget owners", "speaker": "A", "start": 4, "end": 5},
        {"text": "Budget next steps", "speaker": "B", "start": 5, "end": 6},
    ]

    topics = segmenter.segment(segments, threshold=0.5, min_segment_size=2, window_size=1)

    assert len(topics) == 2
    assert topics[0].start_index == 0
    assert topics[0].end_index == 3
    assert topics[0].full_text == "Roadmap intro Roadmap milestones Roadmap risks"
    assert topics[1].start_index == 3
    assert topics[1].end_index == 6
    assert topics[1].speakers_involved == ["A", "B"]


def test_meeting_pipeline_flow_with_fakes(monkeypatch, tmp_path):
    class FakeTranscriptResult:
        segments = [
            {"text": "um roadmap roadmap intro", "speaker": "A", "start": 0, "end": 1},
            {"text": "roadmap milestones", "speaker": "B", "start": 1, "end": 2},
            {"text": "budget review", "speaker": "A", "start": 2, "end": 3},
            {"text": "budget owners", "speaker": "B", "start": 3, "end": 4},
        ]
        audio_duration = 4.0
        num_speakers = 2
        speakers = ["A", "B"]

        def to_dict(self):
            return {"segments": self.segments}

    class FakeWhisperXPipeline:
        def __init__(self, config):
            self.config = config

        def process(self, audio_path):
            return FakeTranscriptResult()

    monkeypatch.setattr("src.pipeline.WhisperXPipeline", FakeWhisperXPipeline)
    monkeypatch.setattr("src.pipeline.NlpTranscriptCleaner", build_cleaner_without_spacy)
    monkeypatch.setattr("src.pipeline.TopicSegmenter", build_segmenter_without_model)

    config = MeetingPipelineConfig(
        output_dir=str(tmp_path),
        topic_threshold=0.5,
        topic_min_size=2,
        topic_window_size=1,
    )
    result = MeetingPipeline(config).process("fake_audio.wav", "fake-meeting")

    assert result["meeting_id"] == "fake-meeting"
    assert result["metadata"]["num_cleaned_segments"] == 4
    assert result["metadata"]["num_topics"] == 2
    assert result["cleaned_segments"][0]["text"] == "Roadmap intro"
    assert Path(tmp_path, "fake-meeting.json").exists()
