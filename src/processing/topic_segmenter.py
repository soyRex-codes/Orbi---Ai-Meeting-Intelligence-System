"""Embedding-based topic segmentation for cleaned meeting transcripts.
Code file Flow overview:
                Transcript segments with text/speaker/timing
                        ↓
                Clean text list with speaker/timing metadata removed
                        ↓
                Context windows built from neighboring utterances
                        ↓
                Embedding vectors for each window using SentenceTransformer
                        ↓
                Cosine similarities between neighbors
                        ↓
                Low similarity = topic boundary
                        ↓
                TopicSegment objects with metadata and full text for each topic

The segmenter detects semantic shifts between neighboring transcript windows. 
A large similarity drop means the conversation likely moved to a new topic.
"""

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass # dataclass saves you from writing self again and agan inside a class, when the task is to just hold data.
class TopicSegment: # Represents a contiguous section of the transcript that belongs to the same topic.
    topic_id: int # eg: 0, 1, 2... assigned sequentially as topics are created.
    title: str # eg: "Topic 1", "Topic 2"... can be enhanced later with actual topic labels if desired.
    start_index: int # Index of the first transcript segment that belongs to this topic.
    end_index: int # Index of the last transcript segment that belongs to this topic (exclusive).
    segments: list[dict] = field(default_factory=list) # List of transcript segments (utterances) that belong to this topic.

    @property
    def num_utterances(self) -> int: # Number of transcript segments grouped into this topic.
        return len(self.segments)

    @property
    def full_text(self) -> str: #This joins all text in the topic into one string..
        """All topic text joined into one readable string."""
        return " ".join(seg.get("text", "").strip() # Get the text of each segment, defaulting to empty string if missing, and strip whitespace. 
            for seg in self.segments
            if seg.get("text", "").strip()
        ).strip()

    @property
    def speakers_involved(self) -> list[str]: # This returns unique speakers in the topic. eg: Alice, Bob, UNKNOWN (if speaker label is missing).
        """Unique speaker labels that appear in this topic."""
        return sorted({seg.get("speaker", "UNKNOWN") for seg in self.segments})

    @property
    def duration_seconds(self) -> float: #not compulsory, but useful
        """
        Topic duration based on first segment start and last segment end.
        eg: {"start": 10, "end": 15}
        """
        
        if not self.segments:
            return 0.0
        """ self.segments[-1] gets the last segment in the topic, and .get("end", 0) retrieves its end time, defaulting to 0 if missing.
        Similarly, self.segments[0] gets the first segment and .get("start", 0) retrieves its start time."""
        end_time = self.segments[-1].get("end", 0)
        start_time = self.segments[0].get("start", 0)
        return float(end_time) - float(start_time)

    def to_dict(self) -> dict:
        """
        This converts the TopicSegment object into a normal dictionary. Why?
        Because APIs, JSON files, databases, and frontend apps usually want plain dictionaries, not custom Python objects.
        """
        """Serialize the topic for JSON output or API responses."""
        full_text = self.full_text
        return {
            "topic_id": self.topic_id, # eg: topic_id = 0
            "title": self.title, # eg: title = "Topic 1"
            "start_index": self.start_index, # eg: start_index = 0
            "end_index": self.end_index, # eg: end_index = 5
            "num_utterances": self.num_utterances, # eg: num_utterances = 5, utterances is the number of segments in this topic
            "speakers": self.speakers_involved, # eg: speakers = ["Alice", "Bob"]
            "duration_seconds": round(self.duration_seconds, 1), # eg: duration_seconds = 15
            "full_text": full_text, # eg: full_text = "Hello, how are you? I'm good, thanks! How about you? Doing well, just working on project."
            "text_preview": full_text[:200] + "..." if len(full_text) > 200 else full_text,
            # last line represents: if text is longer than 200 characters, show first 200 + "...".
            # otherwise show full text, this is useful for frontend preview cards.
            }
        
class TopicSegmenter:
    """Divide cleaned transcript segments into semantic topic sections."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Load the sentence embedding model once and reuse it for all meetings."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers and its PyTorch dependencies are required "
                "for topic segmentation. Use the project's Python 3.12 environment "
                "and install dependencies with "
                "`.venv/bin/python -m pip install -r requirements.txt`. "
                f"Original import error: {exc}"
            ) from exc

        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        logger.info("Embedding model ready")

    def segment(
        self,
        segments: list[dict],
        threshold: float = 0.40,
        min_segment_size: int = 3,
        window_size: int = 3,
    ) -> list[TopicSegment]:
        """Split transcript segments when semantic similarity drops.
        Args:
            segments: Cleaned WhisperX-style segments with text/timing/speaker keys.
            threshold: Boundary cutoff. Higher means more topics; lower means fewer.
            min_segment_size: Minimum number of utterances between topic boundaries.
            window_size: Number of neighboring utterances averaged per embedding.
        """
        if not segments:
            return []

        if min_segment_size < 1:
            raise ValueError("min_segment_size must be at least 1")

        if window_size < 1:
            raise ValueError("window_size must be at least 1")

        if len(segments) < min_segment_size * 2:
            logger.info("Transcript too short to segment meaningfully")
            return [
                TopicSegment(
                    topic_id=0,
                    title="Full Discussion",
                    start_index=0,
                    end_index=len(segments),
                    segments=segments,
                )
            ]
        """EXCPECTED OUTPUT:
                "text": "Some utterance",
                "speaker": "Alice",
                "start": 0,
                "end": 5
        """

        texts = [seg.get("text", "").strip() for seg in segments] 
        windowed_texts = self._build_windows(texts, window_size) 
        """ this creates overlapping windows of text to smooth out the embeddings. For example, with window_size=3,
        the first window might be "Utterance 1 Utterance 2", the second window "Utterance 1 Utterance 2 Utterance 3",
        the third window "Utterance 2 Utterance 3 Utterance 4", and so on.
        This helpscapture context around eachutterance when computing embeddings. """

        logger.info(f"Computing embeddings for {len(windowed_texts)} windows")
        import numpy as np

        embeddings = self.model.encode( # This turns each window into an embedding vector.
            # eg: ["A B"] into [0.1, 0.2]
            windowed_texts,
            show_progress_bar=False,
            normalize_embeddings=True,
        )

        # Embeddings are normalized, so dot product equals cosine similarity.
        similarities = [
            float(np.dot(embeddings[i], embeddings[i + 1]))
            for i in range(len(embeddings) - 1)
        ]

        # This finds the indices where the similarity drops below the threshold, indicating a topic change.
        boundaries = self._find_boundaries(similarities, threshold, min_segment_size)
        topics = self._build_topics(segments, boundaries)

        logger.info(
            f"Segmentation complete: {len(topics)} topics "
            f"(threshold={threshold}, min_size={min_segment_size})"
        )
        return topics

    def _build_windows(self, texts: list[str], window_size: int) -> list[str]:
        """Combine neighboring utterances to create stable embedding inputs."""
        windows = []
        half = window_size // 2

        for i in range(len(texts)):
            start = max(0, i - half)
            end = min(len(texts), i + half + 1)
            windows.append(" ".join(texts[start:end]).strip())

        return windows

    def _find_boundaries(self, similarities: list[float], threshold: float, min_size: int) -> list[int]:
        """Find segment indices where a new topic should begin. eg: [0, 5, 10] means topics are segments[0:5], segments[5:10], segments[10:end]."""
        boundaries = [0]

        for i, sim in enumerate(similarities):
            boundary_index = i + 1  # Similarity i compares window i with i + 1.

            if sim < threshold and boundary_index - boundaries[-1] >= min_size:
                boundaries.append(boundary_index)

        return boundaries

    def _build_topics(self, segments: list[dict], boundaries: list[int]) -> list[TopicSegment]:
        """Create TopicSegment objects from boundary start indices."""
        topics = []

        for topic_id, start in enumerate(boundaries):
            end = boundaries[topic_id + 1] if topic_id + 1 < len(boundaries) else len(segments)
            topics.append(TopicSegment(
                topic_id=topic_id,
                title=f"Topic {topic_id + 1}",
                start_index=start,
                end_index=end,
                segments=segments[start:end],
            ))

        return topics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    segmenter = TopicSegmenter()
    segments = [
        {"text": "Hello, how are you?", "speaker": "Alice", "start": 0, "end": 5},
        {"text": "I'm good, thanks! How about you?", "speaker": "Bob", "start": 6, "end": 10},
        {"text": "Doing well, just working on a project.", "speaker": "Alice", "start": 11, "end": 15},
        {"text": "That's great to hear. What project?", "speaker": "Bob", "start": 16, "end": 20},
        {"text": "It's a topic segmentation task using sentence transformers.", "speaker": "Alice", "start": 21, "end": 25},
        {"text": "Sounds interesting! Good luck with that.", "speaker": "Bob", "start": 26, "end": 30},
    ]
    topics = segmenter.segment(segments)
    for topic in topics:
        print(topic.to_dict())
