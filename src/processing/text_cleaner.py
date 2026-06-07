import logging # for logging progress and issues
import re # for regex-based pre-cleaning
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class NlpTranscriptCleaner:
    """Conservative transcript cleaner for WhisperX segment text.

    The cleaner removes low-value speech artifacts while preserving meaning,
    timing metadata, speaker labels, acronyms, and original wording.
    """

    SINGLE_FILLERS = {"um", "uh", "umm", "basically"}
    PHRASE_FILLERS = {"you know", "i mean"}

    def __init__(self, model_name: str = "en_core_web_sm"):
        """Load the spaCy pipeline once so batch cleaning stays efficient."""
        try:
            import spacy

            self.nlp = spacy.load(model_name, exclude=["ner", "lemmatizer", "textcat"])
            logger.info(f"Successfully loaded spaCy model: {model_name}")
        except ImportError as exc:
            raise ImportError(
                "spaCy is required for transcript cleaning. "
                "Install dependencies with `python3 -m pip install -r requirements.txt`."
            ) from exc
        except OSError:
            logger.error(
                f"Model '{model_name}' not found. "
                f"Did you run 'python -m spacy download {model_name}'?"
            )
            raise

    def clean_text(self, text: str) -> str:
        """Clean one transcript string."""
        if not text or not text.strip():
            return ""

        doc = self.nlp(self._preclean_text(text))
        return self._process_doc(doc)

    def clean_segments_batch(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Clean WhisperX segments while preserving the original metadata."""
        logger.info(f"Cleaning batch of {len(segments)} segments...")

        texts = [
            self._preclean_text(seg.get("text", ""))
            for seg in segments
        ]

        cleaned_segments = []
        for i, doc in enumerate(self.nlp.pipe(texts)):
            cleaned_text = self._process_doc(doc)

            if cleaned_text:
                cleaned_segments.append(segments[i] | {"text": cleaned_text})

        return cleaned_segments

    def _preclean_text(self, text: str) -> str:
        """Apply cheap regex cleanup before the text enters spaCy."""
        text = text.strip()
        for phrase in self.PHRASE_FILLERS:
            pattern = rf"\b{re.escape(phrase)}\b[\s,]*"
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)
        return re.sub(r"\s+", " ", text).strip()

    def _process_doc(self, doc) -> str:
        """Apply token-level cleanup to an already parsed spaCy doc."""
        cleaned_tokens = []
        last_kept_lower = None

        for token in doc:
            token_lower = token.text.lower()

            if token_lower in self.SINGLE_FILLERS:
                continue

            if token_lower == "like" and token.pos_ == "INTJ":
                continue

            if last_kept_lower == token_lower and token.pos_ not in ["AUX", "VERB"]:
                continue

            if token.is_punct and not cleaned_tokens:
                continue

            cleaned_tokens.append(token.text_with_ws)

            if not token.is_punct:
                last_kept_lower = token_lower

        return self._normalize_output("".join(cleaned_tokens))

    def _normalize_output(self, text: str) -> str:
        """Normalize final spacing and first-letter casing only."""
        text = re.sub(r"\s+", " ", text).strip()
        if text and text[0].islower():
            return text[0].upper() + text[1:]
        return text
