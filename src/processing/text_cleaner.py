# this feature rakes raw whsiperz ouput and produces clean, readable text while preserving meaning,
# intentionally conservative - better to leave messingness than lose content.

"""
#main features here:
1. Removes Filler words (um, uhh, you know)
2. mergees stuttered repeatations ( the the -> the)
3. fixes orphaned punctuation (lowercaser to uppercase first letter)
4. preserves meaningful self-corrections ("no wait, actually...")
"""

"""
What it avoids
1. fixing grammar
2. sentence restructuring to preserve meaning, removing anything that might carry meaning.
"""

#we start
import spacy
import re
import logging
#using dataclass while dealing with machine learning is a bad practice
from typing import List, Dict, Any

logger = logging.getLogger(__name__) # Module-level logger

class NlpTranscriptCleaner: #NLP pipeline for cleaning transcript text, uses spacy for grammatical accuracy while removing filler and sttuers.
    FILLER_WORDS = {"um", "uh", "umm", "you know", "i mean", "basically"}

    def __init__(self, model_name: str = "en_core_web_sm"):
        """Initializes the cleaner and loads the NLP model into memory"""
        try:
            # Disabling pipeline components we don't need for pure speed
            # We need 'tagger' and 'attribute_ruler' for POS, and 'parser' for dependencies
            self.nlp = spacy.load(model_name, exclude=["ner", "lemmatizer", "textcat"])
            logger.info(f"Successfully loaded spaCy model: {model_name}")
        except OSError:
            logger.error(f"Model '{model_name}' not found. Did you run 'python -m spacy download {model_name}'?")
            raise

    def clean_text(self, text: str) -> str:
        # Cleaning single string of text using POS tagging
        if not text or not text.strip():
            return ""
        doc = self.nlp(text)
        cleaned_tokens = []

        for i, token in enumerate(doc): # Iterate through tokens in the document
            token_lower = token.text.lower()

            #1. Handle Explict Fillers & Interjections
            if token_lower in self.FILLER_WORDS:
                continue  # Skip filler words entirely

            # Handle the world 'like' safely (Only remove if its an interjection/discourse marker)
            if token_lower == "like" and token.pos_ in ["INT", "ADP"] and token.dep_ == "intj":
                continue

            # 2. Handle True Stuttered Repetitions (e.g., "the the" -> "the")
            if cleaned_tokens and token_lower == cleaned_tokens[-1].lower():
                if token.pos_ not in ["AUX", "VERB"]: # It's safe to drop repeated nouns, determiners, etc.
                    continue
            
            # 3. Handle Punctuation attached to deleted words
            # If we deleted the previous word, we don't want a random comma floating around.
            if token.is_punct and not cleaned_tokens:
                continue # Don't start a sentence with a comma
            
            # If it passed all checks, keep the exact original text (preserving original case)
            # We use token.text_with_ws to preserve natural spacing
            cleaned_tokens.append(token.text_with_ws)

        # Reconstruct the string and do a final whitespace cleanup
        final_text = "".join(cleaned_tokens).strip()
        
        # Ensure capitalization is preserved if it was lowered by cleaning
        if final_text and final_text[0].islower():
            final_text = final_text[0].upper() + final_text[1:]

        return final_text

    def clean_segments_batch(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Cleans a list of WhisperX segments efficiently using nlp.pipe.
        This is the method you would call in your worker queue.
        """
        logger.info(f"Cleaning batch of {len(segments)} segments...")
        
        # Extract just the texts for efficient batch processing
        texts = [seg.get("text", "").strip() for seg in segments]
        
        # Use nlp.pipe for highly optimized C-level batch processing
        cleaned_segments = []
        for i, doc in enumerate(self.nlp.pipe(texts)):
            # We apply the same logic as clean_text, but leverage the pre-computed docs
            # (In a fully refactored version, you'd extract the loop logic to share it)
            cleaned_text = self._process_doc(doc)
            
            if cleaned_text:
                # Merge the cleaned text back with the original timing metadata
                cleaned_segments.append(segments[i] | {"text": cleaned_text})

        return cleaned_segments

    def _process_doc(self, doc) -> str:
        # (This contains the same loop logic as clean_text, abstracted for reuse)
        cleaned_tokens = []
        for token in doc:
             # ... same validation logic ...
             if token.text.lower() in self.FILLER_WORDS: continue
             if token.text.lower() == "like" and token.pos_ in ["INTJ"]: continue
             if cleaned_tokens and token.text.lower() == cleaned_tokens[-1].lower() and token.pos_ not in ["AUX", "VERB"]: continue
             if token.is_punct and not cleaned_tokens: continue
             cleaned_tokens.append(token.text_with_ws)
        return "".join(cleaned_tokens).strip().capitalize()