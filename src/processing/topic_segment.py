"""Compatibility imports for the topic segmentation module.

The implementation lives in ``topic_segmenter.py``. This file keeps older
imports such as ``src.processing.topic_segment`` working.
"""

from src.processing.topic_segmenter import TopicSegment, TopicSegmenter

__all__ = ["TopicSegment", "TopicSegmenter"]
