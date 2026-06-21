# Day 1:
Today, we set up the project structure. We chose Orbi because it sounds modern, deals with the Agentic system, which are one of the hot modern technologies, and would make us competitive in the job market.

We still have to download the sample meetings to test with, which will have multiple speakers with different accents, which we think will be good for the transcription system.

Question we have:
 - How does Whisper work?
 - How does Whisper handle overlapping speed?
 - What happens when audio quality is poor, and the audio file is broken in the middle?
 - How accurate is speaker diarization really?

# Day 2:
We downloaded the OpenAI whisper model on our local machine and also the sample audio. Then, We run our model and gets the initial audio-to-text conversion. We get segments- chunks of text with timestamps:

"segments": {
      "id": 0,
      "seek": 0,
      "start": 0.0,
      "end": 8.42,
      "text": " So I was having dinner recently, and I happened to be with somebody who runs a large fashion",
      "tokens": [50364,407286,390,...........],
      "temperature": 0.0,
      "avg_logprob": -0.1849344693697416,
      "compression_ratio": 1.6810344827586208,
      "no_speech_prob": 0.20176942646503448
    }

# Two things to notice here:
1) avg_logprob = How confident the model is. Lower(more negative) means less confident. We can use this to flag segments that
might be wrong.
2) no_speech_prob = How likely the model thinks this segment is actually silence or noise, not speech. High values mean "this 
probably isn't real speech."

Also, we knew about the "Sampling rate" of the audio and whisper model, which always expect 16000Hz sample rate to run it. There's one
library called "ffmpeg" which does this thing internally to convert audio to 16kHz and feeds the whisper model.

# Day 3:
I built the audio processor feature to convert the audio to whisper-friendly type, basically cleaning the audio before feeding
it to the model. I also added error handling case and runs the test case to check this feature.

# Day 4:
whisper_transcriber and learnings from it, and why I built this:
1. The transcriber adds structure, cleanliness, and abstraction, not better transcription.
2. The raw output carries thousands of token IDs, seek positions, temperature, and compression_ratio — internal Whisper bookkeeping we'll never need downstream. Our transcriber drops all of that, cutting file size by 55%.
3. Whisper outputs text with leading spaces like " So I was having dinner...". Our transcriber strips those, giving cleaner segments for any NLP pipeline downstream.

Raw Whisper returns a dict. Our code wraps it in typed dataclasses with: 
- Computed properties (.duration)
- Named fields (start_time vs start)
- .to_dict() for serialization
    
This means any Python code that uses our results gets autocomplete, type checking, and will crash immediately on wrong field names instead of silently producing bugs.
 Day 5, 6, 7:
 Testing pipeline to test both Audio porcessor and wishper transcriber together: testing both together reveled that they both work correctly as expected togther, audio gets processed and then gets passed to trasnscriber to transcriber, where both runs their methods successfully and giving out the results such as processing timel, checks were performed with different models such as tiny, base medium and large, and was foudn out that each larger model performs little better than previous smaller model in recognixing words and full context, while it was also found out that in some cases smaller model were better able to identify what speaker was speaking which was very surprising, so it was concluding that each models have there uniqe strengths even if the sizes differs.

- initally we had gotthen rid of confidence score asw it was not accurate insted farbricated by almost -+50%.
- another major things is, when users have thicker accents eg: phlipino, or affircan, the model struggles to identify their speach but works great when listens modern english native speakers, which points that mdoel maybe trained highly on english native speakers first.
- there were some minor mistakes the team learned how useful git commands and git ignore is and to write explicit comments to tell what the code is and hwo its works -  very useful for rest of the member of the team.

# Week 4:
This week we completed the processing layer of the project. The main goal was to take the transcript segments that come from WhisperX and make them more useful for the rest of the meeting intelligence system.

We worked on two important files:

1. `src/processing/text_cleaner.py`

This file cleans the transcript text after transcription. WhisperX gives us useful segments with timestamps and speakers, but the text can still have extra speech artifacts. The cleaner removes things like:

- filler words such as `um`, `uh`, `umm`, and `basically`
- filler phrases such as `you know` and `i mean`
- repeated duplicate words
- leading punctuation
- extra whitespace
- empty segments after cleaning

The important thing we learned is that cleaning text should be conservative. We do not want to rewrite the meaning of what was said. We only remove low-value artifacts while preserving the speaker label, start time, end time, and other metadata.

2. `src/processing/topic_segmenter.py`

This file groups cleaned transcript segments into topic sections. The idea is that a meeting is not just one long text block. It usually moves through different topics, and our system should detect those topic changes.

The topic segmenter works like this:

Audio transcript segments
        ↓
Cleaned text segments
        ↓
Build context windows around neighboring utterances
        ↓
Create embeddings using SentenceTransformer
        ↓
Compare neighboring embeddings using cosine similarity
        ↓
Low similarity means the conversation probably shifted topic
        ↓
Return `TopicSegment` objects with metadata and full topic text

The output contains useful information like:

- `topic_id`
- `title`
- `start_index`
- `end_index`
- `num_utterances`
- `speakers`
- `duration_seconds`
- `full_text`
- `text_preview`

We also reviewed `src/processing/topic_segment.py` and found that it was only a compatibility wrapper that imported from `topic_segmenter.py`. The actual implementation lives in `topic_segmenter.py`, and the project was not using `topic_segment.py`, so it was unnecessary.

# Week 4 Testing:
We added focused tests for the processing layer in `tests/test_processing_pipeline.py`.

These tests check:

- text cleaner removes filler words and repeated words
- text cleaner removes segments that become empty
- text cleaner preserves metadata like speaker and timestamps
- topic segmenter splits topics when semantic similarity drops
- mocked meeting pipeline flow works from transcript to cleaned text to topic segments to JSON output

We also cleaned up old quick test files. Some files under `tests/` were actually scripts that loaded models immediately when pytest collected them. That caused problems because tests would download models or run audio processing even when we only wanted fast tests. We converted those into real pytest test functions and marked heavy model/audio tests as `slow`.

Important commands that passed:

```bash
.venv/bin/python -m pytest tests/test_processing_pipeline.py -q
```

Result:

```text
3 passed
```

```bash
.venv/bin/python -m pytest -m "not slow" -q
```

Result:

```text
11 passed, 8 deselected
```

```bash
.venv/bin/python -m pytest tests/test_pipeline_e2e.py -q
```

Result:

```text
1 passed
```

We also tested diarization with Hugging Face `HF_TOKEN` and pyannote. The test passed and printed speaker time ranges like:

```text
[    1.1s ->     3.2s] SPEAKER_02
```

This means pyannote detected that one anonymous speaker label spoke during that time range. These labels are not real names yet, just speaker IDs.

Final Week 4 pipeline:

Audio
 -> WhisperX transcription
 -> text cleaning
 -> topic segmentation
 -> final meeting JSON

At the end of Week 4, the text cleaner, topic segmenter, and full pipeline flow were working and tested successfully.
