# Day 1:
Today we setup the project structure. we chose Orbi because It sounds modern, deals with Agentic system which are one of the hot modern technologies and would makes us compettitive in the job market.

we still have to download the sample meetings to test with, which will have multiple speakers with different accents, which we think will be good for the trascription system.

Question we have:
 - how does Whisper works?
 - How does Whisper handles overlapping speed?
 - What happens when audio quality is poor and audio file is broken at the middle?
 - how accurate is speaker diarization really?

# Day 2:
We downloaded the OpenAI whisper model on our local machine and also the sample audio. Then, We run our model and gets the initial audio-to-text conversion. We get segmemnts- chunks of text with timestamps:
"segments": {
      "id": 0,
      "seek": 0,
      "start": 0.0,
      "end": 8.42,
      "text": " So I was having dinner recently and I happened to be with somebody who runs a large fashion",
      "tokens": [50364,407286,390,...........],
      "temperature": 0.0,
      "avg_logprob": -0.1849344693697416,
      "compression_ratio": 1.6810344827586208,
      "no_speech_prob": 0.20176942646503448
    }
Two things to notice here:
1) avg_logprob = How confident the model is. Lower(more negative) means less confident. We can use this to flag segments that
might be wrong.
2) no_speech_prob = How likely the model thinks this segment is actually silence or noise, not speech. High values mean "this 
probably isn't real speech."

Also, we knew about "Sampling rate" of the audio and whisper model always expect 16000Hz sample rate to run it. There's one 
library called "ffmpeg" which does this thing internally to convert audio to 16kHz and feeds the whisper model.

# Day 3:
I built the audio processor feature to convert the audio to whisper friendly type, basically cleaning the audio before feeding
it to model. I also added error handling case and runs the test case to check this feature.

# Day 3:
# whisper_transcriber
# learnings from whisper_transcriber and why I build this:
# 1. The transcriber adds structure, cleanliness, and abstraction, not better transcription.
# 2. The raw output carries thousands of token IDs, seek positions, temperature, and compression_ratio — internal Whisper bookkeeping we 'll never need downstream. Our transcriber drops all of that, cutting file size by 55%.
# 3. Whisper outputs text with leading spaces like " So I was having dinner...". Our transcriber strips those, giving cleaner segments for any NLP pipeline downstream.
'''Raw Whisper returns a dict. Our code wraps it in typed dataclasses with: 
    Computed properties (.duration)
    Named fields (start_time vs start)
    .to_dict() for serialization
# This means any Python code that uses our results gets autocomplete, type checking, and will crash immediately on wrong field names instead of silently producing bugs.