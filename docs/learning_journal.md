Day 1:
Today we setup the project structure. we chose Orbi because It sounds modern, deals with Agentic system which are one of the hot modern technologies and would makes us compettitive in the job market.

we still have to download the sample meetings to test with, which will have multiple speakers with different accents, which we think will be good for the trascription system.

Question we have:
 - how does Whisper works?
 - How does Whisper handles overlapping speed?
 - What happens when audio quality is poor and audio file is broken at the middle?
 - how accurate is speaker diarization really?


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


 