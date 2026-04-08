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

