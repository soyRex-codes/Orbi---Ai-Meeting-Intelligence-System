import whisper
import json

#model loading and transcript extraction from audio
model = whisper.load_model("medium")
result = model.transcribe("data/sample_audio/meeting_02.mp3")

#saving the transcript
with open("data/outputs/test_transcript_raw.json", "w") as f:
    json.dump(result, f, indent=2)

#simple print of the response in terminal for quality checks
print(result["text"][:500])
print(f"\nNumber of segments: {len(result['segments'])}")
print(f"\nFirst segment: {result['segments'][0]}")