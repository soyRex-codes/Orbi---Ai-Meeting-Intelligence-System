import whisper
model = whisper.load_model("small")
result = model.transcribe("data/sample_audio/meeting_01.webm")

print(result["text"][:500])
print(f"\nNumber of segments: {len(result['segments'])}")
print(f"\nFirst Segment: {result['segments'][0]}")