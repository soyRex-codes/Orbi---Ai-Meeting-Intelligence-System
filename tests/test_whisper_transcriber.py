
from src.transcription.audio_processor import AudioProcessor
from src.transcription.whisper_transcriber import WhisperTranscriber
import json

#prepare Audio
processer = AudioProcessor()
clean_audio = processer.prepare_audio('data/sample_audio/meeting_01.webm')[1] # We did slicing here to get only the str path of the audio not the numpy array.
# explain more? 

# Transcribe Audio
transcriber = WhisperTranscriber(model_size="base")
result = transcriber.transcribe(clean_audio) #clean_audio is the path of the processed audio
# print languge, segments, processing time
print(f"\nlanguage detected: {result.language}")
print(f"segments: {result.segments}")

#save the full transcription result to a json file
with open('data/outputs/whisper_transcriper_result.json', 'w') as f:
    json.dump(result.to_dict(), f, indent = 2)

print("Transcription result saved to data/outputs/whisper_transcriper_result.json")

if __name__ == "__main__":
    pass


#


