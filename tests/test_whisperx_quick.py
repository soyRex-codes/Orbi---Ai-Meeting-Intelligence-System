"""
Quick test: does whisperx run on this device?
This will verfies installation, model download,
and basic transcription.
"""
import whisperx
import os
from dotenv import load_dotenv

load_dotenv()

# Device Configuration
DEVICE = "cpu"          # MPS is experimental, CPU is more stable
COMPUTE_TYPE = "int8"   # Fastest on CPU, use "float32" for max accuracy
BATCH_SIZE = 4          # Low for CPU, GPU users would use 16
MODEL_SIZE = "base"     # Fast for testing, can upgrade to "small" or "medium" later

audio_file = 'data/sample_audio/meeting_01.webm'

print(f'Device: {DEVICE}, Compute: {COMPUTE_TYPE}, Model: {MODEL_SIZE}')
print(f'Audio: {audio_file}')
print()

# Loading model and transcribe
print('Loading WhisperX model......')
model = whisperx.load_model(MODEL_SIZE, DEVICE, compute_type = COMPUTE_TYPE)

print('Loadind audio.......')
audio = whisperx.load_audio(audio_file)

print('Transcribing(this may take some time)....')
result = model.transcribe(audio, batch_size = BATCH_SIZE)

print(f'\nTranscription Complete')
print(f'Language detected: {result['language']}')
print(f'Segments: {len(result['segments'])}')
print(f'\nFirst 3 segments (before alignment):')
for seg in result['segments'][:3]:
    print(f" [{seg['start']:.1f}s -- {seg['end']:.1f}s]' {seg['text']}")

print(f'\nWhisperX basic transcription works.')