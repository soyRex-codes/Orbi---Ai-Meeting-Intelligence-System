# Testing for audio processor module.
from src.transcription.audio_processor import AudioProcessor

def test_audio_processor():
    processor = AudioProcessor()
    
    # Testing with sample audio file
    info = processor.get_audio_info('data/sample_audio/meeting_01.webm')
    print(f'Input file info: {info}')

    prepared_audio = processor.prepare_audio('data/sample_audio/meeting_01.webm')[1]    # We did slicing here to get only the str path of the audio not the numpy array.
    processed_info = processor.get_audio_info(prepared_audio)
    print(f'Processed file info: {processed_info}')

    
if __name__ == "__main__":
    test_audio_processor()
