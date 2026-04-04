import whisper
from pathlib import Path
from scipy.io import wavfile
import torchaudio

class AudioProcessor:
    """
    Prepares raw audio/video files for transcription.
    Whisper expects 16kHz mono WAV audio, but real
    meeting recordings come in every formats. This 
    class normalizes everything into a format whisper
    can work with.
    """
    SUPPORTED_FORMATS = {".mp3", ".mp4", ".m4a", ".wav", ".webm", ".ogg", ".flac"}
    TARGET_SAMPLE_RATE = 16000   # Whisper expects this sample rate audio.

    def __init__(self, output_dir: str = 'data/processed_audio'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def prepare_audio(self, input_path: str | Path): # using Path from pathlib to handle file paths more elegantly. It can accept both string and Path objects as input.
        """
        Convert any supported audio/video file to Whisper ready format.
        """
        input_path = Path(input_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Audio file not found: {input_path}")
        
        if input_path.suffix.lower() not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Format {input_path.suffix} not supported.")
        
        audio = whisper.load_audio(str(input_path))
        print(f"Audio Loaded: {len(audio)/self.TARGET_SAMPLE_RATE:.1f} seconds")
        
        output_path = self.output_dir / f"{input_path.stem}_processed.wav"      # Saving the processed audio 
        wavfile.write(str(output_path), self.TARGET_SAMPLE_RATE, audio)         # with help of wavfile from scipy library

        return audio, str(output_path)
    
    def get_audio_info(self, input_path: str) -> dict:
        """
        Get basic audio info using torchaudio's loader.
        """
        try:
            audio_data, sample_rate = torchaudio.load(str(input_path))       # This line is for finding the sample rate of the audio. 
            duration = audio_data.shape[1] / sample_rate
            
            return {
                'duration_seconds': duration,
                'duration_formatted': self._format_duration(duration),
                'sample_rate': sample_rate
            }

        except Exception:
            return {'error':'Could not read audio info'}
        

    def _format_duration(self, seconds: float) -> str:
        minutes = int(seconds // 60)
        remaining_seconds = int(seconds % 60)

        if minutes >= 60:
            hours = minutes // 60
            minutes = minutes % 60
            return f"{hours}:{minutes:02d}:{remaining_seconds:02d}"
        
        return f"{minutes}:{remaining_seconds:02d}"
