# TO RUN TEST FILE USE pytest filename.py # NEVER USE python3 filename.py for running test file

import pytest # how is this helpful? pytest is a testing framework that allows you to write simple and scalable test cases for your code. It provides features like fixtures, parameterization, and assertions that make it easier to write and maintain tests. In this context, we can use pytest to create test cases for the AudioProcessor and WhisperTranscriber components, ensuring that they work correctly both individually and together. By using pytest, we can automate the testing process and catch any issues early on in the development cycle.
import os # os is a built-in Python module that provides a way to interact with the operating system. In the context of testing the AudioProcessor and WhisperTranscriber, we can use the os module to check for the existence of files, create directories, and manage file paths. For example, after processing an audio file with AudioProcessor, we can use os.path.exists() to verify that the processed audio file was created successfully. Additionally, we can use os.remove() to clean up any test files created during the testing process. Overall, the os module helps us manage file operations and ensure that our tests are working with the correct files and directories.
from pathlib import Path
from src.transcription.audio_processor import AudioProcessor
from src.transcription.whisper_transcriber import WhisperTranscriber, TranscriptSegment, TranscriptionResult


#1. Test that Auidoprocessor can take an input audio file, convert it to the correct format 16kHz mono WAV, and save it to the output directory. Check that the output file exists and has the expected properties (sample rate, duration).
# how to write tests to test both features individually and together.
# writing that in English first, then code it out.

# ------------- AudioProcessor Tests -------------
class TestAudioProcessor:
    def test_reject_unsupported_format(self):
        processor = AudioProcessor() # this is an instance/object of the AudioProcessor class, where AudioProcessor() is the constructor of the class and processor is the instance/object of the class.
        with pytest.raises(ValueError, match="not supported"): # pytest.raises is a context manager that is used to test for exceptions. It is used to test that a specific exception is raised by a piece of code. In this case, we are testing that a ValueError is raised by the prepare_audio() method when it is called with a fake file path.
            processor.prepare_audio("fake_file.xyz") # why use fake_file_xyz instead of real audio file? # because we are testing the error handling part of the code, not the actual processing part.
        # what is the use of match="not supported"? # it is used to check that the exception message contains the string "not supported".

    def test_reject_missing_file(self):
        processor = AudioProcessor()
        with pytest.raises(FileNotFoundError):   # used FileNotFoundError because we are looking for a file
            processor.prepare_audio("non_existent.mp3")
    
    @pytest.mark.skipif(
         not Path("data/sample_audio/meeting_02.mp3").exists(), # if the file does not exist, skip the test
         reason="Sample audio file not found"
    )
    def test_process__real_audio(self):
        processor = AudioProcessor()
        audio_array, output_path = processor.prepare_audio( # <- tuple, not a single value? what does it mean? # it means that the prepare_audio() method returns two values: the processed audio array and the path to the processed audio file. We can unpack these values into two separate variables, audio_array and output_path, using tuple unpacking.
        "data/sample_audio/meeting_02.mp3" # tuple is a data structure that can hold multiple values. In this case, the prepare_audio() method returns a tuple containing the processed audio array and the output path of the processed audio file. By using tuple unpacking, we can assign these two values to separate variables for easier access and readability in our test code.
        )
        assert Path(output_path).exists()
        info = processor.get_audio_info(output_path)
        assert info["sample_rate"] == 16000

# ------------- TranscriptSegment Tests -------------
class TestTranscriptSegment:
    def test_duration_calculation(self): # this function is testing the duration calculation of the TranscriptSegment class. 
        seg = TranscriptSegment(
            text="Hello world",
            start_time=1.0, # why hardcode 1.0 and 3.5? # because we are testing the duration calculation, not the actual time values.
            end_time=3.5,
            avg_logprob=-0.2,
            is_likely_speech=True,
        )
        assert seg.duration == 2.5
    
    def test_fields_accessible(self): # this function is testing the fields accessibility of the TranscriptSegment class. 
        seg = TranscriptSegment(
            text="Test",
            start_time=0.0,
            end_time=1.0,
            avg_logprob=-0.85,
            is_likely_speech=True,
        )
        assert seg.text == "Test"
        assert seg.avg_logprob == -0.85
        assert seg.is_likely_speech is True

# ------------- TranscriptionResult Tests -------------
class TestTranscriptionResult:
    def test_to_dict(self): #this function is testing the to_dict() method of the TranscriptionResult class. whish is used to convert the TranscriptionResult object to a dictionary.
        seg = TranscriptSegment("Hi", 0.0, 1.0, -0.3, True) # this is a TranscriptSegment object, which is a data structure that holds the results of a single segment of transcription.
        result = TranscriptionResult(
            segments=[seg], # we created seg before for this because TranscriptionResult expects a list of TranscriptSegment objects.
            full_text="Hi",
            language="en",
            model_size="medium",
        )
        d = result.to_dict()             
        assert d["full_text"] == "Hi"
        assert d["segments"][0]["text"] == "Hi"

# ------------- WhisperTranscriber - Integration Tests 
class TestWhisperTranscriber:
    """Integration tests — only run when model + audio are available."""

    def test_rejects_invalid_model_size(self): # this function is testing the invalid model size of the WhisperTranscriber class. 
        """This is fast — no model loading needed."""
        with pytest.raises(ValueError, match="Model size must be one of"):
            WhisperTranscriber(model_size="xxl")

    @pytest.mark.slow # custom marker to skip in fast runs
    @pytest.mark.skipif(
        not Path("data/sample_audio/meeting_02.mp3").exists(),
        reason="Sample audio not available"
    )
    def test_transcribes_real_audio(self):
        """Full pipeline: load model -> transcribe -> verify output structure."""
        transcriber = WhisperTranscriber(model_size="tiny")  # tiny! not medium, becaus it reuires less memory
        result = transcriber.transcribe("data/sample_audio/meeting_02.mp3") # why use meeting_01 because it is a small file and it is a meeting audio file. does it have to be a valid file? yes it has to be a valid file.

        # Structure checks (not content — wording varies by model)
        assert isinstance(result, TranscriptionResult)
        assert len(result.segments) > 0
        assert len(result.full_text) > 0
        assert result.language != "unknown"
        assert result.model_size == "tiny"

        # Every segment has valid timing
        for seg in result.segments:
            assert seg.start_time >= 0
            assert seg.end_time > seg.start_time
            assert isinstance(seg.is_likely_speech, bool)