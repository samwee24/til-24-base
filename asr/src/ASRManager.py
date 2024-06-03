# +
import io
import torch
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

class ASRManager:
    def __init__(self):
        # Initialize the Wav2Vec2 processor and model from the specified paths
        self.model = Wav2Vec2ForCTC.from_pretrained("modelnew2")
        self.processor = Wav2Vec2Processor.from_pretrained("modelnew2")

    def transcribe(self, audio_bytes: bytes) -> str:
        # Load audio from bytes using librosa
        audio_input, sample_rate = librosa.load(io.BytesIO(audio_bytes), sr=16000)
        
        # Preprocess the audio input
        input_values = self.processor(audio_input, sampling_rate=sample_rate, return_tensors="pt").input_values
        
        # Perform inference
        with torch.no_grad():
            logits = self.model(input_values).logits

        # Decode the predicted ids to get the transcription
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)[0]
    
        return transcription[0].upper() + transcription[1:] + '.'
# -


