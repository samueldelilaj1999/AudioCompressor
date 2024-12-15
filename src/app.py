import torch
import soundfile as sf
from transformers import EncodecModel
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import librosa

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AudioData(BaseModel):
    audio_codes: list[str]
    dims: list[int]
    file_name: str

@app.post("/process-audio/")
async def process_audio(data: AudioData):
    try:
        audio_codes = [int(code) for code in data.audio_codes]
        model = EncodecModel.from_pretrained("facebook/encodec_24khz")

        audio_codes_np = torch.tensor(audio_codes, dtype=torch.long).reshape(data.dims)

        with torch.no_grad():
            audio_values = model.decode(audio_codes_np, [None])[0]

        original_sample_rate = 24000
        target_sample_rate = 22050
        resampled_audio = librosa.resample(audio_values.squeeze().detach().numpy(), orig_sr=original_sample_rate, target_sr=target_sample_rate)

        output_file = f"{data.file_name}_decoded_22050Hz.wav"
        sf.write(output_file, resampled_audio, target_sample_rate)

        return {"message": "Audio processed successfully", "file_name": output_file}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))