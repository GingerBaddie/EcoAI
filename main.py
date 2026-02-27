from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once at startup
processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

@app.post("/caption")
async def caption(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs, max_new_tokens=30)

    caption = processor.decode(out[0], skip_special_tokens=True)

    return {"caption": caption}