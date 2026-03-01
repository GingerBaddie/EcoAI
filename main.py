import torch
import clip
from transformers import BlipProcessor, BlipForConditionalGeneration
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io

# -------------------------------------------------
# App Setup
# -------------------------------------------------

app = FastAPI()

# Enable CORS (important for frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = "cpu"

# -------------------------------------------------
# Load Models Once (Startup)
# -------------------------------------------------

@app.on_event("startup")
def load_models():
    global clip_model, clip_preprocess
    global blip_processor, blip_model

    print("Loading CLIP...")
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

    print("Loading BLIP...")
    blip_processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )
    blip_model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    ).to(device)

    print("Models Loaded Successfully")

# -------------------------------------------------
# Environmental Labels
# -------------------------------------------------

environment_labels = [
    "people planting trees",
    "tree plantation activity",
    "beach cleaning drive",
    "garbage cleaning activity",
    "recycling activity",
    "environmental awareness event",
    "community cleanup activity",
    "pollution control activity"
]

# -------------------------------------------------
# API Endpoint
# -------------------------------------------------

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):

    # -------- Read Image Safely --------
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # -------- BLIP Caption Generation --------
    inputs = blip_processor(image, return_tensors="pt").to(device)

    out = blip_model.generate(
        **inputs,
        max_new_tokens=35,
        num_beams=5,
        early_stopping=True
    )

    caption = blip_processor.decode(
        out[0],
        skip_special_tokens=True
    )

    # -------- CLIP Classification --------
    image_input = clip_preprocess(image).unsqueeze(0).to(device)
    text = clip.tokenize(environment_labels).to(device)

    with torch.no_grad():
        logits_per_image, _ = clip_model(image_input, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]

    best_idx = probs.argmax()
    best_label = environment_labels[best_idx]
    confidence = float(probs[best_idx])

    # Threshold for validation
    is_valid = confidence > 0.45

    # -------- Final Response --------
    return {
        "caption": caption,
        "classification": best_label,
        "confidence": confidence,
        "environment_valid": is_valid
    }

# -------------------------------------------------
# Health Check Route
# -------------------------------------------------

@app.get("/")
def root():
    return {"message": "AI Environmental Analyzer is running"}