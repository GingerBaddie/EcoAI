from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

# Load model
processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

# Load image
image = Image.open("cleandrive.webp").convert("RGB")

# Process
inputs = processor(image, return_tensors="pt")
out = model.generate(**inputs)

caption = processor.decode(out[0], skip_special_tokens=True)

print("Caption:", caption)