from ultralytics import YOLO

# Load pretrained model (small & fast)
model = YOLO("yolov8n.pt")

# Run detection on image
results = model("cleandrive.webp", show=True)

# Print detected objects
for r in results:
    for box in r.boxes:
        label = model.names[int(box.cls)]
        confidence = float(box.conf)
        print(f"{label} - {confidence:.2f}")