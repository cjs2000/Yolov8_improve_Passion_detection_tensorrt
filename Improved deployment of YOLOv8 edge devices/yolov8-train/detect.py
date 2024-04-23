from ultralytics import YOLO

# Load a model
model = YOLO('runs/yolov8n/weights/best.pt')  # load an official model


# Predict with the model
results = model(source='1',show=True)