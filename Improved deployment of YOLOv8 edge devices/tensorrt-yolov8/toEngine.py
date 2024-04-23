from ultralytics import YOLO
import torch
# Load the YOLOv8 model  workspace
model = YOLO(r'runs/best.pt')
device = torch.device("cuda")
model.cuda()
# Export the model to TensorRT format
model.export(format='engine')  # creates 'yolov8n.engine'



