from ultralytics import YOLO
import torch
# Load the YOLOv8 model  workspace
model = YOLO(r'runs/yolov8n-im9/weights/best.pt')
device = torch.device("cuda")
model.cuda()
# Export the model to TensorRT format
model.export(format='engine',half=True,dynamic=True)  # creates 'yolov8n.engine'



