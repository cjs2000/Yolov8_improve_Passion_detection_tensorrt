from ultralytics import YOLO
import multiprocessing
import time
import os
if __name__ == '__main__':

    # Load a model

    modeld = YOLO(r"cfg/yolov8-PA-p2-SlimNeck.yaml")
    modeld.train(data=r"C:\data\Divide-10\i\data.yaml",
                 epochs=300,
                 cache=False,
                 batch=8,
                 patience=0,
                 imgsz=640,
                 workers=13,
                 device=0,
                 lr0=0.01,
                 project='Divide-10-im',
                 name='i',
                 pretrained=False,
                 )