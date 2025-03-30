import cv2
import numpy as np
from ultralytics import YOLO


class objectDetector:
    def __init__(self, model: YOLO, conf) -> None:
        self.model = model
        self.conf = conf
    
    def __call__(self, frame):
        ids = []
        boxes = []
        frame = (frame * 255).astype('uint8')
        
        results = self.model.predict(frame, conf = self.conf)
        
        boxesResults = results[0].boxes
        
        for box in boxesResults:
            boxes.append(box.xywh[0].cpu().numpy().tolist())
            ids.append(int(box.cls))
        return ids, boxes