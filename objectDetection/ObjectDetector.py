import cv2
import numpy as np

class objectDetector:
    def __init__(self, model) -> None:
        self.model = model
    
    def __call__(self, frame):
        ids = []
        boxes = []
        frame = (frame * 255).astype('uint8')
        
        results = self.model(frame)
        
        boxesResults = results[0].boxes
        
        for box in boxesResults:
            boxes.append(box.xywh[0].cpu().numpy().tolist())
            ids.append(int(box.cls))
        return ids, boxes