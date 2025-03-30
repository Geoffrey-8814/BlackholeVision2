import cv2
import numpy as np
from ultralytics import YOLO


class objectDetector:
    def __init__(self, modelPath, conf) -> None:
        self.model: YOLO = YOLO(modelPath)
        self.model.to('cuda') #TODO
        self.conf = conf
    
    def __call__(self, frame):
        ids = []
        boxes = []
        frame = (frame * 255).astype('uint8')
        results = self.model.predict(frame, conf = self.conf)
        
        boxesResults = results[0].boxes
        
        # annotated_frame = results[0].plot()

        # # Display the annotated image
        # cv2.imshow('YOLO Detection', annotated_frame)
            
        # key = cv2.waitKey(1)
        
        for box in boxesResults:
            boxes.append(box.xywh[0].cpu().numpy().tolist())
            ids.append(int(box.cls))
        
        return ids, boxes