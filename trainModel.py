from ultralytics import YOLO
import cv2
if __name__ == '__main__':
    # Load a COCO-pretrained YOLO11n model
    model = YOLO("runs\\detect\\train5\\weights\\last.pt")

    model.to('cuda')

    results = model.train(data="reefscape-collation-part-i.v6i.yolov11\data.yaml", epochs=50, imgsz=640)