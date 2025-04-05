from ultralytics import YOLO
import cv2
import torch
from camera2world import CoralOrientationSolver
import numpy as np

# Load a COCO-pretrained YOLO11n model


model = YOLO("runs\\detect\\train6\\weights\\last.pt")

model.to("cuda" if torch.cuda.is_available() else "cpu")

# 初始化参数
camera_matrix = np.array([[905.32671946, 0, 679.6204086],
                            [0, 906.14946047, 331.96782248],
                            [0, 0, 1]])
distortion = np.array([0.02907126, -0.03349167, 0.00055539, -0.00029301, -0.02025189])
height = 0.383
coral_radius = 0.055
coral_length = 0.3
pitch = np.deg2rad(0)

solver = CoralOrientationSolver(camera_matrix, distortion, height,
                                coral_radius, coral_length, pitch)

# initialize camera
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*'MJPG'))#without this the cap will run at 5 or 10fps for arducam
cap.set(cv2.CAP_PROP_SETTINGS, 0)#reset to default
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3) # auto mode
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1) # manual mode

cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)# set frame height
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)# set frame width
cap.set(cv2.CAP_PROP_EXPOSURE, -11)# Set exposure value
cap.set(cv2.CAP_PROP_GAIN, 0)# Set sensor gain
while True:
    ret, frame = cap.read()
    
    if not ret:
        print("read failed")
        break
    print("Frame shape:", frame.shape)
    print("Frame dtype:", frame.dtype)
    results = model(frame)
    boxes = results[0].boxes
    for box in boxes:
        print(box.xywh[0])
        if box.cls == 0:
            u_center, v_center, box_length = box.xywh[0][0], box.xywh[0][1], box.xywh[0][2]
            ccw, cw = solver.solve(u_center, v_center, box_length)

            
    # print('boxes:')
    # print(boxes[0].xywh[0])

    # Plot the results on the image
    annotated_frame = results[0].plot()

    # Display the annotated image
    cv2.imshow('YOLO Detection', annotated_frame)
        
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()