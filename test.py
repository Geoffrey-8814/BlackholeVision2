from ultralytics import YOLO
import cv2

# Load a COCO-pretrained YOLO11n model


model = YOLO("runs\\detect\\train5\\weights\\last.pt")

model.to('cuda')



# initialize camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*'MJPG'))#without this the cap will run at 5 or 10fps for arducam
cap.set(cv2.CAP_PROP_SETTINGS, 0)#reset to default
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3) # auto mode
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1) # manual mode
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)# set frame height
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)# set frame width

while True:
    ret, frame = cap.read()
    if not ret:
        print("read failed")
        break
    
    results = model(frame)
    # Plot the results on the image
    annotated_frame = results[0].plot()

    # Display the annotated image
    cv2.imshow('YOLO Detection', annotated_frame)
        
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()