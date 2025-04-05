import cv2
import numpy as np
from cv2.aruco import CharucoBoard, CharucoDetector, DICT_4X4_100

# configuration
squaresX = 18#11      # columns
squaresY = 11#8     # rows
squareLength = 0.030#0.0196363636363636 # in m
markerLength = 0.022#0.0144 # in m
dictionary = cv2.aruco.getPredefinedDictionary(DICT_4X4_100)
board_size = (int(2360), int(1640))  # save image size

#create charucoBoard
board = CharucoBoard(
    (squaresX, squaresY), 
    squareLength,
    markerLength,
    dictionary
)

# create and save image
board_image = board.generateImage(board_size)
cv2.imwrite("calibration/charuco_board.png", board_image)
print("saved charuco_board.png")

# initialize camera
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*'MJPG'))#without this the cap will run at 5 or 10fps for arducam
cap.set(cv2.CAP_PROP_SETTINGS, 0)#reset to default
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3) # auto mode
# cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1) # manual mode
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)# set frame height
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)# set frame width
if not cap.isOpened():
    print("open failed")
    exit()

# calibration data
all_corners = []
all_ids = []
all_object_points = [] 
all_image_points = [] 

image_size = None

print("capture started! space to capture, q to quit")
while True:
    ret, frame = cap.read()
    if not ret:
        print("read failed")
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detector = CharucoDetector(board)
    charuco_corners, charuco_ids, _, _ = detector.detectBoard(gray)
    
    # display results
    if charuco_ids is not None:
        cv2.aruco.drawDetectedCornersCharuco(frame, charuco_corners, charuco_ids)
    
    cv2.imshow('Calibration', frame)
    
    key = cv2.waitKey(1)
    if key == ord(' '):  # capture
        if charuco_ids is not None and len(charuco_ids) > 5:
            all_corners.append(charuco_corners)
            all_ids.append(charuco_ids)
            image_size = gray.shape[::-1]
            
            object_points, image_points = board.matchImagePoints(
                charuco_corners,
                charuco_ids
            )
            all_object_points.append(object_points)
            all_image_points.append(image_points)
            
            print(f"image {len(all_corners)} saved")
        else:
            print("detection failed, skipped")
    
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# run calibration
if len(all_corners) < 15:
    print("calibration requires at least 15 images")
else:
    print("calibration started...")
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        all_object_points, 
        all_image_points, 
        image_size, 
        None, 
        None
    )
    
    np.savez("calibration\calibration.npz", 
             camera_matrix=camera_matrix, 
             dist_coeffs=dist_coeffs)
    
    print("\n校准完成！结果已保存为 calibration.npz")
    print("相机矩阵:\n", camera_matrix)
    print("畸变系数:\n", dist_coeffs.ravel())
