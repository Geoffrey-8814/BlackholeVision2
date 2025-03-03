import cv2
import numpy as np

class arucoDetector:
    def __init__(self, dictionary_id) -> None:
        self._aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary_id)
        self._aruco_params = cv2.aruco.DetectorParameters()
        self.arucoDetector: cv2.aruco.ArucoDetector = cv2.aruco.ArucoDetector(self._aruco_dict, self._aruco_params)
    
    def __call__(self, frame):
        # print("Image shape:", frame.shape)  # (height, width, channels)
        # print("Image type:", frame.dtype)   # Data type (e.g., uint8)
        #detect tag corners
        frame = (frame * 255).astype('uint8')
        if len(frame.shape) == 3:  # If the image is color (3 channels)
            gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        else:
            gray_image = frame  # Already grayscale
        corners, ids, rejectedImgPoints = self.arucoDetector.detectMarkers(gray_image)
        return ids, corners