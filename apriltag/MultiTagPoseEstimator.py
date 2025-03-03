import cv2
import numpy as np
from wpimath.geometry import *
import convertor

from scipy.spatial import ConvexHull

class multiTagPoseEstimator:
    def __init__(self, tagSize: float, tagLayout:list, cameraMatrix: np.ndarray, distortionCoeffs: np.ndarray, cameraPose: Transform3d) -> None:
        self.tagSize = tagSize

        self.cameraMatrixs= cameraMatrix
        self.distortionCoeffs = distortionCoeffs
        self.cameraToRobot = cameraPose.inverse()
        self.cornerPoses:dict={}
        for tag in tagLayout:
            tagPose = convertor.poseDictToWPIPose3d(tag["pose"])
            corner0 = tagPose + Transform3d(Translation3d(0.0, self.tagSize / 2.0, -self.tagSize / 2.0), Rotation3d())
            corner1 = tagPose + Transform3d(Translation3d(0.0, -self.tagSize / 2.0, -self.tagSize / 2.0), Rotation3d())
            corner2 = tagPose + Transform3d(Translation3d(0.0, -self.tagSize / 2.0, self.tagSize / 2.0), Rotation3d())
            corner3 = tagPose + Transform3d(Translation3d(0.0, self.tagSize / 2.0, self.tagSize / 2.0), Rotation3d())
            tagObjectPoints=[
                convertor.wpilibTranslationtoOpenCv(corner0.translation()),
                convertor.wpilibTranslationtoOpenCv(corner1.translation()),
                convertor.wpilibTranslationtoOpenCv(corner2.translation()),
                convertor.wpilibTranslationtoOpenCv(corner3.translation())
            ]
            self.cornerPoses[str(tag["ID"])]=tagObjectPoints
    def calculate_max_area(self, corners):
        
        points = np.vstack(corners)
        hull = ConvexHull(points[0])
        # hull_points = points[hull.vertices]
        
        # x = hull_points[:, 0]
        # y = hull_points[:, 1]
        # area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        return hull.area
    
    def __call__(self, ids, corners):
        if len(corners) > 0:
            objectPoints:list=[]
            observedPoints:list=[]
            #create the object points (observed corners' pose)
            for i in range(len(corners)):
                observedPoints.extend(corners[i][0])
                objectPoints.extend(self.cornerPoses[str(ids[i][0])])
            # print("object points:",objectPoints)
            # print("observed points:",observedPoints)

            #get opencv pose(field to camera)
            # print(np.array(objectPoints).shape)
            # print(np.array(observedPoints).shape)
            # print(np.array(self.cameraMatrixs).shape)
            # print(np.array(self.distortionCoeffs).shape)
            
            _, rvecs, tvecs, errors = cv2.solvePnPGeneric(np.array(objectPoints), np.array(observedPoints),
                                                                np.array(self.cameraMatrixs),
                                                                np.array(self.distortionCoeffs),
                                                                flags=cv2.SOLVEPNP_SQPNP)
            
            #convert it to wpi pose      
            cameraToFieldPose = convertor.openCvPoseToWpilib(tvecs[0],rvecs[0])
            #transform it to field to robot pose
            cameraToField = Transform3d(cameraToFieldPose.translation(),cameraToFieldPose.rotation())
            fieldToCamera = cameraToField.inverse()
            fieldToCameraPose=Pose3d(fieldToCamera.translation(),fieldToCamera.rotation())
            return fieldToCameraPose.transformBy(self.cameraToRobot), 1/self.calculate_max_area(corners)
        
        return None, None