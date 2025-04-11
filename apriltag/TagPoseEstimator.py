import cv2
import numpy as np
from wpimath.geometry import *
import convertor

from scipy.spatial import ConvexHull

class tagPoseEstimator:
    def __init__(self, tagSize: float, tagLayout:list, cameraMatrix: np.ndarray, distortionCoeffs: np.ndarray, cameraPose: Transform3d) -> None:
        self.tagSize = tagSize

        self.cameraMatrixs= cameraMatrix
        self.distortionCoeffs = distortionCoeffs

        self.robotToCamera = cameraPose
        self.cameraToRobot = cameraPose.inverse()
        self.tagLayout =tagLayout
        
        self.fieldToTagPoses:dict={}
        for tag in tagLayout: # Licensed under the MIT License (c) 2024 Mechanical-Advantage - https://github.com/Mechanical-Advantage/RobotCode2024/blob/main/LICENSE
            tagPose = convertor.poseDictToWPIPose3d(tag["pose"])
            self.fieldToTagPoses[str(tag["ID"])]=tagPose
        #get object points
        self.objectPoints = np.array(((-self.tagSize / 2, self.tagSize / 2, 0),
                                      (self.tagSize / 2, self.tagSize / 2, 0),
                                      (self.tagSize / 2, -self.tagSize / 2, 0),
                                      (-self.tagSize / 2, -self.tagSize / 2, 0)))
    def calculate_area(self, corners):
        
        hull = ConvexHull(corners[0])
        # hull_points = points[hull.vertices]
        
        # x = hull_points[:, 0]
        # y = hull_points[:, 1]
        # area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        return hull.area
    
    def __call__(self, ids: np.ndarray, corners):
        cameraToTagPoses = []
        robotToTagPoses = []
        fieldToRobotPoses = []
        errors = []
        tagIds = ids.tolist() if ids is not None else []
        for i in range(len(corners)):
                #use the solve pnp method to calculate the pose of the tag relative to the camera
                try:
                    _, rvecs, tvecs, projectionErrors = cv2.solvePnPGeneric(self.objectPoints, np.array(corners[i]),
                                                                np.array(self.cameraMatrixs),
                                                                np.array(self.distortionCoeffs))
                except:
                    raise Exception("Failed to solvePnP")
                
                #convert tvec and rvect to wpi pose3d
                wpiPose0=convertor.openCvPoseToWpilib(tvecs[0],rvecs[0])

                #add results to lists
                cameraToTagPoses.append(convertor.robotPoseToList(wpiPose0))
                robotToTagPoses.append(convertor.robotPoseToList(wpiPose0.transformBy(self.robotToCamera)))
                fieldToRobotPoses.append(convertor.robotPoseToList(
                    self.fieldToTagPoses[str(ids[i][0])].transformBy(
                        convertor.pose3dToTransform3d(wpiPose0).inverse()).transformBy(self.robotToCamera)))
                errors.append([1/self.calculate_area(corners[i])])

        while len(cameraToTagPoses) < 10:
            cameraToTagPoses.append([-9999, -9999, -9999, -9999, -9999, -9999])
        while len(robotToTagPoses) < 10:
            robotToTagPoses.append([-9999, -9999, -9999, -9999, -9999, -9999])
        while len(fieldToRobotPoses) < 10:
            fieldToRobotPoses.append([-9999, -9999, -9999, -9999, -9999, -9999])
        while len(errors) < 10:
            errors.append([-9999])
        while len(tagIds) < 10:
            tagIds.append([-9999])
            
        return tagIds, cameraToTagPoses, robotToTagPoses, fieldToRobotPoses, errors