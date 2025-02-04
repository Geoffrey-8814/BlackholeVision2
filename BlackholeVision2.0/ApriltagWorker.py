import cv2
import time
from Process import process
import torch
import convertor

from apriltag.Detector import arucoDetector
from apriltag.MultiTagPoseEstimator import multiTagPoseEstimator

class apriltagWorker(process):
    def __init__(self, tagSize, tagLayout, cameraMatrix, distortionCoeffs, inputTensors, outputTensors, waitEvent, setEvents):
        # args = (tagSize, tagLayout, cameraMatrix, distortionCoeffs, None, None, None, outputPublishers)
        args = (tagSize, tagLayout, cameraMatrix, distortionCoeffs, None, None, None)
        
        super().__init__(args, inputTensors, outputTensors, waitEvent, setEvents)
    
    def setup(self, tagSize, tagLayout, cameraMatrix, distortionCoeffs, configTensor):
        _detector = arucoDetector(cv2.aruco.DICT_APRILTAG_36H11)
        cameraPose = convertor.pose3dToTransform3d(convertor.listToRobotPose(configTensor.cpu().numpy()))#TODO tensor to list?
        _multiTagPoseEstimator = multiTagPoseEstimator(tagSize, tagLayout, cameraMatrix, distortionCoeffs, cameraPose)
        
        return _detector, _multiTagPoseEstimator
        
    def run(self, args, inputTensors):
        tagSize, tagLayout, cameraMatrix, distortionCoeffs, currentConfig, _detector, _multiTagPoseEstimator= args
        configTensor = inputTensors['config']
        if _detector is None or _multiTagPoseEstimator is None or (not torch.equal(currentConfig, configTensor)):
            _detector, _multiTagPoseEstimator = self.setup(tagSize, tagLayout, cameraMatrix, distortionCoeffs, configTensor)
            currentConfig = configTensor.clone()
            time.sleep(1) #wait for setup TODO(be specific)
        
        frame = inputTensors['frame'].cpu().numpy()
        
        ids, corners = _detector(frame)
        pose, error = _multiTagPoseEstimator(ids, corners)
        
        poseTensor = convertor.robotPoseToTensor(pose)
        errorTensor = torch.tensor([error if error else -1])
        
        # print(poseTensor)
        output = {'multiTagPose': poseTensor,
                  'multiTagError': errorTensor,
                  'latency': inputTensors['metaData']}
        
        return output, (tagSize, tagLayout, cameraMatrix, distortionCoeffs, currentConfig, _detector, _multiTagPoseEstimator)
