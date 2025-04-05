import cv2
import time
from Process import process
import torch
import convertor

from objectDetection.ObjectDetector import objectDetector
from objectDetection.ObjectDetector import multiTagPoseEstimator
class objectDetectionWorker(process):
    def __init__(self, modelPath, cameraMatrix, distortionCoeffs, inputTensors, outputTensors, waitEvent, setEvents):
        args = (modelPath, cameraMatrix, distortionCoeffs, None, None, None)
        
        super().__init__(args, inputTensors, outputTensors, waitEvent, setEvents)
    
    def setup(self, modelPath, cameraMatrix, distortionCoeffs, configTensor):
        _detector = objectDetector(modelPath, 0.25)#TODO add conf to arguments
        cameraPose = convertor.pose3dToTransform3d(convertor.listToRobotPose(configTensor.cpu().numpy()))
        _coralPoseEstimator = multiTagPoseEstimator(cameraMatrix, distortionCoeffs, cameraPose)#TODO
        
        return _detector, _coralPoseEstimator
        
    def run(self, args, inputTensors):
        modelPath, cameraMatrix, distortionCoeffs, currentConfig, _detector, _coralPoseEstimator = args
        configTensor = inputTensors['config']
        if _detector is None or _coralPoseEstimator is None or (not torch.equal(currentConfig, configTensor)):
            _detector, _coralPoseEstimator = self.setup(modelPath, cameraMatrix, distortionCoeffs, configTensor)
            currentConfig = configTensor.clone()
            time.sleep(1) #wait for setup TODO(be specific)
        
        frame = inputTensors['frame'].cpu().numpy()
        
        ids, boxes = _detector(frame)
        pose, error = _coralPoseEstimator(ids, boxes)
        
        # print(poseTensor)
        output = {'coralPoses': torch.zeros(40), # [x, y, Theta 1, Theta 2] * 10 TODO
                'coralErrors': torch.zeros(10),
                'algaePoses': torch.zeros(20), # [x, y] * 10
                'latency': inputTensors['metaData']}
        
        return output, (modelPath, cameraMatrix, distortionCoeffs, currentConfig, _detector, _coralPoseEstimator)
