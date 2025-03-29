import cv2
import time
from Process import process
import torch
import convertor

from objectDetection.ObjectDetector import objectDetector

class objectDetectionWorker(process):
    def __init__(self, model, cameraMatrix, distortionCoeffs, inputTensors, outputTensors, waitEvent, setEvents):
        args = (model, cameraMatrix, distortionCoeffs, None, None, None)
        
        super().__init__(args, inputTensors, outputTensors, waitEvent, setEvents)
    
    def setup(self, model, cameraMatrix, distortionCoeffs, configTensor):
        _detector = objectDetector(model)
        cameraPose = convertor.pose3dToTransform3d(convertor.listToRobotPose(configTensor.cpu().numpy()))
        _coralPoseEstimator = multiTagPoseEstimator(cameraMatrix, distortionCoeffs, cameraPose)#TODO
        
        return _detector, _coralPoseEstimator
        
    def run(self, args, inputTensors):
        model, cameraMatrix, distortionCoeffs, currentConfig, _detector, _coralPoseEstimator = args
        configTensor = inputTensors['config']
        if _detector is None or _coralPoseEstimator is None or (not torch.equal(currentConfig, configTensor)):
            _detector, _coralPoseEstimator = self.setup(model, cameraMatrix, distortionCoeffs, configTensor)
            currentConfig = configTensor.clone()
            time.sleep(1) #wait for setup TODO(be specific)
        
        frame = inputTensors['frame'].cpu().numpy()
        
        ids, boxes = _detector(frame)
        pose, error = _coralPoseEstimator(ids, boxes)
        
        poseTensor = convertor.robotPoseToTensor(pose)
        errorTensor = torch.tensor([error if error else -1])
        
        # print(poseTensor)
        output = {'coralPose': poseTensor,
                'algaePose': errorTensor,
                'latency': inputTensors['metaData']}
        
        return output, (model, cameraMatrix, distortionCoeffs, currentConfig, _detector, _coralPoseEstimator)
