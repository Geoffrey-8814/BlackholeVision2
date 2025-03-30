import time
from ultralytics import YOLO
from ApriltagWorker import apriltagWorker
from ObjectDetectionWorker import objectDetectionWorker
from CameraWorker import cameraWorker
from ConfigGenerator import configGenerator
from PublishThread import publishThread
import torch.multiprocessing as mp

if __name__ == "__main__":
    mp.set_start_method("spawn")  # This needs to be at the entry point of the script

    _configGenerator = configGenerator()

    config = _configGenerator.getConfig()
    posePublishers = _configGenerator.getPosePublishers()
    objPosePublishers = _configGenerator.getObjPosePublishers()
    camerasConfigTensors, captureTensors, captureEvents, apriltagConfigTensors, poseTensors, poseEvents, objPoseTensors, objPoseEvents = _configGenerator.getSharedTensorsAndEvents()
    cameraWorkers: dict = {}
    apriltagWorkers: dict = {}
    objDetectionWorkers: dict = {}
    posePublishersThreads: dict = {}
    objPosePublishersThreads: dict = {}
    
    detectionModel = "objectDetection\\weights\\last.pt"
    
    _configGenerator.updateDynamicConfig(camerasConfigTensors, apriltagConfigTensors)
    for name in config['camerasName']:
        cameraConfig = config[name]
        captureTensors[name].update(apriltagConfigTensors[name])
        cameraWorkers[name] = cameraWorker(name, cameraConfig['resolution'], camerasConfigTensors[name], captureTensors[name], None, captureEvents[name])
        
        if config[name]["enableTag"]:
            apriltagWorkers[name] = apriltagWorker(config['tagSize'], config['tagLayout'], cameraConfig['cameraMatrix'], cameraConfig['distortionCoeffs'], 
                                                captureTensors[name], poseTensors[name], captureEvents[name]['apriltag'], poseEvents[name])
            posePublishersThreads[name] = publishThread(posePublishers[name], poseTensors[name], poseEvents[name]['publish'])
        
        if config[name]["enableObj"]:
            objDetectionWorkers[name] = objectDetectionWorker(detectionModel, cameraConfig['cameraMatrix'], cameraConfig['distortionCoeffs'], 
                                                captureTensors[name], objPoseTensors[name], captureEvents[name]['ML'], objPoseEvents[name])
            objPosePublishersThreads[name] = publishThread(objPosePublishers[name], objPoseTensors[name], objPoseEvents[name]['publish'])


    import cv2
    while True:
        # _configGenerator.updateDynamicConfig(camerasConfigTensors, apriltagConfigTensors) #TODO fix reopening cameras
        # print('running')
        # time.sleep(0.1)
        for name in config['camerasName']:
            frame = captureTensors[name]['frame'].cpu().numpy()
            cv2.imshow(name, frame)
        time.sleep(0.03)
        # Press 'q' to exit the loop
        if cv2.waitKey(1) == ord('q'):
            break
    for worker in cameraWorkers.values():
        worker.end()
    for worker in apriltagWorkers.values():
        worker.end()
    for worker in objDetectionWorkers.values():
        worker.end()
    for thread in posePublishersThreads.values():
        thread.end()
    for thread in objPosePublishersThreads.values():
        thread.end()