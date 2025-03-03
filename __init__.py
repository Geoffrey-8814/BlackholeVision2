import time
from ApriltagWorker import apriltagWorker
from CameraWorker import cameraWorker
from ConfigGenerator import configGenerator
from PublishThread import publishThread
import torch.multiprocessing as mp

if __name__ == "__main__":
    mp.set_start_method("spawn")  # This needs to be at the entry point of the script

    _configGenerator = configGenerator()

    config = _configGenerator.getConfig()
    posePublishers = _configGenerator.getPosePublishers()
    camerasConfigTensors, captureTensors, captureEvents, apriltagConfigTensors, poseTensors, poseEvents = _configGenerator.getSharedTensorsAndEvents()
    cameraWorkers: dict = {}
    apriltagWorkers: dict = {}
    posePublishersThreads: dict = {}
    _configGenerator.updateDynamicConfig(camerasConfigTensors, apriltagConfigTensors)
    for name in config['camerasName']:
        cameraConfig = config[name]
        captureTensors[name].update(apriltagConfigTensors[name])
        cameraWorkers[name] = cameraWorker(name, cameraConfig['resolution'], camerasConfigTensors[name], captureTensors[name], None, captureEvents[name])
        
        apriltagWorkers[name] = apriltagWorker(config['tagSize'], config['tagLayout'], cameraConfig['cameraMatrix'], cameraConfig['distortionCoeffs'], 
                                            captureTensors[name], poseTensors[name], captureEvents[name]['apriltag'], poseEvents[name])
        
        posePublishersThreads[name] = publishThread(posePublishers[name], poseTensors[name], poseEvents[name]['publish'])

    import cv2
    while True:
        _configGenerator.updateDynamicConfig(camerasConfigTensors, apriltagConfigTensors)
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
    for thread in posePublishersThreads.values():
        thread.end()