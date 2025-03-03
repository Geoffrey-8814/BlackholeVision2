import cv2
import time
from Process import process, getSharedTensors
import torch
import torch.multiprocessing as mp

class cameraWorker(process):
    def __init__(self, id, resolution, inputTensors, outputTensors, waitEvent, setEvents):
        args = (id, resolution, None, None)
        super().__init__(args, inputTensors, outputTensors, waitEvent, setEvents)
        
    def setupCamera(self, id, resolution, configTensor):
        camera = cv2.VideoCapture()
        if id == "test1":
            id = 1
        else:
            id = '/dev/' + id
        camera.open(id)
        camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*'MJPG'))#without this the camera will run at 5 or 10fps for arducam
        camera.set(cv2.CAP_PROP_SETTINGS,0)#reset to defualt
        camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3) # auto mode
        camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1) # manual mode
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[0])# set frame height
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[1])# set frame width
        camera.set(cv2.CAP_PROP_EXPOSURE, configTensor[0].item())# Set exposure value
        camera.set(cv2.CAP_PROP_GAIN, configTensor[1].item())# Set sensor gain
        camera.set(cv2.CAP_PROP_FPS, configTensor[2].item())# Set FPS
        
        if not camera.isOpened():
           print(f"Could not open camera {id}")
           raise Exception(f"Could not open camera {id}") 
        else:
            print("opened")
        return camera
    
    def run(self, args, inputTensors):
        id, resolution, currentConfig, camera = args
        configTensor = inputTensors['config']
        if (camera is None) or (not torch.equal(currentConfig, configTensor)):
            if camera != None:
                camera.release()
            camera = self.setupCamera(id, resolution, configTensor)
            currentConfig = configTensor.clone()
            time.sleep(1) #wait for setup TODO(be specific)
            
        captureTime = time.time()# TODO use cv2 get tick count to get more accurate time
        ret, frame = camera.read()
        if not ret:
            raise Exception(f"Failed to capture frame with camera {self.id}")
        frameTensor = torch.from_numpy(frame) / 255
        metaDataTensor = torch.tensor([captureTime])#TODO meta data format
        # Return the result as a dictionary
        output = {'frame': frameTensor,
                  'metaData': metaDataTensor}
        return output, (id, resolution, currentConfig, camera)

if __name__ == "__main__":
    inputTensors = getSharedTensors({
        'config' : (3)
    })
    H, W = 720, 1280
    inputTensors['config'].copy_(torch.tensor([40, 0, 100]))
    
    outputTensors = getSharedTensors({
        'frame': (H, W, 3),
        'metaData': (1)
    })

    event = {"capture": mp.Event()}
    _cameraWorker = cameraWorker(1, [H, W], inputTensors, outputTensors, None, event)
    
    i = 0
    
    while True:
        i +=1 
        # if i == 500:
        #     inputTensors['config'].copy_(torch.tensor([40, 0, 100]))
        event['capture'].wait()
        event['capture'].clear()
        
        frame = outputTensors['frame'].cpu().numpy()
        cv2.imshow('Camera', frame)
        
        # Press 'q' to exit the loop
        if cv2.waitKey(1) == ord('q'):
            break
    _cameraWorker.end()