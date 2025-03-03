import torch
import threading
import numpy as np
import time
class publishThread:
    def __init__(self, publishers: dict, outputTensors: dict, waitEvent):
        
        self.publishers = publishers
        self.outputTensors = outputTensors
        self.waitEvent = waitEvent
        
        self.running = True
        
        self.thread = threading.Thread(target=self.run, args=())
        self.thread.start()
    def run(self):
        while self.running:
            if self.waitEvent:
                self.waitEvent.wait()
                self.waitEvent.clear()
            for key, publisher in self.publishers.items():
                output: np.ndarray = self.outputTensors[key].cpu().numpy()
                if output.size == 1:
                    output = output[0]
                    if key == 'latency':
                        output = time.time() - output
                publisher.set(output)
    def end(self):
        self.running = False
        self.waitEvent.set()
        self.thread.join()