import torch
import torch.multiprocessing as mp

class process:
    def __init__(self, args: tuple, inputTensors: dict, outputTensors: dict, waitEvent=None, setEvents: dict=None):
        self.stop_event = mp.Event()
        self.waitEvent = waitEvent
        self.p = mp.Process(target=self.thread, args=(args, inputTensors, outputTensors, self.waitEvent, setEvents, self.stop_event))
        self.p.start()
        
    def thread(self, args, inputTensors, outputTensors, waitEvent, setEvents, stop_event):
        while True:
            if waitEvent:  # Ensure waitEvent is not None
                waitEvent.wait()
                waitEvent.clear()
            if stop_event.is_set():
                break
            
            _outputTensors, args = self.run(args, inputTensors)

            for key, outputTensor in _outputTensors.items():  # Fix incorrect dictionary iteration
                outputTensors[key].copy_(outputTensor)

            if setEvents:  # Ensure setEvents is not None
                for setEvent in setEvents.values():
                    setEvent.set()

    def run(self, args, inputTensors):
        """Override this method in subclasses to implement custom processing."""
        return {}, args
    
    def close(self, args, inputTensors):
        """Override this method in subclasses to implement custom processing."""
        
    def end(self):
        """Graceful stop using an event."""
        if self.waitEvent:
            self.waitEvent.set()
        self.stop_event.set()
        self.p.join()  # Ensure the process is properly stopped

    def forceEnd(self):
        """Force stop the process."""
        self.p.terminate()
        self.p.join()  # Ensure cleanup

def getSharedTensors(shapes: dict):
    """Creates a list of shared memory tensors."""
    sharedTensors = {}
    for key, shape in shapes.items():
        tensor = torch.zeros(shape)
        tensor.share_memory_()  # Move to shared memory
        sharedTensors[key]=tensor
    return sharedTensors

class example(process):
    def __init__(self, args, inputTensors, outputTensors, waitEvent, setEvents):
        super().__init__(args, inputTensors, outputTensors, waitEvent, setEvents)

    def run(self, args, inputTensors):
        """Override the run method to multiply tensors."""
        # Assume inputTensors has two tensors: 'input1' and 'input2'
        print('run')
        num = args
        num = num+1
        print(num)
        input1 = inputTensors['input1']
        input2 = inputTensors['input2']

        # Perform multiplication (you can modify this to match your specific logic)
        result = input1 * input2

        # Return the result as a dictionary
        return {'output': result}, (num)

if __name__ == "__main__":
    mp.set_start_method("spawn")  # This needs to be at the entry point of the script

    # Create shared memory tensors
    inputTensors = getSharedTensors({
        'input1': (3, 3),
        'input2': (3, 3)
    })
    outputTensors = getSharedTensors({
        'output': (3, 3)
    })

    # Initialize waitEvent and setEvents
    waitEvent = mp.Event()
    setEvents = {"done": mp.Event()}

    # Fill input tensors with some data
    inputTensors['input1'].fill_(2)  # Example: tensor filled with 2s
    inputTensors['input2'].fill_(3)  # Example: tensor filled with 3s

    # Create and start the child class process
    multiply_process = example((0), inputTensors, outputTensors, waitEvent, setEvents)

    # Start the computation by setting the waitEvent
    import time
    for i in range(10):
        setEvents['done'].wait()
        print("Resulting output tensor:\n", outputTensors["output"])
        time.sleep(1)

    multiply_process.end()