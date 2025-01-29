from cv2 import VideoCapture, imshow, waitKey, destroyAllWindows
from multiprocessing import Process, Queue

class Camera(Process):
    def __init__(self, queue):
        super().__init__()  # Initialize the Process class
        self.capture = VideoCapture(0)
        self.output = queue
        self.running = True  # Control flag to stop the process

    def run(self):
        try:
            while self.capture.isOpened() and self.running:
                success, frame = self.capture.read()
                if not success:
                    break
                self.output.put(frame)  # Send the frame to the queue    
                 
            self.capture.release()
            
        finally:
            self.capture.release()