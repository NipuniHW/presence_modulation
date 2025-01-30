from cv2 import VideoCapture, imshow, waitKey, destroyAllWindows
from multiprocessing import Process
import time

class Camera(Process):
    def __init__(self, queue):
        super().__init__()  # Initialize the Process class
        self.capture = None
        self.queue = queue
        self.running = True  # Control flag to stop the process

    def run(self):
        self.capture = VideoCapture(0)

        while self.running:
            current_time = time.time()
            ret, frame = self.capture.read()
            if ret:
                self.queue.put([current_time, frame])

        self.capture.release()

    def stop(self):
        self.running = False  # Signal the process to stop
