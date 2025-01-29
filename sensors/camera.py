from cv2 import VideoCapture, imshow, waitKey, destroyAllWindows
from multiprocessing import Process, Queue

class Camera(Process):
    def __init__(self, queue):
        super().__init__()  # Initialize the Process class
        self.capture = None
        self.queue = queue
        self.running = True  # Control flag to stop the process

    def run(self):
        self.capture = VideoCapture(0)

        while self.running:
            ret, frame = self.capture.read()
            if ret:
                self.queue.put(frame)

        self.capture.release()

    def stop(self):
        self.running = False  # Signal the process to stop
