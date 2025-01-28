from cv2 import VideoCapture, imshow, waitKey, destroyAllWindows
from multiprocessing import Process, Queue

class Camera(Process):
    def __init__(self, queue):
        super().__init__()  # Initialize the Process class
        self.capture = VideoCapture(0)
        self.output = queue
        self.running = True  # Control flag to stop the process

    def run(self):
        while self.capture.isOpened() and self.running:
            success, frame = self.capture.read()
            if not success:
                break
            self.output.put(frame)  # Send the frame to the queue
        self.capture.release()


if __name__ == "__main__":
    # Create a queue for communication
    image_queue = Queue()

    # Create and start the worker process
    worker = Camera(image_queue)
    worker.start()

    try:
        while True:
            # Retrieve the frame from the queue
            if not image_queue.empty():
                frame = image_queue.get()
                imshow('Calibrated HRI Attention Detection', frame)

            # Break the loop on 'ESC'
            if waitKey(5) & 0xFF == 27:
                break
    finally:
        # Clean up
        worker.running = False  # Stop the camera process
        worker.join()  # Wait for the process to terminate
        destroyAllWindows()  # Close all OpenCV windows