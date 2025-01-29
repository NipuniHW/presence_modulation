from multiprocessing import Queue
from sensors.camera import Camera
from gaze.gaze_detector import GazeDetector
from cv2 import imshow, waitKey, destroyAllWindows

if __name__ == "__main__":
    # Create a queue for communication
    image_queue = Queue()
    gaze_queue = Queue()

    # Create and start the worker process
    worker = Camera(image_queue)
    worker.start()

    gaze = GazeDetector(image_queue, gaze_queue)
    gaze.start()
    
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
        gaze.running = False  # Stop the gaze process
        worker.join()  # Wait for the process to terminate
        gaze.join()  # Wait for the process to terminate
        destroyAllWindows()  # Close all OpenCV windows