from multiprocessing import Queue
from sensors.camera import Camera
from gaze.gaze_detector import GazeDetector
from cv2 import imshow, waitKey, destroyAllWindows

if __name__ == "__main__":
    # Create a queue for communication
    image_queue = Queue()
    gaze_queue = Queue()
    gaze_image = Queue()
    
    # Create and start the worker process
    worker = Camera(image_queue)
    worker.start()

    gaze = GazeDetector(image_queue, gaze_queue, gaze_image, is_debug=True)
    gaze.start()
    
    try:
        while True:
            # Retrieve the frame from the camera queue
            if not image_queue.empty():
                frame = image_queue.get()
                imshow('Camera Input', frame)
                
            # Retrieve the frame from the gaze queue
            if not gaze_image.empty():
                frame2 = gaze_image.get()
                imshow('Gaze calibration', frame2)

            # Break the loop on 'ESC'
            if waitKey(5) == ord('q'):
                break
    finally:
        destroyAllWindows()  # Close all OpenCV windows
        
        # Clean up
        worker.running = False  # Stop the camera process
        gaze.running = False  # Stop the gaze process
        
        worker.join()  # Wait for the process to terminate
        gaze.join()  # Wait for the process to terminate
        