from multiprocessing import Queue

from sensors.camera import Camera
from sensors.audio import Audio

from gaze.gaze_detector import GazeDetector
from learning.learning_gaze import PresenceLearner

import time
from cv2 import imshow, waitKey, destroyAllWindows


if __name__ == "__main__":
    # Create a queue for communication
    image_queue = Queue()

    gaze_queue          = Queue()
    gaze_image_queue    = Queue()

    # Create and start the worker process
    camera = Camera(image_queue)
    camera.start()

    # Wait for the camera to initialize (avoid GazeDetector running on empty input)
    print("Waiting for camera to initialize...")

    while image_queue.empty():
        time.sleep(0.5)  # Give some time for the first frame to be captured

    print("Camera initialized.")

    gaze = GazeDetector(image_queue, gaze_queue, gaze_image_queue, is_debug=True)
    gaze.start()

    while gaze_queue.empty():
        time.sleep(0.5)  # Give some time for the first frame to be captured
    
    presenceLearner = PresenceLearner(gaze_queue, "training_data")
    presenceLearner.start()

    while not presenceLearner.is_complete():
        if not gaze_image_queue.empty():
            image_packet = gaze_image_queue.get_nowait()
                
            image_time  = image_packet[0]
            image_frame = image_packet[1]

            window_name = f"Camera Image" #: {image_time:.2f}"
            imshow(window_name, image_frame)
        else:
            continue
    else:
        camera.terminate()
        gaze.terminate()
        presenceLearner.terminate()

        camera.join()
        gaze.join()
        presenceLearner.join()
