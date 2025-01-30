from multiprocessing import Queue

from sensors.camera import Camera
from sensors.audio import Audio

from gaze.gaze_detector import GazeDetector
from context.context_detector import ContextDetector
from learning.synchronizer import PacketSynchronizer
from learning.learning import PresenceLearner

import time
from cv2 import imshow, waitKey, destroyAllWindows


if __name__ == "__main__":
    # Create a queue for communication
    image_queue = Queue()
    audio_queue = Queue()

    gaze_queue          = Queue()
    gaze_image_queue    = Queue()
    context_queue       = Queue()
    synchronized_queue  = Queue()

    # Create and start the worker process
    camera = Camera(image_queue)
    camera.start()

    # Create and start the worker process
    audio = Audio(audio_queue, input_device_index=8)
    audio.start()

    # Wait for the camera to initialize (avoid GazeDetector running on empty input)
    print("Waiting for camera to initialize...")

    while image_queue.empty():
        time.sleep(0.5)  # Give some time for the first frame to be captured

    print("Camera initialized.")

    # Wait for the audio to initialize (avoid ContextDetector running on empty input)
    print("Waiting for audio to initialize...")

    while image_queue.empty():
        time.sleep(0.5)  # Give some time for the first frame to be captured

    print("Audio initialized.")

    gaze = GazeDetector(image_queue, gaze_queue, gaze_image_queue, is_debug=True)
    gaze.start()

    context = ContextDetector(audio_queue, context_queue, audio.get_sample_size())
    context.start()

    # Start PacketSynchronizer as a separate process
    synchronizer = PacketSynchronizer(context_queue, gaze_queue, synchronized_queue, max_error=1.0, deletion_timeout=60.0)
    synchronizer.start()

    while synchronized_queue.empty():
        time.sleep(0.5)  # Give some time for the first frame to be captured
    
    presenceLearner = PresenceLearner(synchronized_queue, "training_data")
    presenceLearner.start()

    while not presenceLearner.is_complete():
        if not gaze_image_queue.empty():
            image_packet = gaze_image_queue.get_nowait()
                
            image_time  = image_packet[0]
            image_frame = image_packet[1]

            window_name = f"Camera Image at: {image_time:.2f}"
            imshow(window_name, image_frame)
        else:
            continue
    else:
        audio.terminate()
        camera.terminate()
        context.terminate()
        gaze.terminate()
        synchronizer.terminate()
        presenceLearner.terminate()

        audio.join()
        camera.join()
        context.join()
        gaze.join()
        synchronizer.join()
        presenceLearner.join()
