from multiprocessing import Queue
from sensors.camera import Camera
from gaze.gaze_detector import GazeDetector
from cv2 import imshow, waitKey, destroyAllWindows
import time

if __name__ == "__main__":
    # Create a queue for communication
    image_queue = Queue()
    gaze_queue = Queue()
    gaze_image_queue = Queue()
    
    # Create and start the worker process
    worker = Camera(image_queue)
    worker.start()

    # Wait for the camera to initialize (avoid GazeDetector running on empty input)
    print("Waiting for camera to initialize...")

    while image_queue.empty():
        time.sleep(0.5)  # Give some time for the first frame to be captured

    gaze = GazeDetector(image_queue, gaze_queue, gaze_image_queue, is_debug=True)
    gaze.start()
    
    try:
        while True:
            if not gaze_image_queue.empty():
                image_packet = gaze_image_queue.get_nowait()
                
                image_time  = image_packet[0]
                image_frame = image_packet[1]

                window_name = f"Camera Image at: {image_time:.2f}"
                imshow(window_name, image_frame)
            else:
                continue  # Skip if no frame is available

            if not gaze_queue.empty():
                gaze_packet = gaze_queue.get_nowait()
                
                gaze_time  = gaze_packet[0]
                gaze_score = gaze_packet[1]

                print(f"[{gaze_time}] score : {gaze_score}")
            else:
                continue  # Skip if no frame is available

            # Break the loop on 'ESC'
            if waitKey(5) == ord('q'):
                break
            
    finally:
        print("\nShutting down processes...")
        destroyAllWindows()  # Close all OpenCV windows
        
        # Stop Camera and GazeDetector processes safely
        worker.terminate()
        gaze.terminate()
        
        worker.join()  # Wait for the process to terminate
        gaze.join()  # Wait for the process to terminate
        