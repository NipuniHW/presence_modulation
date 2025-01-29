from multiprocessing import Queue
from sensors.camera import Camera
from cv2 import imshow, waitKey, destroyAllWindows

if __name__ == "__main__":
    # Create a queue for communication
    image_queue = Queue()

    # Create and start the worker process
    worker = Camera(image_queue)
    worker.start()

    try:
        while True:
            if not image_queue.empty():
                frame = image_queue.get_nowait()
                imshow('Camera Test', frame)
                
            # Break the loop on 'ESC'
            if waitKey(1) == ord('q'):
                break

    finally:
        destroyAllWindows()  # Close all OpenCV windows
        
        # Clean up
        worker.terminate()

        worker.join()