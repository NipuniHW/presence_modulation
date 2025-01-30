from multiprocessing import Queue
from sensors.audio import Audio
import pyaudio
import wave
from cv2 import imshow, waitKey, destroyAllWindows

if __name__ == "__main__":
    # Create a queue for communication
    audio_queue = Queue()

    # Create and start the worker process
    worker = Audio(audio_queue, input_device_index=8)
    worker.start()

    try:
        while True:
            try:
                frames = audio_queue.get_nowait()
                print(f"Received audio data of length {len(frames)}")
            except:
                pass

    except KeyboardInterrupt:
        print("Stopping audio process...")
        worker.stop()
        worker.join()