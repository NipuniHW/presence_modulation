from multiprocessing import Queue
from sensors.audio import Audio
import pyaudio
import wave
from cv2 import imshow, waitKey, destroyAllWindows

if __name__ == "__main__":
    # Create a queue for communication
    audio_queue = Queue()

    # Create and start the worker process
#    worker = Audio(audio_queue, input_device_index=8)
    worker = Audio(audio_queue, input_device_index=11)
    worker.start()

    try:
        while True:
            try:
                packet = audio_queue.get_nowait()

                time_stamp = packet[0]
                frames     = packet[1]

                print(f"[{time_stamp}] audio data of length : {len(frames)}")
            except:
                pass

    except KeyboardInterrupt:
        print("Stopping audio process...")
        worker.stop()
        worker.join()