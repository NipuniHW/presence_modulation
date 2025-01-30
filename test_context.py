from multiprocessing import Queue
from sensors.audio import Audio
from context.context_detector import ContextDetector
import pyaudio
import wave
from cv2 import imshow, waitKey, destroyAllWindows

if __name__ == "__main__":
    # Create a queue for communication
    audio_queue = Queue()
    context_queue = Queue()

    # Create and start the worker process
    worker = Audio(audio_queue, input_device_index=8)
    worker.start()

    context = ContextDetector(audio_queue, context_queue, worker.get_sample_size())
    context.start()

    try:
        while True:
            try:
                packet = context_queue.get_nowait() # packet shape (time_stamp, final_label, transcription_text)

                time_stamp  = packet[0]
                final_label = packet[1]
                text        = packet[2]

                print(f"[{time_stamp}] label : {final_label}, text : {text}")
            except:
                pass

    except KeyboardInterrupt:
        print("Stopping audio process...")
        worker.terminate()
        context.terminate()
        context.join()
        worker.join()