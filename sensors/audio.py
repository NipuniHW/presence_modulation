from multiprocessing import Process
import pyaudio
import numpy as np
import time

class Audio(Process):
    def __init__(self, audio_ouput_queue, 
                 format=pyaudio.paFloat32, 
                 chunck_size=1024, 
                 channels=1, 
                 rate=16000, 
                 input=True, 
                 input_device_index=11, 
                 sample_duration=3):
        
        super().__init__()
        self.capture            = None
        self.queue              = audio_ouput_queue
        self.chunk_size         = chunck_size
        self.channels           = channels
        self.rate               = rate
        self.input              = input
        self.input_device_index = input_device_index
        self.running            = True  # Control flag to stop the process
        self.sample_duration    = sample_duration
        self.format             = format
        self.capture            = None
        self.stream             = None
        
    def run(self):
        self.capture            = pyaudio.PyAudio()
        print(f"Audio Port opened")

        self.stream  = self.capture.open(format=self.format,
                                        channels=self.channels,
                                        rate=self.rate,
                                        input=self.input,
                                        input_device_index=self.input_device_index,
                                        frames_per_buffer=self.chunk_size)
        print(f"Audio Stream opened")

        while self.running:
            start_time = time.time()
            audio_frames = []
            for i in range(0, int(self.rate / self.chunk_size * self.sample_duration)):  # 3 seconds
                data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                audio_frames.append(np.frombuffer(data, dtype=np.float32))

            self.queue.put([start_time, audio_frames])

        print(f"Audio Stream closing")
        self.stream.stop_stream()
        self.stream.close()

        print(f"Audio port terminated")
        self.capture.terminate()

    def stop(self):
        self.running = False  # Signal the process to stop
        time.sleep(0.5)  # Allow buffer to clear

        if self.stream:
            self.stream.stop_stream()
            self.stream.close()

        if self.capture:
            self.capture.terminate()


    def get_sample_size(self):
        return self.capture.get_sample_size(self.format) if self.capture else None