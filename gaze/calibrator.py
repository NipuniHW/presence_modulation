from .attention_detector import AttentionDetector
from .attention_calibrator import AttentionCalibrator
import cv2
from cv2 import putText

class Calibrator:
    def __init__(self, image_queue):
        self.detector   = AttentionDetector()
        self.calibrator = AttentionCalibrator()
        self.image_queue = image_queue
        self.is_running = True
        
        self.calibrator.start_calibration()
        
        
    def calibrate(self):
        while self.is_running:
            frame = self.image_queue.get()
            
            # Process frame using existing detector
            frame, attention, sustained, angles, face_found = self.detector.process_frame(frame)
            
            if face_found and angles is not None:
                pitch, yaw, _ = angles
                is_complete, message = self.calibrator.process_calibration_frame(pitch, yaw)
                
                # Display calibration status
                putText(frame, message, (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
                
                if is_complete:
                    print(f"Calibration complete!")
                    print(f"Baseline Pitch: {self.calibrator.baseline_pitch:.2f}")
                    print(f"Baseline Yaw: {self.calibrator.baseline_yaw:.2f}")
                    print(f"Pitch Threshold: {self.calibrator.pitch_threshold:.2f}")
                    print(f"Yaw Threshold: {self.calibrator.yaw_threshold:.2f}")
                    break
            
            cv2.imshow('Calibration', frame)
            if cv2.waitKey(5) & 0xFF == 27:
                break
            
        return self.calibrator