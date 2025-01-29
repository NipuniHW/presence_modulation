from .attention_detector import AttentionDetector
from .attention_calibrator import AttentionCalibrator
from .calibrated_attention_detector import CalibratedAttentionDetector
import cv2
import time
import math
from multiprocessing import Process
from cv2 import putText

class GazeDetector(Process):
    def __init__(self, input_image_queue, output_gaze_queue):
        super().__init__()  # Initialize the Process class
        self.detector           = AttentionDetector()
        self.calibrator         = AttentionCalibrator()
        self.calib_detector     = None
        self.input_image_queue  = input_image_queue
        self.output_gaze_queue  = output_gaze_queue
        self.running            = True
        
        self.calibrator.start_calibration()
        
    def calibrate(self):
        print("Starting calibration process...")
        
        while self.running:
            frame = self.input_image_queue.get()
            
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
    
    def calculate_attention_metrics(self, attention_window, interval_duration=3.0):
   
        if not attention_window:
            return {
                'gaze_time': 0.0,
                'attention_ratio': 0.0,
                'gaze_entropy': 0.0,
                'frames_in_interval': 0,
                'robot_looks': 0,
                'non_robot_looks': 0
            }
            

        # Get current time and filter window to only include last interval_duration seconds
        current_time = attention_window[-1][0]  # Latest timestamp
        filtered_window = [(t, a) for t, a in attention_window 
                        if current_time - t <= interval_duration]
        
        # Count frames
        frames_in_interval = len(filtered_window)
        robot_looks = sum(1 for _, attention in filtered_window if attention)
        non_robot_looks = frames_in_interval - robot_looks
        
        # Calculate attention ratio
        attention_ratio = robot_looks / frames_in_interval if frames_in_interval > 0 else 0.0
        
        # Calculate stationary gaze entropy
        gaze_entropy = 0.0
        if frames_in_interval > 0:
            p_robot = robot_looks / frames_in_interval
            p_non_robot = non_robot_looks / frames_in_interval
            
            # Calculate entropy using Shannon formula
            if p_robot > 0:
                gaze_entropy -= p_robot * math.log2(p_robot)
            if p_non_robot > 0:
                gaze_entropy -= p_non_robot * math.log2(p_non_robot)
        
        # Calculate continuous gaze time for robot
        continuous_gaze_time = 0.0
        start_time = None
        for i, (timestamp, attention) in enumerate(filtered_window):
            if attention:
                if start_time is None:
                    start_time = timestamp
                elif i == len(filtered_window) - 1 or not filtered_window[i + 1][1]:
                    # If attention ends or this is the last frame, calculate duration
                    duration = timestamp - start_time
                    if duration >= interval_duration:
                        continuous_gaze_time += duration
                    start_time = None
            else:
                start_time = None  # Reset if attention breaks

        #print(f"Debug: continous gaze_time is {continuous_gaze_time}")  
        
        return {
            'gaze_time': continuous_gaze_time,
            'attention_ratio': attention_ratio,
            'gaze_entropy': gaze_entropy,
            'frames_in_interval': frames_in_interval,
            'robot_looks': robot_looks,
            'non_robot_looks': non_robot_looks
        }
    
    def calculate_gaze_score(self, metrics, interval_duration=3.0):
        # Extract values from the metrics dictionary
        continuous_gaze_time = metrics['gaze_time']  # Continuous time human is looking at the robot
        attention_ratio = metrics['attention_ratio']  # Proportion of frames with attention on the robot
        gaze_entropy = metrics['gaze_entropy']  # Entropy of gaze distribution
        
        # Normalize the continuous gaze time
        normalized_gaze_time = min(continuous_gaze_time / interval_duration, 1.0)
        
        # Normalize the gaze entropy (lower entropy is better for focused attention)
        normalized_entropy = 1.0 - min(gaze_entropy, 1.0)  # Invert to reward focus
        
        # Define weights for each metric
        weight_gaze_time = 0.5  # Highest weight for continuous gaze
        weight_attention_ratio = 0.3  # Moderate weight for overall attention ratio
        weight_entropy = 0.2  # Lowest weight for entropy
        
        normalized_gaze_time = normalized_gaze_time ** 1.5  # Boost sustained gaze
        attention_ratio = attention_ratio ** 1.2  # Slightly emphasize high attention ratios
        
        # Calculate raw gaze score (weighted sum of metrics)
        raw_score = (
            weight_gaze_time * normalized_gaze_time +
            weight_attention_ratio * attention_ratio +
            weight_entropy * normalized_entropy
        )
        
        # Scale raw score to 0–100
        gaze_score = raw_score * 100
        
        return min(max(round(gaze_score, 1),0),100)

    def run(self):
        self.calibrate()
        
        if not self.calibrator.is_calibrated:
            print("Calibration failed or was interrupted.")
            return
        
        self.calib_detector = CalibratedAttentionDetector(self.calibrator)
        
        attention_window = []
        
        while self.running:
            frame = self.input_image_queue.get()
            
            # Process frame
            frame, attention, sustained, angles, face_found = self.calib_detector.process_frame(frame)
            
            # Update attention window
            current_time = time.time()
            attention_window.append((current_time, attention))
            
            # Remove old entries from attention window (older than 3 seconds)
            attention_window = [(t, a) for t, a in attention_window if t > current_time - 3]
            
            # Calculate metrics
            metrics = self.calculate_attention_metrics(attention_window)
        
            # Update gaze score calculations
            gaze_score =  self.calculate_gaze_score(metrics, interval_duration=3.0)
            
            # Display the frame
            if face_found:
                h, w, _ = frame.shape
                # Add calibration values
                cv2.putText(frame, f'Baseline Pitch: {self.calibrator.baseline_pitch:.1f}', 
                        (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
                cv2.putText(frame, f'Baseline Yaw: {self.calibrator.baseline_yaw:.1f}', 
                        (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
                
                # Add metrics
                cv2.putText(frame, f'Attention Ratio: {metrics["attention_ratio"]:.2f}', 
                        (20, h - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
                cv2.putText(frame, f'Gaze Entropy: {metrics["gaze_entropy"]:.2f}', 
                        (20, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
                cv2.putText(frame, f'Frames in Window: {metrics["frames_in_interval"]}', 
                        (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
            
            
            self.output_gaze_queue.put((current_time, gaze_score))
            
            cv2.imshow('Calibrated HRI Attention Detection', frame)
            
            # Break loop on 'ESC'
            if cv2.waitKey(5) & 0xFF == 27:
                break
            
            