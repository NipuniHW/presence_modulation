import math
import cv2
import mediapipe as mp
import numpy as np
from time import time
from collections import deque
from context import classify_real_time_audio, process_speech_to_text_and_sentiment, classify_context
import threading
import queue
from tensorflow.keras.models import load_model

# Load the CNN model for ambient sound detection
model = load_model(r"/home/nipuni/Documents/Codes/q-learning/emergency_model.h5")
input_shape = model.input_shape[1:] 




def main():
    global gaze_score
    
    # First run calibration
    print("Starting calibration process...")
    calibrator = calibration_main()
    
    if not calibrator.is_calibrated:
        print("Calibration failed or was interrupted.")
        return
    
    # Initialize camera and detector with calibration
    print("\nStarting attention detection with calibrated values...")
    cap = cv2.VideoCapture(0)
    detector = CalibratedAttentionDetector(calibrator)
    
    # Create an output queue to store results from threads
    output_queue = queue.Queue()
   # output_queue2 = queue.Queue()
    
    # Initialize attention window
    attention_window = []
    
    attention = AttentionDetector()
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        # Process frame
        frame, attention, sustained, angles, face_found = detector.process_frame(frame)
        
        # Update attention window
        current_time = time()
        attention_window.append((current_time, attention))

        # Remove old entries from attention window (older than 3 seconds)
        attention_window = [(t, a) for t, a in attention_window if t > current_time - 3]
        
        # Calculate metrics
        metrics = calculate_attention_metrics(attention_window)
        
        # Start threads
        gaze_thread_obj = threading.Thread(target=calculate_gaze_score, args=(output_queue, metrics, 3.0), daemon=True)
        audio_context_thread_obj = threading.Thread(target=sync_context, args=(output_queue, ), daemon=True)
    #   speech_thread_obj = threading.Thread(target=sync_context, args=(output_queue, output_queue2), daemon=True)
        
        gaze_thread_obj.start()
        audio_context_thread_obj.start()
    #   speech_thread_obj.start()
        
        # Timing mechanism for 3-second interval
        last_time = time()
        
        # Calculate gaze score (only if 3 seconds have passed)
        if current_time - last_time >= 3:
            calculate_gaze_score(output_queue, metrics, 3.0)
            last_time = current_time
        
        # Get gaze score and final label from the output queue
        if not output_queue.empty():
            gaze_score = output_queue.get()
            final_label = output_queue.get()
        #    transcription = output_queue2.get()

            yield gaze_score, final_label#, transcription
        
        # Display the frame
        if face_found:
            h, w, _ = frame.shape
            # Add calibration values
            cv2.putText(frame, f'Baseline Pitch: {calibrator.baseline_pitch:.1f}', 
                      (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
            cv2.putText(frame, f'Baseline Yaw: {calibrator.baseline_yaw:.1f}', 
                      (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
            
            # Add metrics
            cv2.putText(frame, f'Attention Ratio: {metrics["attention_ratio"]:.2f}', 
                      (20, h - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
            cv2.putText(frame, f'Gaze Entropy: {metrics["gaze_entropy"]:.2f}', 
                      (20, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
            cv2.putText(frame, f'Frames in Window: {metrics["frames_in_interval"]}', 
                      (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
        
        cv2.imshow('Calibrated HRI Attention Detection', frame)
        
        # Break loop on 'ESC'
        if cv2.waitKey(5) & 0xFF == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"Gaze: {gaze_score}, Context: {final_label}")#, Transcription: {transcription}")
#    return gaze_score, final_label

if __name__ == "__main__":
    main()