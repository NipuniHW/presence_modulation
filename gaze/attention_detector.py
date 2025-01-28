import math
import cv2
import mediapipe as mp
import numpy as np
from time import time
from collections import deque

class AttentionDetector:
    def __init__(self, 
                 attention_threshold=0.5,  # Time in seconds needed to confirm attention
                 pitch_threshold=15,       # Maximum pitch angle for attention
                 yaw_threshold=20,         # Maximum yaw angle for attention
                 history_size=10):         # Number of frames to keep for smoothing
        
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize parameters
        self.attention_threshold = attention_threshold
        self.pitch_threshold = pitch_threshold
        self.yaw_threshold = yaw_threshold
        self.attention_start_time = None
        self.attention_state = False
        
        # Initialize angle history for smoothing
        self.angle_history = deque(maxlen=history_size)
        
        # Define the 3D face model coordinates
        self.face_3d = np.array([
            [285, 528, 200],  # Nose tip
            [285, 371, 152],  # Chin
            [197, 574, 128],  # Left eye corner
            [173, 425, 108],  # Left mouth corner
            [360, 574, 128],  # Right eye corner
            [391, 425, 108]   # Right mouth corner
        ], dtype=np.float64)
        
        # Metrics for gaze quality and time
        self.gaze_quality = 0.0  # Percentage of frames with attention detected
        self.gaze_time = 0.0     # Total time of sustained attention in seconds
        self.frames_with_attention = 0
        self.total_frames = 0
        
    def rotation_matrix_to_angles(self, rotation_matrix):
        """Convert rotation matrix to Euler angles (pitch, yaw, roll)"""
        pitch = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        yaw = math.atan2(-rotation_matrix[2, 0], 
                        math.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2))
        roll = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        return np.array([pitch, yaw, roll]) * 180.0 / math.pi
    
    def smooth_angles(self, angles):
        """Apply smoothing to angles using a moving average"""
        self.angle_history.append(angles)
        return np.mean(self.angle_history, axis=0)
    
    def is_looking_at_robot(self, pitch, yaw):
        """Determine if the person is looking at the robot based on angles"""
        return abs(pitch) < self.pitch_threshold and abs(yaw) < self.yaw_threshold
    
    def process_frame(self, frame):
        """Process a single frame and return attention state and visualization"""
        h, w, _ = frame.shape
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        # Initialize return values
        attention_detected = False
        sustained_attention = False
        angles = None
        face_found = False
        
        if results.multi_face_landmarks:
            face_found = True
            face_2d = []
            for face_landmarks in results.multi_face_landmarks:
                # Get 2D coordinates for key landmarks
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx in [1, 9, 57, 130, 287, 359]:  # Key facial landmarks
                        x, y = int(lm.x * w), int(lm.y * h)
                        face_2d.append([x, y])
                
                face_2d = np.array(face_2d, dtype=np.float64)
                
                # Camera matrix
                focal_length = 1 * w
                cam_matrix = np.array([
                    [focal_length, 0, w / 2],
                    [0, focal_length, h / 2],
                    [0, 0, 1]
                ])
                
                # Distortion matrix
                dist_matrix = np.zeros((4, 1), dtype=np.float64)
                
                # Solve PnP
                success, rot_vec, trans_vec = cv2.solvePnP(
                    self.face_3d, face_2d, cam_matrix, dist_matrix
                )
                
                # Get rotation matrix
                rot_matrix, _ = cv2.Rodrigues(rot_vec)
                
                # Get angles
                angles = self.rotation_matrix_to_angles(rot_matrix)
                
                # Apply smoothing
                smoothed_angles = self.smooth_angles(angles)
                pitch, yaw, roll = smoothed_angles
                
                # Check if looking at robot
                attention_detected = self.is_looking_at_robot(pitch, yaw)
                
                # Track sustained attention
                current_time = time()
                if attention_detected:
                    if self.attention_start_time is None:
                        self.attention_start_time = current_time
                    elif (current_time - self.attention_start_time) >= self.attention_threshold:
                        sustained_attention = True
                else:
                    self.attention_start_time = None
                
                # Visualization
                color = (0, 255, 0) if sustained_attention else (
                    (0, 165, 255) if attention_detected else (0, 0, 255)
                )
                
                # Add text overlays
                cv2.putText(frame, f'Pitch: {int(pitch)}', (20, 20), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(frame, f'Yaw: {int(yaw)}', (20, 50), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # Draw attention status
                status = "Sustained Attention" if sustained_attention else (
                    "Attention Detected" if attention_detected else "No Attention"
                )
                cv2.putText(frame, status, (20, 80), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # Draw nose direction
                nose_2d = face_2d[0]
                nose_3d = self.face_3d[0]
                nose_3d_projection, _ = cv2.projectPoints(
                    nose_3d.reshape(1, 1, 3), rot_vec, trans_vec, cam_matrix, dist_matrix
                )
                
                p1 = (int(nose_2d[0]), int(nose_2d[1]))
                p2 = (int(nose_2d[0] + yaw), int(nose_2d[1] - pitch))
                
                cv2.line(frame, p1, p2, color, 2)
        
                        
        return frame, attention_detected, sustained_attention, angles, face_found,
