import cv2
import mediapipe as mp
import numpy as np
import time
import os
import urllib.request
from dance_metrics import DanceMetricsEngine

POSE_MODEL_URLS = {
    "pose_landmarker_lite.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "pose_landmarker_full.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "pose_landmarker_heavy.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
}

# Initialize MediaPipe components
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19), # Left Arm
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20), # Right Arm
    (11, 23), (12, 24), (23, 24), # Trunk
    (23, 25), (25, 27), (27, 29), (29, 31), (31, 27), # Left Leg
    (24, 26), (26, 28), (28, 30), (30, 32), (32, 28)  # Right Leg
]

class PoseEngine:
    def __init__(self, model_path='pose_landmarker_full.task'):
        self.model_path = model_path
        self._ensure_model()
        
        self.options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.model_path),
            running_mode=VisionRunningMode.VIDEO)
            
        self.landmarker = PoseLandmarker.create_from_options(self.options)
        self.metrics_engine = DanceMetricsEngine(fps=30)
        self.last_pose_landmarks = None

    def set_metrics_fps(self, fps):
        self.metrics_engine.set_fps(fps)

    def draw_cached_overlay(self, frame):
        if self.last_pose_landmarks:
            for pose_landmarks in self.last_pose_landmarks:
                self._draw_landmarks(frame, pose_landmarks)
        return frame
        
    def _ensure_model(self):
        if not os.path.exists(self.model_path):
            print(f"Downloading {self.model_path}...")
            model_name = os.path.basename(self.model_path)
            url = POSE_MODEL_URLS.get(model_name)
            if url is None:
                raise FileNotFoundError(
                    f"Model file not found and no download URL is configured: {self.model_path}"
                )
            urllib.request.urlretrieve(url, self.model_path)
            print("Download complete.")

    def _get_h36m_compatible_landmarks(self, world_landmarks_list):
        def get_xyz(idx):
            return np.array([world_landmarks_list[idx].x, 
                             world_landmarks_list[idx].y, 
                             world_landmarks_list[idx].z]) * 1000

        l_hip = get_xyz(23)
        r_hip = get_xyz(24)
        pelvis = (l_hip + r_hip) / 2
        
        l_shldr = get_xyz(11)
        r_shldr = get_xyz(12)
        neck = (l_shldr + r_shldr) / 2 
        spine = (pelvis + neck) / 2    
        head = get_xyz(0)              

        h36m_joints = np.zeros((17, 3))
        h36m_joints[0]  = pelvis
        h36m_joints[1]  = r_hip
        h36m_joints[2]  = get_xyz(26)
        h36m_joints[3]  = get_xyz(28)
        h36m_joints[4]  = l_hip
        h36m_joints[5]  = get_xyz(25)
        h36m_joints[6]  = get_xyz(27)
        h36m_joints[7]  = spine
        h36m_joints[8]  = neck
        h36m_joints[9]  = head
        h36m_joints[10] = l_shldr
        h36m_joints[11] = get_xyz(13)
        h36m_joints[12] = get_xyz(15)
        h36m_joints[13] = r_shldr
        h36m_joints[14] = get_xyz(14)
        h36m_joints[15] = get_xyz(16)
        h36m_joints[16] = neck        

        return h36m_joints

    def _draw_landmarks(self, image, landmarks):
        h, w, _ = image.shape
        points = {}
        for i, lm in enumerate(landmarks):
            if i < 11: continue
            cx, cy = int(lm.x * w), int(lm.y * h)
            points[i] = (cx, cy)
            cv2.circle(image, (cx, cy), 4, (0, 255, 255), -1)

        for connection in POSE_CONNECTIONS:
            start_idx, end_idx = connection
            if start_idx in points and end_idx in points:
                cv2.line(image, points[start_idx], points[end_idx], (0, 255, 0), 2)
        return image

    def process_frame(self, frame, timestamp_ms, draw_overlay=True):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        result = self.landmarker.detect_for_video(mp_image, int(timestamp_ms))
        
        metrics = self.metrics_engine.get_empty_metrics()
        
        if result.pose_landmarks:
            self.last_pose_landmarks = result.pose_landmarks
            if draw_overlay:
                self.draw_cached_overlay(frame)

        if result.pose_world_landmarks:
            world_landmarks = result.pose_world_landmarks[0] 
            h36m_data = self._get_h36m_compatible_landmarks(world_landmarks)
            
            # Calculate metrics
            metrics = self.metrics_engine.update(h36m_data)
            
        return frame, metrics

    def close(self):
        self.landmarker.close()
