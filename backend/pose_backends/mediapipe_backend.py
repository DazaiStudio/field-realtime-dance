import cv2
import numpy as np
import mediapipe as mp

from pose_backend import PersonPose
from centroid_tracker import CentroidTracker

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


def _mp33_to_h36m17(lms) -> np.ndarray:
    def g(i):
        return np.array([lms[i].x, lms[i].y, lms[i].z]) * 1000.0
    l_hip, r_hip = g(23), g(24)
    pelvis = (l_hip + r_hip) / 2
    l_sh, r_sh = g(11), g(12)
    neck = (l_sh + r_sh) / 2
    spine = (pelvis + neck) / 2
    head = g(0)
    j = np.zeros((17, 3))
    j[0] = pelvis
    j[1] = r_hip;  j[2] = g(26); j[3] = g(28)
    j[4] = l_hip;  j[5] = g(25); j[6] = g(27)
    j[7] = spine;  j[8] = neck;  j[9] = head
    j[10] = l_sh;  j[11] = g(13); j[12] = g(15)
    j[13] = r_sh;  j[14] = g(14); j[15] = g(16)
    j[16] = neck
    return j


class MediaPipeBackend:
    def __init__(self, model_path="pose_landmarker_full.task", num_poses=4):
        opts = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.VIDEO,
            num_poses=num_poses)
        self.landmarker = PoseLandmarker.create_from_options(opts)
        self.tracker = CentroidTracker()

    def estimate(self, frame, timestamp_ms: float):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect_for_video(mp_image, int(timestamp_ms))
        people = []
        if not result.pose_world_landmarks:
            return people
        h, w, _ = frame.shape
        centroids = []
        for img_lms in result.pose_landmarks:
            cx = (img_lms[23].x + img_lms[24].x) / 2 * w
            cy = (img_lms[23].y + img_lms[24].y) / 2 * h
            centroids.append((cx, cy))
        ids = self.tracker.update(centroids)
        for i, world_lms in enumerate(result.pose_world_landmarks):
            img_lms = result.pose_landmarks[i]
            kpts = np.array([[lm.x * w, lm.y * h] for lm in img_lms])
            xs, ys = kpts[:, 0], kpts[:, 1]
            bbox = (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))
            people.append(PersonPose(
                track_id=ids[i], h36m17=_mp33_to_h36m17(world_lms),
                bbox=bbox, kpts_2d=kpts, is_3d=True))
        return people

    def close(self):
        self.landmarker.close()
