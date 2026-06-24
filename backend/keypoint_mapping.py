"""Keypoint -> standard H36M-17 mappings, shared by the pose sources.

Standard H36M-17 layout (matches constants.py / DanceMetricsEngine):
    0 pelvis, 1-3 right leg (hip/knee/ankle), 4-6 left leg,
    7 spine, 8 thorax, 9 neck, 10 head,
    11-13 left arm (shoulder/elbow/wrist), 14-16 right arm.

NOTE: upstream NCKU realtime-dance-analysis shipped a mapping that placed the
wrists on indices 12/15 (and shoulders/head shifted) instead of the standard
13/16, so curvature tracked the shoulders/neck and each arm's energy included
a cross-body segment. These helpers use the correct standard layout.
"""
import numpy as np

# --- MediaPipe 33 landmark indices ---
_MP_NOSE = 0
_MP_L_SH, _MP_R_SH = 11, 12
_MP_L_EL, _MP_R_EL = 13, 14
_MP_L_WR, _MP_R_WR = 15, 16
_MP_L_HIP, _MP_R_HIP = 23, 24
_MP_L_KNEE, _MP_R_KNEE = 25, 26
_MP_L_ANK, _MP_R_ANK = 27, 28

# --- COCO-17 indices (also the first 17 of COCO-WholeBody-133) ---
_C_NOSE = 0
_C_L_SH, _C_R_SH = 5, 6
_C_L_EL, _C_R_EL = 7, 8
_C_L_WR, _C_R_WR = 9, 10
_C_L_HIP, _C_R_HIP = 11, 12
_C_L_KNEE, _C_R_KNEE = 13, 14
_C_L_ANK, _C_R_ANK = 15, 16


def _assemble_h36m17(pelvis, r_hip, r_knee, r_ank, l_hip, l_knee, l_ank,
                     l_sh, l_el, l_wr, r_sh, r_el, r_wr, nose):
    """Build a standard H36M-17 array from named joints. Derived joints:
    thorax = mid-shoulders, spine = mid(pelvis, thorax), neck = mid(thorax, head)."""
    thorax = (l_sh + r_sh) / 2.0
    spine = (pelvis + thorax) / 2.0
    neck = (thorax + nose) / 2.0
    j = np.zeros((17, 3))
    j[0] = pelvis
    j[1] = r_hip;  j[2] = r_knee;  j[3] = r_ank
    j[4] = l_hip;  j[5] = l_knee;  j[6] = l_ank
    j[7] = spine;  j[8] = thorax;  j[9] = neck;  j[10] = nose
    j[11] = l_sh;  j[12] = l_el;   j[13] = l_wr
    j[14] = r_sh;  j[15] = r_el;   j[16] = r_wr
    return j


def mp33_to_h36m17(world_landmarks, scale: float = 1000.0) -> np.ndarray:
    """MediaPipe 33 world landmarks -> standard H36M-17, scaled to ~mm."""
    def g(i):
        lm = world_landmarks[i]
        return np.array([lm.x, lm.y, lm.z]) * scale
    return _assemble_h36m17(
        pelvis=(g(_MP_L_HIP) + g(_MP_R_HIP)) / 2.0,
        r_hip=g(_MP_R_HIP), r_knee=g(_MP_R_KNEE), r_ank=g(_MP_R_ANK),
        l_hip=g(_MP_L_HIP), l_knee=g(_MP_L_KNEE), l_ank=g(_MP_L_ANK),
        l_sh=g(_MP_L_SH), l_el=g(_MP_L_EL), l_wr=g(_MP_L_WR),
        r_sh=g(_MP_R_SH), r_el=g(_MP_R_EL), r_wr=g(_MP_R_WR),
        nose=g(_MP_NOSE),
    )


def coco17_to_h36m17_3d(kpts: np.ndarray) -> np.ndarray:
    """COCO-17 body keypoints WITH z (3D) -> standard H36M-17 (z preserved).
    Used by the rtmpose3d source: RTMW3D returns 133 keypoints whose first 17
    are COCO-17 body order."""
    def p(i):
        return np.asarray(kpts[i], dtype=float)[:3]
    return _assemble_h36m17(
        pelvis=(p(_C_L_HIP) + p(_C_R_HIP)) / 2.0,
        r_hip=p(_C_R_HIP), r_knee=p(_C_R_KNEE), r_ank=p(_C_R_ANK),
        l_hip=p(_C_L_HIP), l_knee=p(_C_L_KNEE), l_ank=p(_C_L_ANK),
        l_sh=p(_C_L_SH), l_el=p(_C_L_EL), l_wr=p(_C_L_WR),
        r_sh=p(_C_R_SH), r_el=p(_C_R_EL), r_wr=p(_C_R_WR),
        nose=p(_C_NOSE),
    )
