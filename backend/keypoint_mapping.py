import numpy as np

# COCO-17 indices
_NOSE = 0
_L_SH, _R_SH = 5, 6
_L_EL, _R_EL = 7, 8
_L_WR, _R_WR = 9, 10
_L_HIP, _R_HIP = 11, 12
_L_KNEE, _R_KNEE = 13, 14
_L_ANK, _R_ANK = 15, 16


def coco17_to_h36m17(coco: np.ndarray) -> np.ndarray:
    """Map COCO-17 keypoints (image coords, 2D) to the standard H36M-17 layout
    expected by DanceMetricsEngine/constants.py. z is set to 0 (2D backend).

    H36M-17: 0 pelvis, 1-3 right leg, 4-6 left leg, 7 spine, 8 thorax,
    9 neck, 10 head, 11-13 left arm (shoulder/elbow/wrist),
    14-16 right arm (shoulder/elbow/wrist)."""
    def p(i):
        return np.array([coco[i][0], coco[i][1], 0.0])

    l_hip, r_hip = p(_L_HIP), p(_R_HIP)
    pelvis = (l_hip + r_hip) / 2.0
    l_sh, r_sh = p(_L_SH), p(_R_SH)
    thorax = (l_sh + r_sh) / 2.0
    head = p(_NOSE)
    spine = (pelvis + thorax) / 2.0
    neck = (thorax + head) / 2.0

    j = np.zeros((17, 3))
    j[0] = pelvis
    j[1] = r_hip;  j[2] = p(_R_KNEE);  j[3] = p(_R_ANK)
    j[4] = l_hip;  j[5] = p(_L_KNEE);  j[6] = p(_L_ANK)
    j[7] = spine;  j[8] = thorax;      j[9] = neck;       j[10] = head
    j[11] = l_sh;  j[12] = p(_L_EL);   j[13] = p(_L_WR)
    j[14] = r_sh;  j[15] = p(_R_EL);   j[16] = p(_R_WR)
    return j
