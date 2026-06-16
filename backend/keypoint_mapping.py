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
    """Map COCO-17 keypoints (image coords, 2D) to the H36M-17 layout
    used by DanceMetricsEngine. z is set to 0 (2D backend)."""
    def p(i):
        return np.array([coco[i][0], coco[i][1], 0.0])

    l_hip, r_hip = p(_L_HIP), p(_R_HIP)
    pelvis = (l_hip + r_hip) / 2.0
    l_sh, r_sh = p(_L_SH), p(_R_SH)
    neck = (l_sh + r_sh) / 2.0
    spine = (pelvis + neck) / 2.0
    head = p(_NOSE)

    j = np.zeros((17, 3))
    j[0] = pelvis
    j[1] = r_hip;  j[2] = p(_R_KNEE);  j[3] = p(_R_ANK)
    j[4] = l_hip;  j[5] = p(_L_KNEE);  j[6] = p(_L_ANK)
    j[7] = spine;  j[8] = neck;        j[9] = head
    j[10] = l_sh;  j[11] = p(_L_EL);   j[12] = p(_L_WR)
    j[13] = r_sh;  j[14] = p(_R_EL);   j[15] = p(_R_WR)
    j[16] = neck
    return j
