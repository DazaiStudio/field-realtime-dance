import numpy as np
from scipy.spatial import ConvexHull
from constants import *

class DanceMetricsEngine:
    def __init__(self, fps=30, history_size=30):
        self.fps = fps
        self.dt = 1.0 / fps
        self.history_size = history_size
        
        # Historical buffers
        self.positions_history = [] # List of (17, 3) arrays
        # Used for derivations
        self.velocities_history = []
        self.accelerations_history = []
        
        # Store limb angular velocity history for sync calculation
        self.omega_l_history = []
        self.omega_r_history = []

    def set_fps(self, fps):
        fps = max(float(fps), 1.0)
        self.fps = fps
        self.dt = 1.0 / fps
        self.history_size = max(2, int(round(fps)))
        self.positions_history = self.positions_history[-self.history_size:]
        self.velocities_history = self.velocities_history[-self.history_size:]
        self.accelerations_history = self.accelerations_history[-self.history_size:]
        self.omega_l_history = self.omega_l_history[-self.history_size:]
        self.omega_r_history = self.omega_r_history[-self.history_size:]

    def update(self, positions):
        """
        Positions: (17, 3) numpy array of joint coordinates (H36M format).
        Returns a dictionary of the 9 metrics.
        """
        if positions is None:
            return self.get_empty_metrics()

        self.positions_history.append(positions)
        if len(self.positions_history) > self.history_size:
            self.positions_history.pop(0)
            
        # Need at least previous frame for velocity/energy
        if len(self.positions_history) < 2:
            return self.get_empty_metrics()

        # Gather metrics
        energy, omegas = self._calculate_energy()
        sync_velocity, sync_correlation = self._calculate_synchronization(omegas)
        expansion = self._calculate_expansion(positions)
        curvature = self._calculate_curvature()
        height, sway = self._calculate_stability(positions)
        torque, jerk = self._calculate_transition(omegas)
        
        return {
            "energy": float(energy),
            "sync_velocity": float(sync_velocity),
            "sync_correlation": float(sync_correlation),
            "expansion": float(expansion),
            "curvature": float(curvature),
            "height": float(height),
            "sway": float(sway),
            "torque": float(torque),
            "jerk": float(jerk)
        }

    def get_empty_metrics(self):
        return {
            "energy": 0.0,
            "sync_velocity": 0.0,
            "sync_correlation": 0.0,
            "expansion": 0.0,
            "curvature": 0.0,
            "height": 0.0,
            "sway": 0.0,
            "torque": 0.0,
            "jerk": 0.0
        }

    def _normalize(self, v):
        norm = np.linalg.norm(v)
        if norm == 0: 
            return v
        return v / norm

    def _calculate_energy(self):
        """1. Energy (Intensity)"""
        t = len(self.positions_history) - 1
        pos_t = self.positions_history[t]
        pos_prev = self.positions_history[t-1]
        
        total_energy = 0.0
        omegas = {'Trunk': 0.0, 'L_Arm': 0.0, 'R_Arm': 0.0, 'L_Leg': 0.0, 'R_Leg': 0.0}
        
        for group_name, limbs in LIMB_GROUPS.items():
            group_weight = LIMB_WEIGHTS[group_name]
            group_energy = 0.0
            group_omega_sum = 0.0
            
            for start_idx, end_idx in limbs:
                v_t = pos_t[end_idx] - pos_t[start_idx]
                v_prev = pos_prev[end_idx] - pos_prev[start_idx]
                
                u_v_t = self._normalize(v_t)
                u_v_prev = self._normalize(v_prev)
                
                cos_theta = np.clip(np.dot(u_v_t, u_v_prev), -1.0, 1.0)
                delta_theta = np.arccos(cos_theta)
                
                omega = delta_theta / self.dt
                omega_sq = omega ** 2
                
                group_energy += (group_weight / len(limbs)) * omega_sq
                group_omega_sum += omega
                
            total_energy += group_energy
            omegas[group_name] = group_omega_sum / len(limbs)
            
        return total_energy, omegas

    def _calculate_synchronization(self, omegas):
        """2 & 3. Synchronization (Velocity & Correlation)"""
        omega_l = omegas['L_Arm'] + omegas['L_Leg']
        omega_r = omegas['R_Arm'] + omegas['R_Leg']
        
        self.omega_l_history.append(omega_l)
        self.omega_r_history.append(omega_r)
        
        if len(self.omega_l_history) > self.history_size:
            self.omega_l_history.pop(0)
            self.omega_r_history.pop(0)
            
        # Metric 2: Velocity Balance (Similarity of magnitudes)
        # 1.0 means perfectly balanced, 0.0 means one side is still
        max_omega = max(omega_l, omega_r)
        sync_velocity = 1.0 - (abs(omega_l - omega_r) / (max_omega + 1e-6))
            
        if len(self.omega_l_history) < 2:
            return sync_velocity, 0.0
            
        # Metric 3: Pearson correlation
        arr_l = np.array(self.omega_l_history)
        arr_r = np.array(self.omega_r_history)
        
        if np.std(arr_l) == 0 or np.std(arr_r) == 0:
            return sync_velocity, 0.0
            
        correlation = np.corrcoef(arr_l, arr_r)[0, 1]
        if np.isnan(correlation):
            return sync_velocity, 0.0
        return sync_velocity, correlation

    def _calculate_expansion(self, positions):
        """4. Body Geometry - Expansion (Volume)"""
        try:
            # Add small noise to prevent Coplanar errors in ConvexHull
            noise = np.random.normal(0, 1e-4, positions.shape)
            hull = ConvexHull(positions + noise)
            # Scale down volume for visualization ease (mm^3 is huge)
            return hull.volume / 100000000.0  
        except:
            return 0.0

    def _calculate_curvature(self):
        """5. Body Geometry - Roundness (Curvature)"""
        if len(self.positions_history) < 3:
            return 0.0
            
        end_effectors = [L_WRIST, R_WRIST, L_ANKLE, R_ANKLE]
        total_curvature = 0.0
        
        for joint in end_effectors:
            p_t0 = self.positions_history[-3][joint]
            p_t1 = self.positions_history[-2][joint]
            p_t2 = self.positions_history[-1][joint]
            
            # v = dp/dt
            v1 = (p_t1 - p_t0) / self.dt
            v2 = (p_t2 - p_t1) / self.dt
            
            # a = dv/dt
            a = (v2 - v1) / self.dt
            v_avg = (v1 + v2) / 2.0
            
            cross_prod = np.cross(v_avg, a)
            v_norm = np.linalg.norm(v_avg)
            
            if v_norm > 1e-5:
                # K = ||v x a|| / ||v||^3
                k = np.linalg.norm(cross_prod) / (v_norm ** 3)
                # Cap the curvature to avoid huge spikes when velocity is very slow
                total_curvature += min(k, 10.0)
                
        return total_curvature / len(end_effectors)

    def _calculate_stability(self, positions):
        """6 & 7. Stability (Height & Sway)"""
        com = np.zeros(3)
        for i in range(17):
            com += positions[i] * MASS_WEIGHTS[i]
            
        # Z axis is height. But often Y is height in image space.
        # In H36M from our conversion, Y is actually pointing down typically, 
        # or Z is depth depending on how it's defined. Let's use Y for height 
        # (normalized to screen) for simplicity, or just Euclidean distance from floor.
        # Let's use raw Y for now, and Sway is XZ plane distance.
        height_val = -com[1]  # Normally Y is down in image coordinates, negate it for "height"
        
        bos = (positions[L_ANKLE] + positions[R_ANKLE]) / 2.0
        
        sway_val = np.sqrt((com[0] - bos[0])**2 + (com[2] - bos[2])**2)
        
        return height_val / 1000.0, sway_val / 1000.0 # scale from mm to m

    def _calculate_transition(self, omegas):
        """8 & 9. Transition (Torque & Jerk)"""
        # We need historical angular velocities to compute derivatives.
        if not hasattr(self, 'omega_history'):
            self.omega_history = []
            
        self.omega_history.append(omegas)
        if len(self.omega_history) > 3:
            self.omega_history.pop(0)
            
        if len(self.omega_history) < 3:
            return 0.0, 0.0
            
        torque = 0.0
        jerk_cost = 0.0
        
        for group_name in LIMB_GROUPS.keys():
            w0 = self.omega_history[-3][group_name]
            w1 = self.omega_history[-2][group_name]
            w2 = self.omega_history[-1][group_name]
            
            alpha1 = (w1 - w0) / self.dt
            alpha2 = (w2 - w1) / self.dt
            
            jerk = (alpha2 - alpha1) / self.dt
            
            weight = LIMB_WEIGHTS[group_name]
            
            torque += weight * abs(alpha2)
            jerk_cost += weight * (jerk ** 2)
            
        return torque, jerk_cost
