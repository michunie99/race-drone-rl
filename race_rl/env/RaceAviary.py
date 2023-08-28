import sys
import time
from collections import deque
from itertools import cycle
import numpy as np
from gymnasium import spaces
import pybullet as p
import pybullet_data
import pkg_resources
from multiprocessing import Process, Value, Manager
from collections import deque

from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics

from race_rl.race_track import Track

class RaceAviary(BaseAviary):
    """Multi-drone environment class for control applications."""
    #TODO - add global statring positions buffer
    ################################################################################
    init_segment=Value('i', 0)

    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 240,
                 gui=False,
                 record=False,
                 obstacles=False,
                 user_debug_gui=True,
                 output_folder='results',
                 track_path='assets/tracks/circle_track.csv',
                 asset_path='assets',
                 gates_lookup=2,
                 world_box=np.array([10,10,10]),
                 type="train",
                 seed=42,
                 coef_gate_filed=0.001,
                 omega_coef=0.001,
                 ):

        # TODO: more seeds ???
        self.seed=seed
        np.random.seed(self.seed)
        self.gates_lookup=gates_lookup
        self.world_box=world_box
        self.coef_gate_filed=coef_gate_filed
        self.omega_coef=omega_coef
        self.infos={}
        self.start_segment=self.init_segment.value
        with self.init_segment.get_lock():
            self.init_segment.value += 1

        # TODO - should work but not sure
        try:
            manager = getattr(type(self), 'manager')
        except AttributeError:
            manager = type(self).manager = Manager()
            start_pos = type(self).start_pos = manager.dict()

        # TODO
        # 1. Initial from chosen segment
        # 2. pyb_freq: 240
        # 3. ctrl_freq: 50hz
        super().__init__(drone_model=drone_model,
                         num_drones=1,
                         neighbourhood_radius=np.inf,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record,
                         obstacles=False,
                         user_debug_gui=user_debug_gui,
                         output_folder=output_folder,
                        )

        # Create track object
        self.track=Track(
            track_path, 
            asset_path,
            self.CLIENT,
        )
        # Track succes rate of a segment
        self.env_segment=self.track.segments[self.start_segment]
        # Queue with last 100 runs successes
        self.runs=deque(maxlen=100)
        start_pos, _=self.env_segment.startPosition()
        self.prev_projection=self.env_segment.projectPoint(start_pos)
        # TODO - add starting positon from the segment
        input()


    ################################################################################
    def _observationSpace(self):
        """Returns the observation space of the environment.

        Must be implemented in a subclass.

        """
        # Obsersvations: Vx, Vy, Vz, rot_matrix, wx, wy, wz, 4*gates_lookup
        obs_lower_bound=np.array([
            -np.inf, -np.inf, -np.inf, 
            -1, -1, -1, -1, -1, -1, -1, -1, -1, 
            -np.inf, -np.inf, -np.inf,
            *[0, -2*np.pi, -np.pi]*self.gates_lookup])
        obs_upper_bound=np.array([
            np.inf, np.inf, np.inf, 
            1, 1, 1, 1, 1, 1, 1, 1, 1, 
            np.inf, np.inf, np.inf,
            *[np.inf, 2*np.pi, np.pi]*self.gates_lookup])
        return spaces.Box(
            low=obs_lower_bound,
            high=obs_upper_bound,
            dtype=np.float64,
        )

    ################################################################################
    def _actionSpace(self):
        """Returns the action space of the environment.

        Returns
        -------
        dict[str, ndarray]
            A Dict of Box(4,) with NUM_DRONES entries,
            indexed by drone Id in string format.

        """
        #### Action vector ######## P0            P1            P2            P3
        act_lower_bound = np.array([-1.,-1.,-1.,-1.])
        act_upper_bound = np.array([1., 1., 1., 1.])
        return spaces.Box(  low=act_lower_bound,
                            high=act_upper_bound,
                            dtype=np.float64)
    
    ################################################################################

    def _computeObs(self):
        """Returns the current observation of the environment. """
        state = self._getDroneStateVector(0)
        # Calculate obseraction between drone and gate
        # obj = ([x, y, z], [qx, qy, qz, qw])
        drone_obj = state[0:3], state[3:7]

        gate_lookup = []
        start_pos=state[0:3]
        for segment in self.track.segments[self.curr_seg+1: self.curr_seg+self.gates_lookup]:
            gate_lookup.append(
                segment.getRelativeObs(start_pos)
            ) 
            start_pos=segment.gate.pos

        rot_matrix = p.getMatrixFromQuaternion(state[3:7])

        obs = np.hstack([state[10:13], 
                         rot_matrix, 
                         state[13:16], 
                         *gate_lookup]).reshape(self.obs_size,)
        
        obs = obs.astype(np.float64)
        if not self.infos["TimeLimit.terminated"] and not self.infos["TimeLimit.truncated"]: 
            self.infos["terminal_observation"] = obs
        return obs

    ################################################################################

    def _computeReward(self):
        """Computes the current reward value(s).

        Returns
        -------
        int

        """
        reward = 0 
        state = self._getDroneStateVector(0)
        # Calculate obseraction between drone and gate
        # obj = ([x, y, z], [qx, qy, qz, qw])
        d_pos, d_quat = state[0:3], state[3:7]

        # 1. Add gate reward
        reward += self.coef_gate_filed * self.env_segment.gate.field_reward(d_pos)

        # 2. Progress calculation
        projection = self.env_segment.projectPoint(d_pos)
        reward += (projection - self.prev_projection) 
        self.prev_projection = projection
        return reward
    
    ################################################################################
    
    def _computeTerminated(self):
        """Computes the current terminated value(s)."""
        terminated = False
        
        if self.current_gate_idx == self.NUMBER_GATES:
            terminated = True
        
        self.infos["TimeLimit.terminated"] = terminated
        return terminated
           
    ################################################################################
    def _computeTruncated(self):
        truncated = False 
        
        state = self._getDroneStateVector(0)
        pos = state[0:3]
        # print(np.abs(pos) > self.world_box_size)
        # Flew too far
        if np.any(np.abs(pos) > self.world_box_size):

            truncated = True
        
        # Termiante when detected collision
        if len(p.getContactPoints(bodyA=self.DRONE_IDS[0],
                                 physicsClientId=self.CLIENT)) != 0:
            truncated = True
        
        self.infos["TimeLimit.truncated"] = truncated
        return truncated
    ################################################################################
    
    def _gateScored(self, thr):
        """ Compute if flew throught a gate """
        # TODO - change to check if flew throught a gate
        scored=False
        d_pos, _ = p.getBasePositionAndOrientation(self.DRONE_IDS[0])
        g_pos, g_ort = p.getBasePositionAndOrientation(self.GATE_IDS[self.current_gate_idx])
        
        state = self._getDroneStateVector(0)
        vel = state[10:13]
    
        # diff_vec = np.array(d_pos) - np.array(g_pos)
        diff_vec = np.array(g_pos) - np.array(d_pos)
        # Transform drone position to gate reference frame
        t_pos, _ = p.invertTransform(position=diff_vec,
                                    orientation=g_ort)
        
        # Transform velocity
        t_vel, _ = p.invertTransform(position=vel,
                                    orientation=g_ort)
        wg = self.gate_sizes[self.current_gate_idx]
        
        x, y, z = t_pos[0], t_pos[1], t_pos[2]
        
        if (0.0 < abs(x) < 0.01 
            and abs(y) < wg/2 
            and abs(z) < wg/2
            and t_vel[0] <= 0):
            scored = True
    
        return scored

    ################################################################################

    def _housekeeping(self):
        super()._housekeeping()
        
    ################################################################################
    
    def _quaternionMultiply(quaternion1, quaternion0):
        """Return multiplication of two quaternions.

            >>> q = quaternion_multiply([4, 1, -2, 3], [8, -5, 6, 7])
            >>> numpy.allclose(q, [28, -44, -14, 48])
            True

            """
        w0, x0, y0, z0 = quaternion0
        w1, x1, y1, z1 = quaternion1
        return np.array([
            -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0, x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
            -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0, x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0
        ],
                            dtype=np.float64)
     
    ################################################################################   
    
    def _computeInfo(self):
        # Return the last observation on episode termination
        
        return self.infos
