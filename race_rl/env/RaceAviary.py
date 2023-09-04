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
from random import sample
from enum import Enum

from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics

from race_rl.race_track import Track

class DeployType(Enum):
    TRAINING = 0
    VALIDATION = 1

class RaceAviary(BaseAviary):

    """Multi-drone environment class for control applications."""
    ################################################################################
    def __init__(self,
                 init_segment,
                 start_dict,
                 drone_model: DroneModel=DroneModel.CF2X,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 60,
                 gui=False,
                 record=False,
                 obstacles=False,
                 user_debug_gui=True,
                 output_folder='results',
                 track_path='assets/tracks/circle_track.csv',
                 asset_path='assets',
                 gates_lookup=2,
                 world_box=np.array([10,10,5]),
                 deploy_type=DeployType.TRAINING,
                 seed=42,
                 coef_gate_filed=0.01,
                 coef_omega=0.0001,
                 runs_que_len=20,
                 start_que_len=20,
                 acceptance_thr=0.8,
                 max_distance_segmnet=2
                    ):

        #### Constants #############################################
        self.G = 9.8
        self.RAD2DEG = 180/np.pi
        self.DEG2RAD = np.pi/180
        self.CTRL_FREQ = ctrl_freq
        self.PYB_FREQ = pyb_freq
        if self.PYB_FREQ % self.CTRL_FREQ != 0:
            raise ValueError('[ERROR] in BaseAviary.__init__(), pyb_freq is not divisible by env_freq.')
        self.PYB_STEPS_PER_CTRL = int(self.PYB_FREQ / self.CTRL_FREQ)
        self.CTRL_TIMESTEP = 1. / self.CTRL_FREQ
        self.PYB_TIMESTEP = 1. / self.PYB_FREQ
        #### Parameters ############################################
        self.NUM_DRONES = 1
        self.NEIGHBOURHOOD_RADIUS = np.inf
        #### Options ###############################################
        self.DRONE_MODEL = drone_model
        self.GUI = gui
        self.RECORD = record
        self.PHYSICS = Physics.PYB
        self.OBSTACLES = True
        self.USER_DEBUG = user_debug_gui
        self.URDF = self.DRONE_MODEL.value + ".urdf"
        self.OUTPUT_FOLDER = output_folder
        #### Load the drone properties from the .urdf file #########
        self.M, \
        self.L, \
        self.THRUST2WEIGHT_RATIO, \
        self.J, \
        self.J_INV, \
        self.KF, \
        self.KM, \
        self.COLLISION_H,\
        self.COLLISION_R, \
        self.COLLISION_Z_OFFSET, \
        self.MAX_SPEED_KMH, \
        self.GND_EFF_COEFF, \
        self.PROP_RADIUS, \
        self.DRAG_COEFF, \
        self.DW_COEFF_1, \
        self.DW_COEFF_2, \
        self.DW_COEFF_3 = self._parseURDFParameters()
        print("[INFO] BaseAviary.__init__() loaded parameters from the drone's .urdf:\n[INFO] m {:f}, L {:f},\n[INFO] ixx {:f}, iyy {:f}, izz {:f},\n[INFO] kf {:f}, km {:f},\n[INFO] t2w {:f}, max_speed_kmh {:f},\n[INFO] gnd_eff_coeff {:f}, prop_radius {:f},\n[INFO] drag_xy_coeff {:f}, drag_z_coeff {:f},\n[INFO] dw_coeff_1 {:f}, dw_coeff_2 {:f}, dw_coeff_3 {:f}".format(
            self.M, self.L, self.J[0,0], self.J[1,1], self.J[2,2], self.KF, self.KM, self.THRUST2WEIGHT_RATIO, self.MAX_SPEED_KMH, self.GND_EFF_COEFF, self.PROP_RADIUS, self.DRAG_COEFF[0], self.DRAG_COEFF[2], self.DW_COEFF_1, self.DW_COEFF_2, self.DW_COEFF_3))
        #### Compute constants #####################################
        self.GRAVITY = self.G*self.M
        self.HOVER_RPM = np.sqrt(self.GRAVITY / (4*self.KF))
        self.MAX_RPM = np.sqrt((self.THRUST2WEIGHT_RATIO*self.GRAVITY) / (4*self.KF))
        self.MAX_THRUST = (4*self.KF*self.MAX_RPM**2)
        if self.DRONE_MODEL == DroneModel.CF2X:
            self.MAX_XY_TORQUE = (2*self.L*self.KF*self.MAX_RPM**2)/np.sqrt(2)
        elif self.DRONE_MODEL == DroneModel.CF2P:
            self.MAX_XY_TORQUE = (self.L*self.KF*self.MAX_RPM**2)
        elif self.DRONE_MODEL == DroneModel.RACE:
            self.MAX_XY_TORQUE = (2*self.L*self.KF*self.MAX_RPM**2)/np.sqrt(2)
        self.MAX_Z_TORQUE = (2*self.KM*self.MAX_RPM**2)
        self.GND_EFF_H_CLIP = 0.25 * self.PROP_RADIUS * np.sqrt((15 * self.MAX_RPM**2 * self.KF * self.GND_EFF_COEFF) / self.MAX_THRUST)
        #### Create attributes for vision tasks ####################
        if self.RECORD:
            self.ONBOARD_IMG_PATH = os.path.join(self.OUTPUT_FOLDER, "recording_" + datetime.now().strftime("%m.%d.%Y_%H.%M.%S"))
            os.makedirs(os.path.dirname(self.ONBOARD_IMG_PATH), exist_ok=True)
        self.VISION_ATTR = False
        if self.VISION_ATTR:
            self.IMG_RES = np.array([64, 48])
            self.IMG_FRAME_PER_SEC = 24
            self.IMG_CAPTURE_FREQ = int(self.PYB_FREQ/self.IMG_FRAME_PER_SEC)
            self.rgb = np.zeros(((self.NUM_DRONES, self.IMG_RES[1], self.IMG_RES[0], 4)))
            self.dep = np.ones(((self.NUM_DRONES, self.IMG_RES[1], self.IMG_RES[0])))
            self.seg = np.zeros(((self.NUM_DRONES, self.IMG_RES[1], self.IMG_RES[0])))
            if self.IMG_CAPTURE_FREQ%self.PYB_STEPS_PER_CTRL != 0:
                print("[ERROR] in BaseAviary.__init__(), PyBullet and control frequencies incompatible with the desired video capture frame rate ({:f}Hz)".format(self.IMG_FRAME_PER_SEC))
                exit()
            if self.RECORD:
                for i in range(self.NUM_DRONES):
                    os.makedirs(os.path.dirname(self.ONBOARD_IMG_PATH+"/drone_"+str(i)+"/"), exist_ok=True)
        #### Connect to PyBullet ###################################
        if self.GUI:
            #### With debug GUI ########################################
            self.CLIENT = p.connect(p.GUI) # p.connect(p.GUI, options="--opengl2")
            for i in [p.COV_ENABLE_RGB_BUFFER_PREVIEW, p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW]:
                p.configureDebugVisualizer(i, 0, physicsClientId=self.CLIENT)
            p.resetDebugVisualizerCamera(cameraDistance=3,
                                         cameraYaw=-30,
                                         cameraPitch=-30,
                                         cameraTargetPosition=[0, 0, 0],
                                         physicsClientId=self.CLIENT
                                         )
            ret = p.getDebugVisualizerCamera(physicsClientId=self.CLIENT)
            print("viewMatrix", ret[2])
            print("projectionMatrix", ret[3])
            if self.USER_DEBUG:
                #### Add input sliders to the GUI ##########################
                self.SLIDERS = -1*np.ones(4)
                for i in range(4):
                    self.SLIDERS[i] = p.addUserDebugParameter("Propeller "+str(i)+" RPM", 0, self.MAX_RPM, self.HOVER_RPM, physicsClientId=self.CLIENT)
                self.INPUT_SWITCH = p.addUserDebugParameter("Use GUI RPM", 9999, -1, 0, physicsClientId=self.CLIENT)
        else:
            #### Without debug GUI #####################################
            self.CLIENT = p.connect(p.DIRECT)
            #### Uncomment the following line to use EGL Render Plugin #
            #### Instead of TinyRender (CPU-based) in PYB's Direct mode
            # if platform == "linux": p.setAdditionalSearchPath(pybullet_data.getDataPath()); plugin = p.loadPlugin(egl.get_filename(), "_eglRendererPlugin"); print("plugin=", plugin)
            if self.RECORD:
                #### Set the camera parameters to save frames in DIRECT mode
                self.VID_WIDTH=int(640)
                self.VID_HEIGHT=int(480)
                self.FRAME_PER_SEC = 24
                self.CAPTURE_FREQ = int(self.PYB_FREQ/self.FRAME_PER_SEC)
                self.CAM_VIEW = p.computeViewMatrixFromYawPitchRoll(distance=3,
                                                                    yaw=-30,
                                                                    pitch=-30,
                                                                    roll=0,
                                                                    cameraTargetPosition=[0, 0, 0],
                                                                    upAxisIndex=2,
                                                                    physicsClientId=self.CLIENT
                                                                    )
                self.CAM_PRO = p.computeProjectionMatrixFOV(fov=60.0,
                                                            aspect=self.VID_WIDTH/self.VID_HEIGHT,
                                                            nearVal=0.1,
                                                            farVal=1000.0
                                                            )
        #### Set initial poses #####################################
        self.seed=seed
        np.random.seed(self.seed)
        # Create track object
        self.track=Track(
            track_path, 
            asset_path,
            self.CLIENT,
        )
        self.track.reloadGates()
        self.NUMBER_GATES=len(self.track.segments)

        self.deploy_type=deploy_type
        self.gates_lookup=gates_lookup
        self.world_box=np.array(world_box)
        self.coef_gate_filed=coef_gate_filed
        self.coef_omega=coef_omega
        self.infos={}
        self.max_distance_segmnet=max_distance_segmnet
        self.start_segment_idx=init_segment%self.NUMBER_GATES
        # with init_segment.get_lock():
        #     init_segment.value += 1
    
        # Track succes rate of a segment
        self.env_segment=self.track.segments[self.start_segment_idx]
        self.curr_segment_idx=self.start_segment_idx
        self.current_segment=self.env_segment
        # Queue with last 100 runs successes
        self.runs=deque(maxlen=runs_que_len)
        self.runs_que_len=runs_que_len
        if self.start_segment_idx == 0:
            start_pos, start_quat=self.track.getTrackStart()
        else:
            start_pos, start_quat=self.env_segment.startPosition()
        self.prev_projection=self.env_segment.projectPoint(start_pos)
        # TODO - check if the 20 last sucessuf runs is enougth
        rpy=p.getEulerFromQuaternion(start_quat)
        initial_state = np.array([
            start_pos[0], start_pos[1], start_pos[2],
            start_quat[0], start_quat[1], start_quat[2], start_quat[3],
            rpy[0], rpy[1], rpy[2],
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            -1.0, -1.0, -1.0, -1.0,
        ])
        self.start_dict=start_dict
        self.start_dict[self.start_segment_idx] = deque([initial_state], maxlen=start_que_len)
        self.start_que_len=start_que_len
        self.acceptance_thr=acceptance_thr
        #start_dict[self.start_segment_idx] = [initial_state]

        self.INIT_XYZS = start_pos
        self.INIT_RPYS = start_quat

        #### Create action and observation spaces ##################
        self.action_space = self._actionSpace()
        self.observation_space = self._observationSpace()
        #### Housekeeping ##########################################
        self._housekeeping()
        #### Update and store the drones kinematic information #####
        self._updateAndStoreKinematicInformation()
        #### Start video recording #################################
        self._startVideoRecording()


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
            *[0, -2*np.pi, -np.pi, 0]*self.gates_lookup])
        obs_upper_bound=np.array([
            np.inf, np.inf, np.inf, 
            1, 1, 1, 1, 1, 1, 1, 1, 1, 
            np.inf, np.inf, np.inf,
            *[np.inf, 2*np.pi, np.pi, np.pi]*self.gates_lookup])
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
        start_pos, start_quat=state[0:3], state[3:7]
        for segment in self.track.segments[self.curr_segment_idx: self.curr_segment_idx+self.gates_lookup]:
            gate_lookup.append(
                segment.getRelativeObs((start_pos, start_quat))
            ) 
            start_pos=segment.gate.pos
            start_quat=segment.gate.quat

        while len(gate_lookup) < self.gates_lookup:
            # When consideting last gate add a virtual gate in front of a current gate
            gate_lookup.append(np.array([2, 0, np.pi/2, 0]))

        rot_matrix = p.getMatrixFromQuaternion(state[3:7])

        obs = np.hstack([state[10:13], 
                         rot_matrix, # Change to quatetions
                         state[13:16], 
                         *gate_lookup])
        
        obs = obs.astype(np.float64)
        if not self.infos.get("TimeLimit.terminated", False) and not self.infos.get("TimeLimit.truncated", False): 
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
        gate_reward = self.current_segment.gate.field_reward(d_pos)
        self.infos["gate_reward"] = gate_reward
        reward += self.coef_gate_filed * gate_reward

        # 2. Progress calculation
        projection = self.env_segment.projectPoint(d_pos)
        progress_reward = (projection - self.prev_projection) 
        self.infos["progress_reward"] = progress_reward
        reward += progress_reward
        self.prev_projection = projection

        # 3. Add the colision penaulty or out of bound penauly
        diff_vec = self.current_segment.gate.pos-d_pos
        dg = np.linalg.norm(diff_vec) # Reward for crash as in paper
        wg=self.current_segment.gate.scale
        crash_reward = 0
        if (len(p.getContactPoints(bodyA=self.DRONE_IDS[0],
                                  physicsClientId=self.CLIENT)) != 0 or 
            np.any(np.abs(d_pos) > self.world_box) or
             self.max_distance_segmnet <= self.current_segment.distanceToSegment(d_pos)):
             #or (not self._gateScored() and self.current_segment.segmentFinished(d_pos))):

            crash_reward = -min((dg/wg)**2, 20)

        self.infos["crash_reward"] = crash_reward
        reward += crash_reward

        # 4. Add reguralization on the angular velocity    
        omega = state[13:16]
        omega_norm = np.linalg.norm(omega)**2
        self.infos["omega_norm"] = omega_norm
        reward -= self.coef_omega * omega_norm

        return reward
    
    ################################################################################
    def _computeTerminated(self):
        """Computes the current terminated value(s)."""
        terminated = False

        state = self._getDroneStateVector(0)
        pos = state[0:3]

        if self.curr_segment_idx == self.NUMBER_GATES:
            terminated = True

        self.infos["TimeLimit.terminated"] = terminated
        return terminated
           
    ################################################################################
    def _computeTruncated(self):
        truncated = False 
        
        state = self._getDroneStateVector(0)
        pos = state[0:3]

        # Flew too far
        if np.any(np.abs(pos) > self.world_box):
            truncated = True

        if self.max_distance_segmnet <= self.current_segment.distanceToSegment(pos):
            truncated = True
        print("Distance to segmen:" ,self.current_segment.distanceToSegment(pos), pos)
        
        # Termiante when detected collision
        if len(p.getContactPoints(bodyA=self.DRONE_IDS[0],
                                 physicsClientId=self.CLIENT)) != 0:
            truncated = True

        # Truncate if segment completed but not scored
        # TODO - does it works ???
        if (not self._gateScored() and 
            self.current_segment.segmentFinished(pos) and
            self.deploy_type == DeployType.TRAINING):
            truncated = True

        self.infos["TimeLimit.truncated"] = truncated
        return truncated

    ################################################################################
    def _gateScored(self):
        """ Compute if flew throught a gate """
        scored=False
        d_pos, _ = p.getBasePositionAndOrientation(self.DRONE_IDS[0])
        g_pos, g_ort = p.getBasePositionAndOrientation(self.current_segment.gate.urdf_id)
        
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
        wg = self.current_segment.gate.scale
        
        x, y, z = t_pos[0], t_pos[1], t_pos[2]
        # Not shure if ok, must check 
        if (0.0 < x
            and abs(y) < wg/2 
            and abs(z) < wg/2
            and t_vel[0] <= 0):
            scored = True
    
        return scored

    ################################################################################
    def _addObstacles(self):
        """Add obstacles to the environment.

        These obstacles are loaded from standard URDF files included in Bullet.

        """
        self.track.reloadGates()

    ################################################################################
    def _housekeeping(self):
        # Sample some initial statring possition
        curr_start = self.start_dict[self.start_segment_idx]
        # Dicinary constains entire state of the drone 
        state = sample(list(curr_start), 1)[0]
        self.current_segment=self.env_segment
        self.curr_segment_idx=self.start_segment_idx

        #### Initialize/reset counters and zero-valued variables ###
        self.RESET_TIME = time.time()
        self.step_counter = 0
        self.first_render_call = True
        self.X_AX = -1*np.ones(self.NUM_DRONES)
        self.Y_AX = -1*np.ones(self.NUM_DRONES)
        self.Z_AX = -1*np.ones(self.NUM_DRONES)
        self.GUI_INPUT_TEXT = -1*np.ones(self.NUM_DRONES)
        self.USE_GUI_RPM=False
        self.last_input_switch = 0
        self.last_action = state[16:21].reshape(1, 4)
        self.last_clipped_action = state[16:21].reshape(1, 4)
        self.gui_input = np.zeros(4)
        #### Initialize the drones kinemaatic information ##########
        self.pos = state[0:3].reshape(1, 3)
        self.quat = state[3:7].reshape(1, 4)
        self.rpy = state[7:10].reshape(1, 3)
        self.vel = state[10:13].reshape(1, 3)
        self.ang_v = state[13:16].reshape(1, 3)
        if self.PHYSICS == Physics.DYN:
            self.rpy_rates = np.zeros((self.NUM_DRONES, 3))

        #### Set PyBullet's parameters #############################
        p.setGravity(0, 0, -self.G, physicsClientId=self.CLIENT)
        p.setRealTimeSimulation(0, physicsClientId=self.CLIENT)
        p.setTimeStep(self.PYB_TIMESTEP, physicsClientId=self.CLIENT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.CLIENT)
        #### Load ground plane, drone and obstacles models #########
        self.PLANE_ID = p.loadURDF("plane.urdf", physicsClientId=self.CLIENT)

        self.DRONE_IDS = np.array([p.loadURDF(pkg_resources.resource_filename('gym_pybullet_drones', 'assets/'+self.URDF),
                                              self.pos[i,:],
                                              self.quat[i,:],
                                              flags = p.URDF_USE_INERTIA_FROM_FILE,
                                              physicsClientId=self.CLIENT
                                              ) for i in range(self.NUM_DRONES)])

        #### Remove default damping #################################
        # for i in range(self.NUM_DRONES):
        #     p.changeDynamics(self.DRONE_IDS[i], -1, linearDamping=0, angularDamping=0)
        #### Show the frame of reference of the drone, note that ###
        #### It severly slows down the GUI #########################
        if self.GUI and self.USER_DEBUG:
            for i in range(self.NUM_DRONES):
                self._showDroneLocalAxes(i)
        #### Disable collisions between drones' and the ground plane
        #### E.g., to start a drone at [0,0,0] #####################
        # for i in range(self.NUM_DRONES):
            # p.setCollisionFilterPair(bodyUniqueIdA=self.PLANE_ID, bodyUniqueIdB=self.DRONE_IDS[i], linkIndexA=-1, linkIndexB=-1, enableCollision=0, physicsClientId=self.CLIENT)
        if self.OBSTACLES:
            self._addObstacles() 

        
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


    ################################################################################   
    def _preprocessAction(self,
                          action
                          ):
        """Pre-processes the action passed to `.step()` into motors' RPMs.

        Must be implemented in a subclass.

        Parameters
        ----------
        action : ndarray | dict[..]
            The input action for one or more drones, to be translated into RPMs.

        """
        return self._normalizedActionToRPM(action)

    ################################################################################
    def _normalizedActionToRPM(self,
                               action
                               ):
        """De-normalizes the [-1, 1] range to the [0, MAX_RPM] range.

        Parameters
        ----------
        action : ndarray
            (4)-shaped array of ints containing an input in the [-1, 1] range.

        Returns
        -------
        ndarray
            (4)-shaped array of ints containing RPMs for the 4 motors in the [0, MAX_RPM] range.

        """
        if np.any(np.abs(action) > 1):
            print("\n[ERROR] it", self.step_counter, "in BaseAviary._normalizedActionToRPM(), out-of-bound action")
        return np.where(action <= 0, (action+1)*self.HOVER_RPM, self.HOVER_RPM + (self.MAX_RPM - self.HOVER_RPM)*action) # Non-linear mapping: -1 -> 0, 0 -> HOVER_RPM, 1 -> MAX_RPM`

    def step(self, 
             action
             ):


        state = self._getDroneStateVector(0)
        pos = state[0:3]

        if self.curr_segment_idx == self.start_segment_idx:
            if self.current_segment.segmentFinished(pos):
                self.runs.append(self._gateScored())
                if sum(list(self.runs)) / self.runs_que_len > self.acceptance_thr:
                    start_state = self.start_dict.get(self.start_segment_idx+1, deque(maxlen=self.start_que_len))
                    start_state.append(state)
                    self.start_dict[self.start_segment_idx+1] = start_state


        if ((self._gateScored() or self.deploy_type == DeployType.VALIDATION) and 
        self.current_segment.segmentFinished(pos)):
            self.curr_segment_idx += 1
            self.current_segment=self.track.segments[self.curr_segment_idx%self.NUMBER_GATES]

        return super().step(action)

    def changeOmegaCoef(self, coef_omega: float):
        self.coef_omega=coef_omega
