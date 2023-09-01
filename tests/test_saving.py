from race_rl.env import RaceAviary, DeployType
import numpy as np
import pybullet as p
from multiprocessing import Manager, Value

manager=Manager()
start_pos = manager.dict()
env = RaceAviary(init_segment=0,
 start_dict=start_pos,
 gui=True,
 track_path='assets/tracks/2_gates.csv',
 deploy_type=DeployType.TRAINING,
 world_box=np.array([100, 100, 100]),
 acceptance_thr=0, 
 start_que_len=1) 
env.reset()

p.setGravity(0, 0, 0, physicsClientId=env.CLIENT)
p.applyExternalForce(env.DRONE_IDS[0], -1,  [1, 0, 0], [0, 0, 0], p.LINK_FRAME, env.CLIENT)

infos_vec = []
use_force = True
while True:
    #a= env.action_space.sample()
    a=np.array([-1,-1,-1,-1])
    # Use force
    if use_force:
        p.applyExternalForce(env.DRONE_IDS[0], -1,  [1, 0, 0], [0, 0, 0], p.LINK_FRAME, env.CLIENT)
    p.setGravity(0, 0, 0, physicsClientId=env.CLIENT)
    observation, reward, terminated, truncated, info = env.step(a)
    infos_vec.append(info)
    gate_reward = info["gate_reward"]
    progress_reward = info["progress_reward"]
    crash_reward = info["crash_reward"]
    omega_norm = info["omega_norm"]

    # print(f"Gate reward: {gate_reward}\nProgress reward: {progress_reward}\nCrash reward: {crash_reward}\nOmega norm: {omega_norm}") print(observation, reward)
    # print(env.curr_segment_idx)
    if truncated or terminated:
        print(env.start_dict)
        if len(env.start_dict.get(1, [])) == env.start_que_len:
            input()
            env.start_segment_idx = 1
            env.env_segment = env.track.segments[1]
        print(env.start_dict) 
        print(truncated, terminated)
        print(env.curr_segment_idx)
        env.reset()