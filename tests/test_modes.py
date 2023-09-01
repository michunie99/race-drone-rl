from race_rl.env import RaceAviary, DeployType
import numpy as np
import pybullet as p
from multiprocessing import Manager, Value

manager=Manager()
start_pos = manager.dict()
env = RaceAviary(init_segment=0,
 start_dict=start_pos,
 gui=True,
 track_path='assets/tracks/gates_inline.csv',
 deploy_type=DeployType.VALIDATION,
 world_box=np.array([100, 100, 100])) 
env.reset()

p.setGravity(0, 0, 0, physicsClientId=env.CLIENT)
p.applyExternalForce(env.DRONE_IDS[0], -1,  [1, 0, 0], [0, 0, 0], p.LINK_FRAME, env.CLIENT)

while True:
    #a= env.action_space.sample()
    a=np.array([-1,-1,-1,-1])
    p.applyExternalForce(env.DRONE_IDS[0], -1,  [1, 0, 0], [0, 0, 0], p.LINK_FRAME, env.CLIENT)
    print(a)
    observation, reward, terminated, truncated, info = env.step(a)
    print(observation, reward)
    print(env.curr_segment_idx)
    if truncated or terminated:
        print(truncated, terminated)
        print(env.curr_segment_idx)
        input()
        env.reset()