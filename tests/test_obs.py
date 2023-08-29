from race_rl.env import RaceAviary
import numpy as np
import pybullet as p

env = RaceAviary(gui=True,track_path='assets/tracks/gates_inline.csv') 
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