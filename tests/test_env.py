from race_rl.env import RaceAviary
import numpy as np

env = RaceAviary(gui=True)
env.reset()

while True:
    a= env.action_space.sample()
    print(a)
    env.step(a)
    input()