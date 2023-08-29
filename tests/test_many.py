from race_rl.env import RaceAviary
import numpy as np
from multiprocessing import Process
from time import sleep

def proc():
    env = RaceAviary(gui=True)
    env.reset()

    while True:
        a= env.action_space.sample()
        print(a)
        env.step(a)
        sleep(0.1)

processes=[]
for _ in range(4):
    processes.append(Process(target=proc))

for proc in processes:
    proc.start()