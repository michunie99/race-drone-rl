from race_rl.env import RaceAviary
import numpy as np
from multiprocessing import Process, Value, Manager
from time import sleep


def proc(init_segment, start_pos):
    env = RaceAviary(init_segment, start_pos, gui=True)
    env.reset()

    while True:
        a= env.action_space.sample()
        print(a)
        env.step(a)
        sleep(0.1)

with Manager() as manager:
    init_segment=Value('i', 0)
    start_pos = manager.dict()
    processes=[]

    for _ in range(4):
        processes.append(Process(target=proc, args=(init_segment, start_pos)))

    for proc in processes:
        proc.start()

    while True:
        pass