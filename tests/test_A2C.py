import time
from typing import Union
import pickle
from pathlib import Path

import gymnasium as gym
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecNormalize,
    VecMonitor,
    StackedObservations,
    VecFrameStack

)

from stable_baselines3 import PPO, A2C
from stable_baselines3.common.callbacks import CheckpointCallback
import numpy as np
from multiprocessing import Manager, Value

import wandb
from wandb.integration.sb3 import WandbCallback

from gym_pybullet_drones.utils.enums import DroneModel, Physics
#from src.build_network import DronePolicy
from race_rl.env import RaceAviary, DeployType, DecreaseOmegaCoef

TRACK_PATH="assets/tracks/single_gate.csv"

def make_env(gui, num, init_segment, start_pos):
    #if num == 0:
        #gui = True
    env_builder = lambda: gym.make(
        "race-aviary-v0",
        init_segment=num,
        start_dict=start_pos,
        drone_model=DroneModel('cf2x'),
        # TODO - change the intialil_xyzs
        physics=Physics('pyb'),
        pyb_freq=240,
        ctrl_freq=60,
        gui=gui,
        record=False,
        gates_lookup=2,
        track_path=TRACK_PATH,
        user_debug_gui=False, 
        coef_gate_filed=0.1,
        coef_omega=0.0001,
    )
    return env_builder

def run():
    # Create enviroments
    
    # Create primitives for synchronization
    manager = Manager()
    start_pos = manager.dict()
    init_segment=Value('i', 0)

    envs = [make_env(False, i ,init_segment, start_pos) for i in range(12)]
    vec_env = SubprocVecEnv(envs)
        
    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=True,
    )

    # Add monitor wrapper
    vec_env = VecMonitor(
        vec_env,
        filename="logs"
    ) 

    # Add to add the accelerationi (but storing 2 speeds)
    vec_env = VecFrameStack(
        vec_env,
        2,
    )

    # Set up wdb
    curr_time = time.gmtime()
    run_id = f'{time.strftime("%d_%m_%y_%H_%M", curr_time)}_race_exp'
    
    # Create logs file
    logs_path = Path(f"./logs/models/{run_id}")
    if not logs_path.exists():
        logs_path.mkdir(parents=True)

    callbacks = []
        
    callbacks.append(CheckpointCallback(
            save_freq=1_000,
            save_path=f"./logs/models/{run_id}",
            name_prefix=f"{run_id}_race_model",
            save_replay_buffer=True,
            save_vecnormalize=True,
    ))

    callbacks.append(DecreaseOmegaCoef(10_000_000, 0))
    
    #model = PPO(
        ##DronePolicy,
        #"MlpPolicy", 
        #vec_env,
        #learning_rate=0.001,
        #gamma=0.95,
        #seed=42,
        #verbose=1,
        #batch_size=6*1024,

        #tensorboard_log="./logs/tensor_board/", 
        ##n_steps=240,
        #n_steps=1024,
    #)

    model = A2C(
        "MlpPolicy",
        vec_env,
        gamma=0.97,
        verbose=1,
        seed=42,
        tensorboard_log="./logs/tensor_board/", 
        n_steps=180
    )

    model.learn(
        total_timesteps=50_000_000,
        callback=callbacks,
        tb_log_name=run_id,
    )

    
if __name__ == "__main__":

    run() 