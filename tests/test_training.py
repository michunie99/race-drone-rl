import time
from typing import Union
import pickle
from pathlib import Path

import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize, VecMonitor
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
import numpy as np
from multiprocessing import Manager, Value

import wandb
from wandb.integration.sb3 import WandbCallback

from gym_pybullet_drones.utils.enums import DroneModel, Physics
#from src.build_network import DronePolicy
from race_rl.env import RaceAviary

TRACK_PATH="assets/tracks/2_gates.csv"

def make_env(gui, num, init_segment, start_pos):
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
        coef_gate_filed=0.001,
        coef_omega=0.0001,
    )
    return env_builder

def run():
    # Create enviroments
    
    # Create primitives for synchronization
    manager = Manager()
    start_pos = manager.dict()
    init_segment=Value('i', 0)

    envs = [make_env(True, i ,init_segment, start_pos) for i in range(2)]
    vec_env = SubprocVecEnv(envs)
        
    # vec_env = VecNormalize(
    #     vec_env,
    #     norm_obs=True,
    #     norm_reward=True,
    # )

    # Add monitor wrapper
    vec_env = VecMonitor(
        vec_env,
        filename="logs"
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
            save_freq=1000,
            save_path=f"./logs/models/{run_id}",
            name_prefix=f"{run_id}_race_model",
            save_replay_buffer=True,
            save_vecnormalize=True,
    ))
    
    model = PPO(
        #DronePolicy,
        "MlpPolicy",
        vec_env,
        learning_rate=0.001,
        gamma=0.98,
        seed=42,
        verbose=1,
        tensorboard_log="./logs/tensor_board/", 
        n_steps=512,
    )

    model.learn(
        total_timesteps=200_000,
        callback=callbacks,
        tb_log_name=run_id,
    )

    
if __name__ == "__main__":

    run() 