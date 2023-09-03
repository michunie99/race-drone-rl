import argparse
import time
from typing import Union
import pickle
from pathlib import Path

import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize, VecMonitor, VecFrameStack
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
import numpy as np
from multiprocessing import Manager

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from race_rl.nn import DronePolicy
from race_rl.env import RaceAviary, DeployType

def parse_args():
    parser = argparse.ArgumentParser()
    
    # Env parameters
    parser.add_argument("--num-valid", type=int, default=10,
                        help="Number of validation enviroments")
    parser.add_argument("--track-path", type=str, default='tracks/single_gate.csv',
                            help="Path for the track file")
    parser.add_argument("--model-dir", type=str, default=None,
                        help="path to models dir")
    parser.add_argument("--run-number", type=int, default=None,
                        help="number of run to run")
#    parser.add_argument("--norm-path", type=str, default=None,
#                        help="Path to normaliztion script")
#    parser.add_argument("--args-path", type=str, default=None,
#                        help="Path to model save args")
    
    args = parser.parse_args()
    return args

def load_experiment(args_path):
    with open(args_path, "br") as f:
        o_args = pickle.load(f)
    return o_args 

def make_env(args, gui, start_pos):
    env = make_vec_env(
        "race-aviary-v0",
        env_kwargs={
            "init_segment":0,
            "start_dict":start_pos,
            "pyb_freq":args.pdb_freq,
            "ctrl_freq":args.crt_freq,
            "coef_gate_filed":args.field_coef,
            "coef_omega":args.omega_coef,
            "drone_model":DroneModel('cf2x'),
            "physics":Physics('pyb'),
            "gui":gui,
            "record":False,
            "gates_lookup":args.gate_lookup,
            "world_box":args.world_box,
            "track_path":args.track_path,
            "user_debug_gui":False, 
            "deploy_type":DeployType.TRAINING
        }
    )
    return env

def run(args):
    model_dir = Path(args.model_dir)
    model_path = model_dir / f"{model_dir.stem}_race_model_{args.run_number}_steps.zip"
    norm_path = model_dir / f"{model_dir.stem}_race_model_vecnormalize_{args.run_number}_steps.pkl"
    args_path = model_dir / "args.pkl"
    env_args = load_experiment(args_path)

    manager = Manager()
    start_pos = manager.dict()
    # Create enviroments
    env = make_env(env_args, True, start_pos)
        
    # Create a normalization wrapper
    env = VecNormalize.load(
            norm_path,
            env,
    )

    env = VecFrameStack(
        env,
        2
    )


    env.training = False
    env.norm_reward = False

    model = PPO.load(model_path)

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=50)
    print(f"Mean reward = {mean_reward:.2f} +/- {std_reward:.2f}")
    
if __name__ == "__main__":
    args = parse_args()

    run(args) 
