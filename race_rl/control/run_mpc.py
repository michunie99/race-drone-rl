import pybullet as p
import numpy as np

from race_rl.env import MPCAviary
from race_rl.control import MPC

from gym_pybullet_drones.utils.enums import DroneModel


def run_mpc(track_path: str, mpc_file: str, gui: bool = True):
    _T = 2
    _dt = 0.02
    _N = _T / _dt
    pyb_freq = 250
    ctrl_freq = 50

    mpc = MPC(_T, _dt, mpc_file)
    env = MPCAviary(
        gui = gui,
        pyb_freq = pyb_freq,
        ctrl_freq = ctrl_freq,
        track_path=track_path,
        drone_model=DroneModel.CF2P
    )

    infos = []
    # env.step_counter

    obs, _ = env.reset()
    t = 0
    while True:
        # t = env.step_counter / pyb_freq
        ref_traj = env.planer.getRefPath(t, _dt, _T)
        t += 1/ctrl_freq

        obs = obs.tolist()
        ref_traj = obs + ref_traj
        quad_act, pred_traj = mpc.solve(ref_traj)

        # quad_act = np.array([quad_act[1], quad_act[0], quad_act[3], quad_act[2]])
        # Perform simulation step
        obs, reward, terminated, truncated, info = env.step(quad_act)

        info = {
            "quad_obs": obs,
            "quad_act": quad_act,
            "pred_traj": pred_traj
        }
        infos.append(info) 

        if terminated or truncated:
            break

    return infos, t

if __name__ == "__main__":
    mpc_file = "mpc.so"
    track_path = "assets/tracks/thesis-tracks/split_s.csv"
    gui = True

    run_mpc(track_path, mpc_file, gui) 
