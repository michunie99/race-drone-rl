from pathlib import Path

import pybullet as p
import numpy as np

class Gate:
    def __init__(
            self,
            pos: np.array,
            quat: np.array,
            scale: float,
            asset: Path,
            clientID: int,
            ):
        # Save position and calculate normal
        self.pos=pos
        self.quat=quat
        self.scale=scale
        # Load asset to pybullet client
        self.urdf_id=self._loadBullet(asset, clientID)

    def _loadBullet(self, asset: Path, clientID: int) -> int:
        # import pdb; pdb.set_trace()
        urdf_id=p.loadURDF(
                str(asset),
                self.pos,
                self.quat,
                globalScaling=self.scale,
                useFixedBase=1,
                physicsClientId=clientID,
        )
        return urdf_id

    def field_reward(self, d_pos: np.array) -> float:
        diff_vec = d_pose-self.pos 
        # Transform drone position to gate reference frame
        t_pos, _ = p.invertTransform(position=diff_vec,
                                     orientation=self.quat)

        dp, dn = t_pos[0], np.sqrt(t_pos[1]**2 + t_pos[2]**2)
        # TODO - check the cooeficent of 1.5
        f = lambda x: max(1-x/1.5, 0.0)
        v = lambda x, y: max((1- y) * (x/6.0), 0.05)
        filed_reward = -f(dp)**2 * (1 - np.exp(- 0.5 * dn**2 / v(wg, f(dp))))
        return filed_reward
    
    
