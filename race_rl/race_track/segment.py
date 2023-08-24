from typing import List, Tuple

import pybullet as p
import numpy as np

from .gate import Gate
from .utils import calculateRelativeObseration

class Segment:
    def __init__(
            self,
            start_pos: np.array,
            gate: Gate,
            ):
        # Elements related to gates and their relative pos
        self.start_pos=start_pos
        self.gate=gate
        self.segment=gate.pos-start_pos
        self.norm_segment=np.linalg.norm(self.segment)

    def projectPoint(self, pos: np.array) -> np.array:
        projection=np.dot(pos-self.gate.pos, self.segment)
        return projection/self.norm_segment
    
    def getRelativeObs(self, d_obj: Tuple[np.array, np.array]) -> np.array:
        g_obj=self.gate.pos, self.gate.ort
        return calculateRelativeObseration(d_obj, g_obj)        

    #@property
    def startPosition(self):
        pos=self.start_pos+self.segment/2
        g_dir=self.segment/self.norm_segment
        # Cast to Z axis
        z_cast=np.dot(g_dir, np.array([0,0,1]))
        # Calculate projecion on XY plane
        xy_cast=self.segment-z_cast
        oz_angel=np.arctan2(xy_cast[1], xy_cast[0])
        ort=p.getQuaternionFromEuler([oz_angel, 0, 0])
        return pos, ort
