from typing import List

import pybullet as p
import numpy as np

from .gate import Gate

class Segment:
    def __init__(
            self,
            start_gate: Gate,
            end_gate: Gate,
            ):
        # Elements related to gates and their relative pos
        self.start_gate=start_gate
        self.end_gate=end_gate
        self.segment=end_gate.pos-start_gate.pos
        self.norm_segment=np.linalg.norm(self.segment)

    def projectPoint(self, pos: np.array) -> np.array:
        projection=np.dot(pos-self.end_gate.pos, self.segment)
        return projection/self.norm_segment
