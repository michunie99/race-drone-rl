from pathlib import Path
import csv
from typing import Union
from collections import deque

import pybullet as p
import numpy as np


def calculateRelativeObseration(obj1, obj2):
    """
    obj = ([x, y, z], [qx, qy, qz, qw])
    """
    # TODO - keep info about quaterions and not shperical
    pos1, ort1 = obj1
    pos2, ort2 = obj2

    # Step 1 - Vector between two points
    vec_diff = pos2 - pos1
    quat_diff = p.getDifferenceQuaternion(ort1, ort2)
    inv_p, inv_o = p.invertTransform([0,0,0], ort1)
    rot_vec, _ = p.multiplyTransforms(inv_p, inv_o,
                               vec_diff, [0, 0, 0, 1])
    # Step 2 - calculate shperical coordinates
    r, theta, phi = cart2shp(rot_vec)
    # Step 3 - calculate angle between normals
    _, alpha = p.getAxisAngleFromQuaternion(quat_diff)
    return np.array([r, theta, phi, alpha])
    # return np.array([np.linalg.norm(vec_diff), *quat_diff]).astype('float64')


def cart2shp(cart):
    xy = np.sqrt(cart[0]**2 + cart[1]**2)
    r = np.sqrt(xy**2 + cart[2]**2)
    theta = np.arctan2(cart[1], cart[0]) # for elevation angle defined from Z-axis down
    phi = np.arctan2(xy, cart[2])
    return r, theta, phi
