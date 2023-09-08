import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pybullet as p

from race_rl.path_planing import visualize_points, convert_for_planer
from race_rl.race_track import Track

class Gate:
    def __init__(self, position=(0, 0, 0), orientation=(0, 0, 0)):
        self.position = np.array(position, dtype=float)
        self.orientation = np.array(orientation, dtype=float)
    
    def translate(self, translation):
        self.position += np.array(translation)
    
    def rotate(self, angles_deg):
        # Convert degrees to radians
        angles_rad = (angles_deg)
        self.orientation += angles_rad

    def get_gate_vertices(self):
        # Define the gate's shape (e.g., as a cube)
        # Adjust these coordinates according to the gate's geometry
        vertices = np.array([
            [0 ,-0.25, -0.25],
            [0, 0.25, -0.25],
            [0, 0.25, 0.25],
            [0, -0.25, 0.25],
            [0, -0.25, -0.25],
        ])
        
        # Apply rotation and translation to the vertices
        rotation_matrix = self.get_rotation_matrix()
        translated_vertices = np.dot(vertices, rotation_matrix.T) + self.position
        
        return translated_vertices

    def get_rotation_matrix(self):
        # Compute the rotation matrix based on orientation angles
        roll, pitch, yaw = self.orientation
        rotation_matrix = np.array([
            [np.cos(yaw)*np.cos(pitch), np.cos(yaw)*np.sin(pitch)*np.sin(roll) - np.sin(yaw)*np.cos(roll), np.cos(yaw)*np.sin(pitch)*np.cos(roll) + np.sin(yaw)*np.sin(roll)],
            [np.sin(yaw)*np.cos(pitch), np.sin(yaw)*np.sin(pitch)*np.sin(roll) + np.cos(yaw)*np.cos(roll), np.sin(yaw)*np.sin(pitch)*np.cos(roll) - np.cos(yaw)*np.sin(roll)],
            [-np.sin(pitch), np.cos(pitch)*np.sin(roll), np.cos(pitch)*np.cos(roll)]
        ])
        
        return rotation_matrix



class TrackVis:
    def __init__(self, track_path):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(projection='3d')
        # Set labels for the axes
        self.ax.set_xlabel('X Axis')
        self.ax.set_ylabel('Y Axis')
        self.ax.w_zaxis.line.set_lw(0.)
        self.ax.set_zticks([])
        self.visualize_track(track_path)

    def visualize_track(self, track_path):
        track = Track(track_path, "assets")
        
        for p_gate in track.gates:
            gate = Gate()
            # Translate and rotate the gate
            gate.translate(p_gate.pos)
            gate.rotate(p.getEulerFromQuaternion(p_gate.quat)) 
            # Get the vertices of the gate after applying transformations
            gate_vertices = gate.get_gate_vertices()

            # Plot the gate as a wireframe
            self.ax.plot(gate_vertices[:, 0], gate_vertices[:, 1], gate_vertices[:, 2], c='k', marker='.')
            
    def visualize_trajectory(self, trajectory):
        data = []
        for point in trajectory:
            pos = point.pos
            vel = point.vel
            data.append([*pos, np.linalg.norm(vel)])

        x_coords, y_coords, z_coords, vel_norm = zip(*data)
        self.ax.scatter(x_coords, y_coords, z_coords, c=vel_norm, marker='.')


    def show(self):
        self.ax.set_box_aspect([ub - lb for lb, ub in (getattr(self.ax, f'get_{a}lim')() for a in 'xyz')])
        plt.show()

if __name__ == "__main__":
    from race_rl.path_planing import PathPlanner
    track = TrackVis("assets/tracks/thesis-tracks/long_track.csv")
    T = convert_for_planer("assets/tracks/thesis-tracks/long_track.csv")
    points = np.array(list(map(lambda x: x[0], T)))

    print("Calculating")
    pp = PathPlanner(points, max_velocity=10, kt=100)
    print("Finished")

    tracjetory = []
    for time in np.linspace(0.001, pp.TS[-1], 1000):
        state = pp.getStateAtTime(time)
        tracjetory.append(state)

    track.visualize_trajectory(tracjetory)


    track.show()
