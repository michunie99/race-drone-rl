import numpy as np
from numpy import linalg as LA
from scipy import optimize

from collections import namedtuple

DesiredState = namedtuple('DesiredState', 'pos vel acc jerk yaw yawdot')
        
class PathPlanner():
    def __init__(self, waypoint: np.array, max_velocity: float, kt = 100):
        self.waypoint = waypoint
        self.max_velocity = max_velocity
        self.kt = kt
        # Order of the polynomial between 2 points
        self.order = 10
        n, dim = waypoint.shape
        self.len = n
        self.dim = dim
        self.TS = np.zeros(self.len)

        # Optimize track
        self.optimize()
        self.yaw = 0 
        self.heading = np.zeros(2)

    def optimize(self):
        relative = self.waypoint[0:-1] - self.waypoint[1:]
        T_init = LA.norm(relative, axis=-1) / self.max_velocity

        # Why like this  ????
        T = optimize.minimize(self.getCost, T_init, method="COBYLA", constraints= ({'type': 'ineq', 'fun': lambda T: T-T_init}))['x']
        self.TS[1:] = np.cumsum(T)
        self.cost, self.coef = self.minSnapTrajectory(T)


    def getCost(self, T):
        cost, _ = self.minSnapTrajectory(T)
        cost += self.kt * np.sum(T)
        return cost

    def minSnapTrajectory(self, T):
        unkns = 4*(self.len - 2)

        Q = Hessian(T)
        A,B = self.getConstrains(T)

        invA = LA.inv(A)

        if unkns != 0:
            R = invA.T@Q@invA

            Rfp = R[:-unkns,-unkns:]
            Rpp = R[-unkns:,-unkns:]

            B[-unkns:,] = -LA.inv(Rpp)@Rfp.T@B[:-unkns,]

        P = invA@B
        cost = np.trace(P.T@Q@P)

        return cost, P

    def getConstrains(self, T):
        n = self.len-1
        order = self.order

        A = np.zeros((n*order, n*order))
        B = np.zeros((n*order, self.dim))

        B[:n,:] = self.waypoint[:-1, :]
        B[n:n*2,:] = self.waypoint[1:, :]

        # Way point constraints
        for i in range(n):
            A[i, order*i: order*(i+1)] = polyder(0)
            A[i+n, order*i: order*(i+1)] = polyder(T[i])

        #continuity contraints
        for i in range(n-1):
            A[2*n + 4*i: 2*n + 4*(i+1), order*i : order*(i+1)] = -polyder(T[i],'all')
            A[2*n + 4*i: 2*n + 4*(i+1), order*(i+1) : order*(i+2)] = polyder(0,'all')

        #start and end at rest
        A[6*n - 4 : 6*n, : order] = polyder(0,'all')
        A[6*n : 6*n + 4, -order : ] = polyder(T[-1],'all')

        #free variables
        for i in range(1,n):
            A[6*n + 4*i : 6*n + 4*(i+1), order*i : order*(i+1)] = polyder(0,'all')

        return A,B

    def getStateAtTime(self, t):
        if t >= self.TS[-1]: t = self.TS[-1] - 0.001

        i = np.where(t >= self.TS)[0][-1]

        t = t - self.TS[i]
        coeff = (self.coef.T)[:,self.order*i:self.order*(i+1)]

        pos  = coeff@polyder(t)
        vel  = coeff@polyder(t,1)
        accl = coeff@polyder(t,2)
        jerk = coeff@polyder(t,3)

        #set yaw in the direction of velocity
        yaw, yawdot = self.getYaw(vel[:2])

        return DesiredState(pos, vel, accl, jerk, yaw, yawdot)


    def getYaw(self, vel):
        curr_heading = vel/LA.norm(vel)
        prev_heading = self.heading
        cosine = max(-1,min(np.dot(prev_heading, curr_heading),1))
        dyaw = np.arccos(cosine)
        norm_v = np.cross(prev_heading,curr_heading)
        self.yaw += np.sign(norm_v)*dyaw

        if self.yaw > np.pi: self.yaw -= 2*np.pi
        if self.yaw < -np.pi: self.yaw += 2*np.pi

        self.heading = curr_heading
        yawdot = max(-30,min(dyaw/0.005,30))
        return self.yaw,yawdot


def Hessian(T, order=10, opt=4):
    n = len(T)
    Q = np.zeros((n*order, n*order))
    for s in range(n):
        m = np.arange(0, opt, 1)
        for ii in range(order):
            for jj in range(order):
                if ii >= opt and jj >= opt:
                    pow = ii+jj-2*opt+1
                    # TODO: how to claculate a hessian
                    Q[order*s+ii,order*s+jj] = 2*np.prod((ii-m)*(jj-m))*T[s]**pow/pow

    return Q

def polyder(t, k = 0, order = 10):
    if k == 'all':
        terms = np.array([polyder(t,k,order) for k in range(1,5)])
    else:
        terms = np.zeros(order)
        coeffs = np.polyder([1]*order,k)[::-1]
        pows = t**np.arange(0,order-k,1)
        terms[k:] = coeffs*pows
    return terms

def collision_aviodance(self,):
    # TODO - add colision with gate deterion
    pass

if __name__ == "__main__":
    from utils import convert_for_planer, visualize_points
    T = convert_for_planer("assets/tracks/thesis-tracks/straight_track.csv")
    points = np.array(list(map(lambda x: x[0], T)))
    fig, ax = visualize_points(T)
    import matplotlib.pyplot as plt

    print("Calculating")
    pp = PathPlanner(points, max_velocity=10, kt=100)
    print("Finished")

    points = []
    for time in np.linspace(0.001, pp.TS[-1], 1000):
        state = pp.getStateAtTime(time)
        points.append((state.pos))
    
    x_coords, y_coords, z_coords = zip(*points)


    # plot the points as scatter
    ax.scatter(x_coords, y_coords, z_coords, c='g', marker='.')
    plt.show() 

