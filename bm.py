from process import *
import numpy as np


class BrownianMotion(StochasticProcess):

    '''A class representing standard Brownian Motion in Euclidean space'''

    def __init__(self, dim):
        super(BrownianMotion, self).__init__(dim)

    def sample(self, tA=0, tB=1, n_sample_points=1000):
        dt = (tB-tA)/(n_sample_points-1)

        G = np.sqrt(dt)*np.random.randn(n_sample_points-1, self.dim)
        X = np.zeros((n_sample_points, self.dim))
        X[1:] = np.cumsum(G, axis=0)
        X += np.sqrt(tA)

        self.sample_path = X
        return X


class BrownianBridge(StochasticProcess):
    '''A Brownian bridge, a real-valued process whose law is the Wiener distribution,
        conditional on having values A and B at times tA and tB.'''

    def __init__(self, tA=0, tB=1, A=0, B=0):
        super(BrownianBridge, self).__init__(1)
        self.tA = tA
        self.tB = tB
        self.A = A
        self.B = B
        self.W = BrownianMotion(1)

    def sample(self, n_sample_points=1000):
        Tspan = (self.tB-self.tA)
        W_traj = self.W.sample(self.tA, self.tB, n_sample_points)
        times = np.arange(0, n_sample_points, step=1)
        X = (self.tB-times)*(self.A+W_traj[times] -
                             W_traj[0])/Tspan+(times-self.tA)*(self.B+W_traj[-1]-W_traj[times])/Tspan
        self.sample_path = X
        return X


class BrownianInterpolation(StochasticProcess):

    '''A multidimensional Brownian motion conditioned on having values X at times T. 
    This generalizes the Brownian bridge to multiple waypoints and dimensions.'''

    def __init__(self, dim, T, X):
        super(BrownianInterpolation, self).__init__(dim)
        self.T = T
        self.X = X
        self.W = BrownianMotion(dim)

    def sample(self, tA=None, tB=None, n_sample_points=1000):
        if tA is None:
            tA = self.T[0]
        if tB is None:
            tB = self.T[-1]

        dt = (tB-tA)/n_sample_points

        times = np.linspace(tA, tB, n_sample_points)
        W_sample = self.W.sample(tA, tB, n_sample_points)
        X = W_sample.copy()

        for i in range(self.T.size-1):

            if self.T[i]*(n_sample_points-1) % (tB-tA):
                k = int(self.T[i]*(n_sample_points-1)/(tB-tA))
                # if W is not sampled at time T[i], interpolate the W sample to get a sample value for W(T[i])
                W_Ti = W_sample[k] + (W_sample[k+1]-W_sample[k])*(self.T[i]-times[k])/dt\
                    + np.sqrt((times[k+1]-self.T[i]) *
                              (self.T[i]-times[k])/dt)*np.random.randn()
                # this is a brownian bridge interpolation
            else:
                W_Ti = W_sample[int(self.T[i]*(n_sample_points-1)/(tB-tA))]

            if self.T[i+1]*(n_sample_points-1) % (tB-tA):
                k = int(self.T[i+1]*(n_sample_points-1)/(tB-tA))
                # repeat for T[i+1]
                W_Ti_1 = W_sample[k] + (W_sample[k+1]-W_sample[k])*(self.T[i+1]-times[k])/dt\
                    + np.sqrt((times[k+1]-self.T[i+1]) *
                              (self.T[i+1]-times[k])/dt)*np.random.randn()
            else:
                W_Ti_1 = W_sample[int(self.T[i+1]*(n_sample_points-1)/(tB-tA))]

            time_slice = times[(self.T[i] < times)&(times <= self.T[i+1])]
            X[(self.T[i] < times)&( times<= self.T[i+1])] += (time_slice-self.T[i])/dt*(
                self.X[i]-W_Ti)+(self.T[i+1]-time_slice)/dt*(self.X[i+1]-W_Ti_1)

        if self.T[-1] < tB:
            if self.T[-1]*(n_sample_points-1) % (tB-tA):
                k = int(self.T[-1]*(n_sample_points-1)/(tB-tA))
                # repeat for T[-1]
                W_T_last = W_sample[k] + (W_sample[k+1]-W_sample[k])*(self.T[-1]-times[k])/dt\
                    + np.sqrt((times[k+1]-self.T[i+1]) *
                              (self.T[-1]-times[k])/dt)*np.random.randn()
            else:
                W_T_last = W_sample[int(self.T[-1]*(n_sample_points-1)/(tB-tA))]

            X[self.T[-1] < times] += self.X[-1]-W_T_last

        self.sample_path = X

        return X
