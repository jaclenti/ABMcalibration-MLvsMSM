#In this script we define the simulators of the bounded confidence model we use in the calibration with MSM.
#A simulator receive data and available information of the model, and is able to produce a trajectory of the agent based model with the given parameter of input.
#In the calibration phase, we produce traces of the ABM with different choices of parameters (epsilon), and compare the time series of the summary statistics (s and evidences) between observed and simulated time series.



import numpy as np
import sys
sys.path += ["../src"]
import torch
import simulator_opinion_dynamics as sod
from scipy.special import expit as sigmoid





########################### Fully Observed BCM #################################

# All the edges and opinions are observed
# So the time series can simulate the outcomes of the interactions at each timestep, always knowing the opinion and the interacting nodes

class FBCM_simulator():
    def __init__(self, X0, edges, N, mu, real_epsilon, rho = 16, seed = 1):
        self.X0 = X0
        self.N = N
        self.edges = edges.clone().long()
        _, self.edge_per_t, _ = edges.shape
        self.mu = mu
        self.real_epsilon = real_epsilon
        self.rho = rho
        self.seed = seed
        # self.ts is the time series of the proportion of signs s (the 3rd column of edges)
        # the calibrator will compare the real ts with the simulated ts
        self.ts = np.atleast_2d(edges[:,:,2].sum(dim = 1) / self.edge_per_t).T
    
    def simulate_ts(self, theta, N = 200, seed = None):
        epsilon, = theta
        mean_s_pred = []
        
        
        if seed != None:
            np.random.seed(self.seed)
        
        X_t = self.X0.clone()
        
        diff_X = self.X0[:,None] - self.X0[None,:]
        
        T, edge_per_t, _ = self.edges.size()
        
        for t in range(T):
            u,v,s_obs = self.edges[t].T
            
            s_pred = (torch.rand(self.edge_per_t) < torch.sigmoid(self.rho * (epsilon - torch.abs(diff_X[u,v])))) + 0.
            mean_s_pred.append(s_pred.mean())
            
            X_t, diff_X = sod.opinion_update_BC(diff_X, X_t, self.edges[t], self.N, (epsilon, self.mu, self.rho))
            
        return np.atleast_2d(mean_s_pred).T
    


########################### Partially Observed BCM #################################

# Partially observed BCM. In this case we observe all the opinions, and only a portion of the interacting nodes.
# The simulator simulates the outcomes of the interactions (s), knowing opiinos and a fraction of the interacting nodes. For the rest of the interactions, it samples some pairs of nodes and simulate the interactions between them

class PBCM_simulator():
    def __init__(self, X0, edges, N, mu, real_epsilon, rho = 16, seed = 1, sample_all_edges = True):
        self.X0 = X0
        self.N = N
        self.edges = edges.clone().long()
        T_, self.edge_per_t, _ = edges.shape
        self.T = T_ + 1
        self.mu = mu
        self.real_epsilon = real_epsilon
        self.rho = rho
        self.seed = seed
        self.ts = np.atleast_2d(edges[:,:,2].sum(dim = 1) / self.edge_per_t).T
        self.s_edges = edges[:,:,2]
        
        self.sample_all_edges = sample_all_edges
    
    def simulate_ts(self, theta, N = 200, seed = None):
        epsilon, = theta
        mean_s_pred = []
        
        
        if seed != None:
            np.random.seed(self.seed)
        
        X_t = self.X0.clone()
        diff_X = self.X0[:,None] - self.X0[None,:]
        
        
        for t in range(self.T - 1):
            edges_t = self.edges[t]
            random_edges_t = torch.randint(low = 0, high = self.N, size = (self.edge_per_t, 2))#, dtype = torch.float32)
            mask = edges_t[:,2] == 0
            edges_t[mask, :2] = random_edges_t[mask]
            
            u,v,s_obs = edges_t.T
            
            if self.sample_all_edges:
                u,v = random_edges_t.T
            else:
                u,v = edges_t.T[:2,:]

            s_pred = (torch.rand(self.edge_per_t) < torch.sigmoid(self.rho * (epsilon - torch.abs(diff_X[u,v])))) + 0.
            mean_s_pred.append(s_pred.mean().item())
            
            X_t, diff_X = sod.opinion_update_BC(diff_X, X_t, self.edges[t], self.N, (epsilon, self.mu, self.rho))
            
        return np.atleast_2d(mean_s_pred).T

    



########################### Noisy Observed BCM #################################

# in this scenario, the interactions are observed, while the opinions are latent. Only some evidences on the opinons are available.
# the simulator has to simulate the initial opinions (X0) and simulate the interactions between nodes 
# finally it compares the signs of the interactions (s), but also the evidences obtained from simulated and observed time series (only mean and variance)

class NBCM_simulator():
    def __init__(self, N, edges, evidences, mu, real_epsilon, evidence_distribution = "bernoulli", 
                 sum_ab = 1, rho = 16, seed = 1):
        T, edge_per_t, _ = edges.shape
        self.evidence_distribution = evidence_distribution
        self.evidences_per_t = len(evidences[0][0])
        self.T = T
        self.edge_per_t = edge_per_t
        self.N = N
        self.evidences = evidences
        self.edges = edges.clone().long()
        self.sum_ab = sum_ab
        self.mu = mu
        self.real_epsilon = real_epsilon
        self.rho = rho
        self.seed = seed
        #the time series used in the comparison takes into account the signs, the mean and the variances of the evidences (proxies of the opinions)
        self.ts = np.atleast_2d([(torch.mean(evidences[t][1]).item(),
                                  torch.var(evidences[t][1]).item(),
                                  edges[t,:,2].sum() / edge_per_t) for t in range(T)])
    
    def simulate_ts(self, theta, N = 200, seed = None):
        epsilon, = theta
        
        mean_ev_pred, var_ev_pred, mean_s_pred = [], [], []
        
        if seed != None:
            np.random.seed(self.seed)
        
        X0 = torch.rand(self.N)
        X_t = X0.clone()
        diff_X = X0[:,None] - X0[None,:]
        
        for t in range(self.T):
            u,v,s_obs = self.edges[t].T
            
            s_pred = (torch.rand(self.edge_per_t) < torch.sigmoid(self.rho * (epsilon - torch.abs(diff_X[u,v])))) + 0.
            X_t, diff_X = sod.opinion_update_BC(diff_X, X_t, self.edges[t], self.N, (epsilon, self.mu, self.rho))
            
            u_evidences = self.evidences[t][0]
            
            if self.evidence_distribution == "beta":
                a,b = self.sum_ab * (X_t[u_evidences]), self.sum_ab * (1 - (X_t[u_evidences]))
                evidences_t = Beta(torch.Tensor(a), torch.Tensor(b)).sample()
                mean_ev_pred.append(torch.mean(evidences_t).item())
                var_ev_pred.append(torch.var(evidences_t).item())
            
            if self.evidence_distribution == "bernoulli":
                evidences_t = (torch.rand(self.evidences_per_t) < X_t[u_evidences]) + 0.
                mean_ev_pred.append(torch.mean(evidences_t))
                var_ev_pred.append(torch.var(evidences_t))
            
            
            mean_s_pred.append(s_pred.mean())
        
        return np.atleast_2d([mean_ev_pred, var_ev_pred, mean_s_pred]).T







