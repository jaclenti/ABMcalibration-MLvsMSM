import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path += ["../src"]
from calibrator_blackit import Calibrator #i just commented few lines of code to remove the output print
#from black_it.calibrator import Calibrator
import pandas as pd
from typing import Sequence
import os
import contextlib
import torch
from tqdm import tqdm
from time import time
import simulator_opinion_dynamics as sod
import simulator_BC_coo as sim_coo
from scipy.special import expit as sigmoid


class BC_simulator():
    def __init__(self, X0, edges, N, mu, real_epsilon, rho = 70, seed = 1):
        self.X0 = np.array(X0).copy()
        self.N = N
        self.edges = edges.clone().long()
        _, edge_per_t, _ = edges.shape
        self.mu = mu
        self.real_epsilon = real_epsilon
        self.rho = rho
        self.seed = seed
        self.ts = np.atleast_2d(edges[:,:,2].sum(dim = 1) / edge_per_t).T
    
    def simulate_ts(self, theta, N = 200, seed = None):
        epsilon, = theta
        mean_s_pred = []
        
        
        if seed != None:
            np.random.seed(self.seed)
        
        X_t = self.X0.copy()
        diff_X = self.X0[:,None] - self.X0[None,:]
        
        T, edge_per_t, _ = self.edges.size()
        
        for t in range(T):
            u,v,s_obs = self.edges[t].T
            
            s_pred = (np.random.random(edge_per_t) < sigmoid(self.rho * (epsilon - np.abs(diff_X[u,v])))) + 0.
            new_X_t, diff_X = sim_coo.opinion_update(diff_X, X_t, u, v, s_pred, self.N, self.mu)
            
            X_t = new_X_t.copy()
            
            mean_s_pred.append(s_pred.mean())
            
            X_t  = np.clip(X_t, 10e-5, 1 - 10e-5)
        return np.atleast_2d(mean_s_pred).T

    
class BC_simulator_positive_observations():
    def __init__(self, X0, edges, N, mu, real_epsilon, rho = 70, seed = 1, sample_all_edges = True):
        self.X0 = X0 #np.array(X0).copy()
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
        
        #X_t = self.X0.copy()
        X_t = self.X0.clone()
        diff_X = self.X0[:,None] - self.X0[None,:]
        
        
        for t in range(self.T - 1):
            edges_t = self.edges[t]
            random_edges_t = torch.randint(low = 0, high = self.N, size = (self.edge_per_t, 2))#, dtype = torch.float32)
            mask = edges_t[:,2] == 0
            edges_t[mask, :2] = random_edges_t[mask]
            
            u,v,s_obs = edges_t.T
            
            #s_pred = (np.random.random(self.edge_per_t) < sigmoid(self.rho * (epsilon - np.abs(diff_X[u,v])))) + 0.
            if self.sample_all_edges:
                u,v = random_edges_t.T
            else:
                u,v = edges_t.T[:2,:]

            s_pred = (torch.rand(self.edge_per_t) < torch.sigmoid(self.rho * (epsilon - torch.abs(diff_X[u,v])))) + 0.
            
            
            
            X_t, diff_X = sod.opinion_update_BC(diff_X, X_t, self.edges[t], self.N, (epsilon, self.mu, self.rho))
            #new_X_t, diff_X = sim_coo.opinion_update(diff_X, X_t, u, v, np.array(s_obs), self.N, self.mu)
            
            #X_t = new_X_t.copy()
            
            mean_s_pred.append(s_pred.mean().item())
            
            #X_t  = np.clip(X_t, 10e-5, 1 - 10e-5)
        return np.atleast_2d(mean_s_pred).T

    


    
class BC_simulator_hidden_mu():
    def __init__(self, X0, edges, N, real_mu, real_epsilon, rho = 70, seed = 1):
        self.X0 = np.array(X0).copy()
        self.N = N
        self.edges = edges.clone().long()
        _, edge_per_t, _ = edges.shape
        self.real_mu = real_mu
        self.real_epsilon = real_epsilon
        self.rho = rho
        self.seed = seed
        self.ts = np.atleast_2d(edges[:,:,2].sum(dim = 1) / edge_per_t).T
    
    def simulate_ts(self, theta, N = 200, seed = None):
        epsilon, mu = theta
        mean_s_pred = []
        
        
        if seed != None:
            np.random.seed(self.seed)
        
        X_t = self.X0.copy()
        diff_X = self.X0[:,None] - self.X0[None,:]
        
        T, edge_per_t, _ = self.edges.size()
        
        for t in range(T):
            u,v,s_obs = self.edges[t].T
            
            s_pred = (np.random.random(edge_per_t) < sigmoid(self.rho * (epsilon - np.abs(diff_X[u,v])))) + 0.
            new_X_t, diff_X = sim_coo.opinion_update(diff_X, X_t, u, v, s_pred, self.N, mu)
            
            X_t = new_X_t.copy()
            
            mean_s_pred.append(s_pred.mean())
            
            X_t  = np.clip(X_t, 10e-5, 1 - 10e-5)
        return np.atleast_2d(mean_s_pred).T

class BC_simulator_hidden_mu_X0():
    def __init__(self, edges, N, real_mu, real_epsilon, rho = 70, seed = 1):
        self.N = N
        self.edges = edges.clone().long()
        _, edge_per_t, _ = edges.shape
        self.real_mu = real_mu
        self.real_epsilon = real_epsilon
        self.rho = rho
        self.seed = seed
        self.ts = np.atleast_2d(edges[:,:,2].sum(dim = 1) / edge_per_t).T
    
    def simulate_ts(self, theta, N = 200, seed = None):
        epsilon, mu = theta
        X = torch.rand(self.N)
        
        mean_s_pred = []
        
        
        if seed != None:
            np.random.seed(self.seed)
        
        X0, _, _ = sim_coo.intialize_simulator(None, self.N, [])
        X_t = X0.copy()
        diff_X = X0[:,None] - X0[None,:]
        
        T, edge_per_t, _ = self.edges.size()
        
        for t in range(T):
            u,v,s_obs = self.edges[t].T
            
            s_pred = (np.random.random(edge_per_t) < sigmoid(self.rho * (epsilon - np.abs(diff_X[u,v])))) + 0.
            new_X_t, diff_X = sim_coo.opinion_update(diff_X, X_t, u, v, s_pred, self.N, mu)
            
            X_t = new_X_t.copy()
            
            mean_s_pred.append(s_pred.mean())
            
            X_t  = np.clip(X_t, 10e-5, 1 - 10e-5)
        return np.atleast_2d(mean_s_pred).T

    
class BC_simulator_X_evidences():
    def __init__(self, N, edges, evidences, mu, real_epsilon, evidence_distribution = "bernoulli", sum_ab = 1, rho = 70, seed = 1):
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
        #self.mean_s = np.atleast_2d(edges[:,:,2].sum(dim = 1) / edge_per_t).T
        self.ts = np.atleast_2d([(torch.mean(evidences[t][1]).item(),
                                  torch.var(evidences[t][1]).item(),
                                  edges[t,:,2].sum() / edge_per_t) for t in range(T)])
    
    def simulate_ts(self, theta, N = 200, seed = None):
        epsilon, = theta
        
        mean_ev_pred, var_ev_pred, mean_s_pred = [], [], []
        
        if seed != None:
            np.random.seed(self.seed)
        
        X0, _, _ = sim_coo.intialize_simulator(None, self.N, [])
        X_t = X0.copy()
        diff_X = X0[:,None] - X0[None,:]
        
        for t in range(self.T):
            u,v,s_obs = self.edges[t].T
            
            s_pred = (np.random.random(self.edge_per_t) < sigmoid(self.rho * (epsilon - np.abs(diff_X[u,v])))) + 0.
            new_X_t, diff_X = sim_coo.opinion_update(diff_X, X_t, u, v, s_pred, self.N, self.mu)
            
            X_t = new_X_t.copy()
            
            u_evidences = self.evidences[t][0]
            
            if self.evidence_distribution == "beta":
                a,b = self.sum_ab * (X_t[u_evidences]), self.sum_ab * (1 - (X_t[u_evidences]))
                evidences_t = Beta(torch.Tensor(a), torch.Tensor(b)).sample()
                mean_ev_pred.append(torch.mean(evidences_t).item())
                var_ev_pred.append(torch.var(evidences_t).item())
            
            if self.evidence_distribution == "bernoulli":
                evidences_t = np.random.random(self.evidences_per_t) < X_t[u_evidences]
                mean_ev_pred.append(np.mean(evidences_t))
                var_ev_pred.append(np.var(evidences_t))
            
            
            mean_s_pred.append(s_pred.mean())
            
            X_t  = np.clip(X_t, 10e-5, 1 - 10e-5)
        
        return np.atleast_2d([mean_ev_pred, var_ev_pred, mean_s_pred]).T

    
class BC_simulator_X_evidences_backfire():
    def __init__(self, N, edges, evidences, mu_plus, mu_minus, real_epsilon_plus, real_epsilon_minus, evidence_distribution = "bernoulli", sum_ab = 1, rho = 70, seed = 1):
        T, edge_per_t, _ = edges.shape
        self.evidence_distribution = evidence_distribution
        self.evidences_per_t = len(evidences[0][0])
        self.T = T
        self.edge_per_t = edge_per_t
        self.N = N
        self.evidences = evidences
        self.edges = edges.clone().long()
        self.sum_ab = sum_ab
        self.mu_plus = mu_plus
        self.mu_minus = mu_minus
        self.real_epsilon_plus = real_epsilon_plus
        self.real_epsilon_minus = real_epsilon_minus
        self.rho = rho
        self.seed = seed
        #self.mean_s = np.atleast_2d(edges[:,:,2].sum(dim = 1) / edge_per_t).T
        self.ts = np.atleast_2d([(torch.mean(evidences[t][1]).item(),
                                  torch.var(evidences[t][1]).item(),
                                  edges[t,:,2].sum() / edge_per_t,
                                  edges[t,:,3].sum() / edge_per_t
                                 ) for t in range(T)])
    
    def simulate_ts(self, theta, N = 200, seed = None):
        epsilon_plus,epsilon_minus, = sorted(theta)
        
        mean_ev_pred, var_ev_pred, mean_s_plus_pred, mean_s_minus_pred = [], [], [], []
        
        if seed != None:
            np.random.seed(self.seed)
        
        X0, _, _ = sim_coo.intialize_simulator(None, self.N, [])
        X_t = X0.copy()
        diff_X = X0[:,None] - X0[None,:]
        
        for t in range(self.T):
            u,v,s_plus_obs,s_plus_minus = self.edges[t].T
            
            s_plus_pred = (np.random.random(self.edge_per_t) < sigmoid(self.rho * (epsilon_plus - np.abs(diff_X[u,v])))) + 0.
            s_minus_pred = (np.random.random(self.edge_per_t) < sigmoid(-self.rho * (epsilon_minus - np.abs(diff_X[u,v])))) + 0.
            new_X_t, diff_X = sim_coo.opinion_update_backfire(diff_X, X_t, u, v, s_plus_pred, s_minus_pred, self.N, self.mu_plus, self.mu_minus)
            
            X_t = new_X_t.copy()
            
            u_evidences = self.evidences[t][0]
            
            if self.evidence_distribution == "beta":
                a,b = self.sum_ab * (X_t[u_evidences]), self.sum_ab * (1 - (X_t[u_evidences]))
                evidences_t = Beta(torch.Tensor(a), torch.Tensor(b)).sample()
                mean_ev_pred.append(torch.mean(evidences_t).item())
                var_ev_pred.append(torch.var(evidences_t).item())
            
            if self.evidence_distribution == "bernoulli":
                evidences_t = np.random.random(self.evidences_per_t) < X_t[u_evidences]
                mean_ev_pred.append(np.mean(evidences_t))
                var_ev_pred.append(np.var(evidences_t))
            
            
            mean_s_plus_pred.append(s_plus_pred.mean())
            mean_s_minus_pred.append(s_minus_pred.mean())
            
            X_t  = np.clip(X_t, 10e-5, 1 - 10e-5)
        
        return np.atleast_2d([mean_ev_pred, var_ev_pred, mean_s_plus_pred, mean_s_minus_pred]).T
