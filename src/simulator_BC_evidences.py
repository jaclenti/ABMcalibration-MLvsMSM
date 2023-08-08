import torch
import numpy as np
import pandas as pd
from time import time
import seaborn as sns
import matplotlib.pyplot as plt
from torch.distributions.beta import Beta

def simulator_BC_evidence(N, T, edge_per_t, epsilon, mu, 
                          with_evidences = False, sum_ab = 1,
                          rho = 70, seed = None, X0 = [], evidences_per_t = None):
    if seed != None:
        torch.manual_seed(seed)

    if len(X0) == 0:
        X0 = torch.rand(N)

    edges = torch.empty([0, edge_per_t, 3], dtype = torch.long)
    evidences = torch.empty([0, N])

    t = 0
    X = X0.clone()[None,:] #all opinions of the users


    for t in range(T - 1):
        X_t = X[-1, :].clone()
        u,v = torch.randint(low = 0, high = N, size = [2, edge_per_t])

        diff_X = X_t[u] - X_t[v]

        s_t = torch.rand(edge_per_t) < torch.sigmoid(rho * (epsilon - torch.abs(diff_X)))

        X_t[u[s_t]] -= mu * diff_X[s_t]
        X_t[v[s_t]] += mu * diff_X[s_t]

        X = torch.cat([X, X_t[None, :]], dim = 0)

        edges_t = torch.cat([u.long()[:, None], v.long()[:, None], s_t.long()[:, None]], dim = 1)
        edges = torch.cat([edges,  edges_t[None, :, :]])
    
    X = torch.clamp(X, min = 0., max = 1.)
    
    if with_evidences:
        evidences = Beta(sum_ab * X + 1e-7, sum_ab * (1 - X) + 1e-7).sample()
        if evidences_per_t != None:
            sampled_evidences = [(u, evidences[t, u]) for t in range(T) for u in torch.randint(low = 0, high = N, size = [1, evidences_per_t])]
            return X, edges, sampled_evidences
        else:
            return X, edges, evidences
    else:
        return X, edges
    
    
#given a tensor X, of size [h,k], return a tensor diff_X of size [h,k,k], such that diff_X[i1, i2, i3] = X[i1, i2] - X[i1, i3]
#it is used to store all the opinion distances between interacting nodes
def differences_tensor(X, N, T):
    diff_X = ((X[:,:,None] * torch.ones([1,N,N])) - (X[:,None,:] * torch.ones([1,N,N]))) #matrix of the opinion differences
    return diff_X

#create a plot with all the opinion time trajectories of all users
def print_opinion_trajectory(X, figsize = (5,5)):
    fig, ax = plt.subplots(figsize = figsize)
    ax.plot(np.matrix(X))
    
    fig.show()
    
#convert the edges tensor
#input edges, such that edges[t][0] = u,v,s
#output u_v_t_s, such that edges[0] = u,v,t,s
def convert_edges_uvst(edges):
    max_T, edge_per_t, _ = edges.size()
    uvst = torch.cat((edges.view(((max_T) * edge_per_t,3)), torch.Tensor(np.repeat(np.arange(max_T), edge_per_t))[:, None]), dim = 1).T.long()
    return uvst

def convert_uvst_edges(uvst, T):
    edge_per_t = int(uvst.shape[0] / (T - 1))
    edges = uvst[:,None,[0,1,2]].reshape([(T - 1), edge_per_t, 3])
    return edges

#max_d = max distance at which two nodes interacted
#min_d = min distance at which two nodes did not interact
def max_min_interaction_distance(edges, X):
    T, N = X.size()
    diff_X = differences_tensor(X, N, T)
    u,v,s,t = convert_edges_uvst(edges)
    
    return max(torch.abs(diff_X[t,u,v])[s == 1]).item(), min(torch.abs(diff_X[t,u,v])[s == 0]).item() 
   