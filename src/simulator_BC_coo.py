import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import time
from scipy.sparse import coo_array
from scipy.special import expit as sigmoid
from torch.distributions.beta import Beta
from torch.distributions.bernoulli import Bernoulli

def create_edges(N, edge_per_t, rho, diff_X, epsilon):
    u,v = np.random.choice(N, size = [2, edge_per_t])
    s = (np.random.random(edge_per_t) < sigmoid(rho * (epsilon - np.abs(diff_X[u,v])))) + 0.
    
    return u,v,s
        

def opinion_update(diff_X, X_t, u, v, s, N, mu):
    diff_X_uv = coo_array((diff_X[u, v] * s, (u, v)), shape = (N, N))
    
    new_X_t = X_t.copy()
    
    updates = mu * (diff_X_uv.sum(axis = 0) - diff_X_uv.sum(axis = 1))
    new_X_t += updates
    new_X_t = np.clip(new_X_t, 1e-5, 1-1e-5)
    
    diff_X = new_X_t[:,None] - new_X_t[None,:]
    
    return new_X_t, diff_X

def opinion_update_backfire(diff_X, X_t, u, v, s_plus, s_minus, N, mu_plus, mu_minus):
    diff_X_uv_plus = coo_array((diff_X[u, v] * s_plus, (u, v)), shape = (N, N))
    diff_X_uv_minus = coo_array((diff_X[u, v] * s_minus, (u, v)), shape = (N, N))
    
    new_X_t = X_t.copy()
    
    updates = mu_plus * (diff_X_uv_plus.sum(axis = 0) - diff_X_uv_plus.sum(axis = 1))
    updates -= mu_minus * (diff_X_uv_minus.sum(axis = 0) - diff_X_uv_minus.sum(axis = 1))
    new_X_t += updates
    new_X_t = np.clip(new_X_t, 1e-5, 1-1e-5)
    
    diff_X = new_X_t[:,None] - new_X_t[None,:]
    
    return new_X_t, diff_X


def torch_convert(X, edges):
    X = torch.from_numpy(np.array(X))
    edges = torch.from_numpy(np.array(edges)).permute(0,2,1)
    
    return X, edges

def intialize_simulator(seed, N, X0):
    if seed != None:
        np.random.seed(seed)
    if len(X0) == 0:
        X0 = np.random.random(N)

    edges, X = [], []

    return X0, edges, X

def beta_evidence(sum_ab, X, evidences_per_t, T, N):
    beta_samples = Beta(sum_ab * X + 1e-7, sum_ab * (1 - X) + 1e-7).sample()
    if evidences_per_t != None:
        evidences = [(u, beta_samples[t, u]) for t in range(T) 
                     for u in torch.randint(low = 0, high = N, size = [1, evidences_per_t])]
    else:
        evidences = beta_samples
    return evidences
            
def bernoulli_evidence(X, evidences_per_t, T, N):
    bernoulli_samples = Bernoulli(X).sample()
    if evidences_per_t != None:
        evidences = [(u, bernoulli_samples[t, u]) for t in range(T) 
                     for u in torch.randint(low = 0, high = N, size = [1, evidences_per_t])]
    else:
        evidences = bernoulli_samples
    return evidences
            
    
def simulator_BC_np_coo(N, T, edge_per_t, epsilon, mu, 
                     with_evidences = False, sum_ab = 1, distribution_evidences = "beta",
                     rho = 70, seed = None, X0 = [], evidences_per_t = None, as_torch = True):
    
    X0, edges, X = intialize_simulator(seed, N, X0)
    
    t = 0
    X_t = X0.copy() #all opinions of the users
    diff_X = X0[:,None] - X0[None,:]
    X.append(X_t)

    for t in range(T - 1):
        u,v,s = create_edges(N, edge_per_t, rho, diff_X, epsilon)
        new_X_t, diff_X = opinion_update(diff_X, X_t, u, v, s, N, mu)
        X.append(new_X_t)
        
        X_t = new_X_t.copy()
        edges.append([u,v,s])
        
        
    if as_torch:
        X, edges = torch_convert(X, edges)
        
    if with_evidences:
        if not as_torch:
            X, edges = torch_convert(X, edges)
        if distribution_evidences == "beta":
            return X, edges, beta_evidence(sum_ab, X, evidences_per_t, T, N)
        if distribution_evidences == "bernoulli":
            return X, edges, bernoulli_evidence(X, evidences_per_t, T, N)
    else:
        return X, edges
    edg
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
    uvst = torch.cat((edges.reshape(((max_T) * edge_per_t,3)), torch.Tensor(np.repeat(np.arange(max_T), edge_per_t))[:, None]), dim = 1).T.long()
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
   
    