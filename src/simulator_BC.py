import sys
import torch
import random
import numpy as np
import matplotlib.pyplot as plt

#create opinion matrix X
#X[i,j] = opinion of user j at time t
#create edges torch.tensor of a bounded confidence model
#edges[t] = [(u,v,s)_1, (u,v,s)_2,..., (u,v,s)_edge_per_t] 
#u and v are the users, t is the time, and s = 0 if the interaction did not occurr abs(xu - xv) > epsilon, s = 1 if the interaction occurred abs(xu - xv) < epsilon
def simulator(N, T, edge_per_t, epsilon, mu, seed = None, x0 = []):
    if seed != None:
        np.random.seed(seed)
    if len(x0) == 0:
        x0 = np.random.uniform(size = [N])
    
    edges = torch.zeros(T - 1, edge_per_t, 3)
    t = 0
    X = [x0] #all opinions of the users
    for t in range(T - 1):
        xt = X[-1]
        """
        ### THIS BLOCK IS NOT CORRECT
        # Now we extract u and v with replacement
        """
        interacting_nodes = []
        for new_edge in range(edge_per_t):

            xt_cp = xt.copy()

            
            
            while True:
                u = np.random.randint(N) #pick a user
                if u not in interacting_nodes:
                    interacting_nodes.append(u)
                    break
            while True:
                v = np.random.randint(N)
                if v not in interacting_nodes:
                    interacting_nodes.append(v)
                    break
        
            dist = np.abs(xt[u] - xt[v])

            if dist < epsilon:
                xt_cp[v] += mu * (xt[u] - xt[v])
                xt_cp[u] += mu * (xt[v] - xt[u])
                edges[t,new_edge] = torch.Tensor([u, v, 1])
                
            else:
                edges[t,new_edge] = torch.Tensor([u, v, 0])
                            
            xt = xt_cp

        X.append(xt_cp)

    X = torch.from_numpy(np.vstack(X))
    return X, edges.long()

#define also a stochastic BC model, where P(s = 1|xu, xv) = sigmoid_rho(epsilon - abs(xu - xv))
def simulator_stoch(N, T, edge_per_t, epsilon, mu, steepness = 10, seed = None, x0 = []):
    if seed != None:
        np.random.seed(seed)
    if len(x0) == 0:
        x0 = np.random.uniform(size = [N])
    
    edges = torch.zeros(T - 1, edge_per_t, 3)
    t = 0
    X = [x0] #all opinions of the users
    for t in range(T - 1):
        xt = X[-1]
        """
        interacting_nodes = []
        for new_edge in range(edge_per_t):
            #xt = X[-1] # opinion of the users at previous time

            xt_cp = xt.copy()

            while True:
                u = np.random.randint(N) #pick a user
                if u not in interacting_nodes:
                    interacting_nodes.append(u)
                    break
            while True:
                v = np.random.randint(N)
                if v not in interacting_nodes:
                    interacting_nodes.append(v)
                    break
            dist = np.abs(xt[u] - xt[v])

            if torch.rand([]) < torch.sigmoid(torch.Tensor([steepness * (epsilon - dist)])):
                xt_cp[u] += mu * (xt[v] - xt[u])
                xt_cp[v] += mu * (xt[u] - xt[v])
                edges[t,new_edge] = torch.Tensor([u, v, 1])
                
            else:
                edges[t,new_edge] = torch.Tensor([u, v, 0])
                
            xt = xt_cp

        X.append(xt_cp)
        """
        
        for new_edge in range(edge_per_t):
            xt_cp = xt.copy()
            u,v = np.random.choice(np.arange(N), size = 2, replace = False)
            
            dist = np.abs(xt[u] - xt[v])
            
            if torch.rand([]) < torch.sigmoid(torch.Tensor([steepness * (epsilon - dist)])):
                xt_cp[u] += mu * (xt[v] - xt[u])
                xt_cp[v] += mu * (xt[u] - xt[v])
                edges[t,new_edge] = torch.Tensor([u, v, 1])
                
            else:
                edges[t,new_edge] = torch.Tensor([u, v, 0])
                
            xt = xt_cp

        X.append(xt_cp)

    X = torch.from_numpy(np.vstack(X))
    return X, edges.long()
    
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