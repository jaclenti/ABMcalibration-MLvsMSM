import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.sparse import coo_array
from torch.distributions.beta import Beta
from torch.distributions.bernoulli import Bernoulli



class simulator_opinion_dynamics():
    
    def __init__(self, create_edges, opinion_update, generate_evidences = None, num_parameters = 3, dim_edges = 3):
        self.create_edges = create_edges
        self.opinion_update = opinion_update
        self.generate_evidences = generate_evidences
        self.num_parameters = num_parameters
        self.dim_edges = dim_edges
        
    def initialize_simulator(self, N, T, edge_per_t, evidences_per_t = 0, X0 = [], seed = None):
        self.N = N
        self.T = T
        self.edge_per_t = edge_per_t
        self.evidences_per_t = evidences_per_t
        
        if seed is not None:
            np.random.seed(seed) 
        
        if len(X0) == 0:
            self.X0 = torch.rand(N)
        
        
    def simulate_trajectory(self, parameters):
        assert len(parameters) == self.num_parameters, f"Required {self.num_parameters} parameters"
        
        edges, X = torch.zeros(self.T-1, self.edge_per_t, self.dim_edges), torch.zeros(self.T, self.N)
        X[0] = self.X0
        diff_X = self.X0[:,None] - self.X0[None,:]
        
        if self.evidences_per_t > 0:
            evidences = torch.zeros(self.T, self.evidences_per_t)
        
        
        for t in range(self.T-1):
            edges[t] = self.create_edges(self.N, self.edge_per_t, diff_X, parameters)
            X[t+1], diff_X = self.opinion_update(diff_X, X[t], edges[t], self.N, parameters)
        trajectory = [X, edges]
        
        if self.evidences_per_t > 0:
            evidences = self.generate_evidences(X, self.evidences_per_t, self.T, self.N, parameters)
            trajectory.append(evidences)
        
        return trajectory
    

    
    
    
    
    
    
########################## Bounded Confidence ###############################################
        


def create_edges_BC(N, edge_per_t, diff_X, parameters):
    epsilon, mu, rho = parameters
    
    u, v = torch.randint(low = 0, high = N, size = [2, edge_per_t], dtype = torch.long)
    s = (torch.rand(edge_per_t) < torch.sigmoid(rho * (epsilon - torch.abs(diff_X[u,v])))).to(torch.long)
    
    return torch.cat([u[:,None], v[:,None], s[:,None]], dim = 1)
        

def opinion_update_BC(diff_X, X_t, edges_t, N, parameters):
    epsilon, mu, rho = parameters
    u, v, s = edges_t.to(torch.long).T
    s = s.to(torch.float32)
    diff_X_uv = coo_array((diff_X[u, v] * s, (u, v)), shape = (N, N))
    
    updates = mu * (diff_X_uv.sum(axis = 0) - diff_X_uv.sum(axis = 1))
    X_t += updates
    X_t = torch.clamp(X_t, 1e-5, 1-1e-5)
    
    diff_X = X_t[:,None] - X_t[None,:]
    
    return X_t, diff_X

    
    
    
    
########################## Bounded Confidence with Backfire ###################################



def create_edges_BC_backfire(N, edge_per_t, diff_X, parameters):
    epsilon_plus, epsilon_minus, mu_plus, mu_minus, rho = parameters
    
    u, v = torch.randint(low = 0, high = N, size = [2, edge_per_t], dtype = torch.long)
    s_plus = (torch.rand(edge_per_t) < torch.sigmoid(rho * (epsilon_plus - torch.abs(diff_X[u,v])))).to(torch.long)
    s_minus = (torch.rand(edge_per_t) < torch.sigmoid(-rho * (epsilon_minus - torch.abs(diff_X[u,v])))).to(torch.long)
    
    return torch.cat([u[:,None], v[:,None], s_plus[:,None], s_minus[:,None]], dim = 1)
    
    
    
def opinion_update_BC_backfire(diff_X, X_t, edges_t, N, parameters):
    epsilon_plus, epsilon_minus, mu_plus, mu_minus, rho = parameters
    
    u, v, s_plus, s_minus = edges_t.to(torch.long).T
    s_plus, s_minus = s_plus.to(torch.float32), s_minus.to(torch.float32)
    
    diff_X_uv_plus = coo_array((diff_X[u, v] * s_plus, (u, v)), shape = (N, N))
    diff_X_uv_minus = coo_array((diff_X[u, v] * s_minus, (u, v)), shape = (N, N))
    
    updates_plus = mu_plus * (diff_X_uv_plus.sum(axis = 0) - diff_X_uv_plus.sum(axis = 1))
    X_t += updates_plus
    
    updates_minus = mu_minus * (diff_X_uv_minus.sum(axis = 0) - diff_X_uv_minus.sum(axis = 1))
    X_t -= updates_minus
    X_t = torch.clamp(X_t, 1e-5, 1-1e-5)
    
    
    diff_X = X_t[:,None] - X_t[None,:]
    
    return X_t, diff_X




######################### Evidences ######################################

def bernoulli_evidence(X, evidences_per_t, T, N, parameters):
    
    bernoulli_samples = Bernoulli(X).sample()
    
    evidences = [(u, bernoulli_samples[t, u]) for t in range(T) 
                 for u in torch.randint(low = 0, high = N, size = [1, evidences_per_t])]
    return evidences


def beta_evidence(sum_ab, X, evidences_per_t, T, N):
    beta_samples = Beta(sum_ab * X + 1e-7, sum_ab * (1 - X) + 1e-7).sample()
    if evidences_per_t != None:
        evidences = [(u, beta_samples[t, u]) for t in range(T) 
                     for u in torch.randint(low = 0, high = N, size = [1, evidences_per_t])]
    else:
        evidences = beta_samples
    return evidences

########################  Utils #####################################



#create a plot with all the opinion time trajectories of all users
def print_opinion_trajectory(X, figsize = (5,5)):
    fig, ax = plt.subplots(figsize = figsize)
    ax.plot(np.matrix(X))
    
    fig.show()

    
#convert the edges tensor
#input edges, such that edges[t][0] = u,v,s
#output u_v_t_s, such that edges[0] = u,v,t,s
def convert_edges_uvst(edges):
    max_T, edge_per_t, num_s = edges.size()
    
    uvst = torch.cat((edges.reshape(((max_T) * edge_per_t, num_s)), torch.Tensor(np.repeat(np.arange(max_T), edge_per_t))[:, None]), dim = 1).T.long()
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
   
    
    
def kappa_from_epsilon(epsilon, diff_X, rho):
    
    return torch.sigmoid(rho * (epsilon - torch.abs(diff_X)))
    
################### simulators ################


    
def BC_simulator(evidences = bernoulli_evidence):
    
    return simulator_opinion_dynamics(create_edges_BC, opinion_update_BC, bernoulli_evidence, num_parameters = 3, dim_edges = 3)
    
    

def BC_simulator_backfire(evidences = bernoulli_evidence):
    
    return simulator_opinion_dynamics(create_edges_BC_backfire, opinion_update_BC_backfire, bernoulli_evidence, num_parameters = 5, dim_edges = 4)
    
    
def simulate_BC(N,T,edge_per_t,evidences_per_t,parameters, model = "simple"):
    
    if model == "simple":
        simulator = BC_simulator()
    
    if model == "bf":
        simulator = BC_simulator_backfire()
    simulator.initialize_simulator(N,T,edge_per_t, evidences_per_t)
    
    X, edges, evidences = simulator.simulate_trajectory(parameters)
    
    return X, edges, evidences


