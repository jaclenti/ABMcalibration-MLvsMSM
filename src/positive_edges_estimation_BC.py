import torch
import torch.nn as nn
import numpy as np
from scipy.special import expit as sigmoid
import sys
sys.path += ['../src/']
import simulator_BC as sim_bc
from time import time
from scipy.special import logit
from tqdm import tqdm
from nodes_interactor import NodesInteractor
from initialize_model import EarlyStopping,RandomizeEpsilon,choose_optimizer


class BC_observed_positive(nn.Module): #PyTorch model for gradient optimization, for estimating epsilon observing edges and x0
    
    def __init__(self, epsilon0 = 0.1):
        
        super().__init__()
        theta = torch.tensor([logit(epsilon0)], requires_grad = True) # initialize weights with epsilon0
        self.theta = nn.Parameter(theta)     #define the parameters of the model
    
    def forward(self):
        epsilon = torch.sigmoid(self.theta) #at each step clip epsilon in the interval [0,1]
        return epsilon
    
    
def log_likelihood_t(edges_t, kappa_t, edge_per_t):
    N = len(kappa_t)
    
    positive_edges = edges_t[edges_t[:,2] == 1]
    u,v,s = positive_edges.T
    log_likelihood_pos = torch.sum(torch.log(kappa_t[u,v]))
    
    log_likelihood_neg = (edge_per_t - len(positive_edges)) * (torch.log(torch.sum((1 - kappa_t) / (N * (N - 1)))))
    
    return log_likelihood_pos, log_likelihood_neg
    
    
    
def tot_neg_log_likelihood(edges, epsilon, mu, rho, X, with_nodes_interactor = True, nodes_sampled = 50):
    T, edge_per_t, _ = edges.shape
    _, N = X.shape
    if nodes_sampled == 0:
        nodes_sampled = N
    
    if with_nodes_interactor:
        nodes_interactor = NodesInteractor(X[0], epsilon, mu, rho)

        tot_neg_log_likelihood_pos, tot_neg_log_likelihood_neg = 0, 0
        for t in range(T-1):
            log_likelihood_pos, log_likelihood_neg = log_likelihood_t(edges[t], nodes_interactor.kappa, edge_per_t)
            tot_neg_log_likelihood_pos += log_likelihood_pos
            tot_neg_log_likelihood_neg += log_likelihood_neg

            nodes_interactor.update_interactions_probabilities(edges[t], X[t])
    else:
        #kappa = torch.sigmoid(rho * (epsilon -  torch.abs(X[:-1,None,:] - X[:-1,:,None])))
        
        uvst = sim_bc.convert_edges_uvst(edges)
        u,v,s,t = positive_edges = uvst[:, uvst[2,:] == 1]
        #tot_neg_log_likelihood_pos = - torch.sum(torch.log(kappa[t,u,v]))
        kappa_pos = torch.sigmoid(rho * (epsilon -  torch.abs(X[t,u] - X[t,v])))
        tot_neg_log_likelihood_pos = - torch.sum(torch.log(kappa_pos))
        
        extracted_nodes = torch.Tensor([np.random.choice(np.arange(N), nodes_sampled) for k in range(T)]).long()
        extracted_X = torch.Tensor([np.array(X[t, extracted_nodes[t]]) for t in range(T)])
        u_triu, v_triu = np.array([(u,v) for u in range(nodes_sampled) for v in range(u+1, nodes_sampled)]).T
        diff_X = (extracted_X[:, None, :] - extracted_X[:, :, None])[:, u_triu, v_triu]
        
        kappa_neg = torch.sigmoid(rho * (epsilon - torch.abs(diff_X)))
        log_likelihood_neg_one_interaction_per_t = torch.log(1 - kappa_neg.mean(axis = 1))
        n_neg_interactions = edge_per_t - edges.sum(axis = 1)[:,2]
        
        tot_neg_log_likelihood_neg = - torch.sum(n_neg_interactions * log_likelihood_neg_one_interaction_per_t)
        
    return tot_neg_log_likelihood_neg + tot_neg_log_likelihood_pos    
    
"""
def tot_log_likelihood_observe_only_positive(X, edges, T, N, edge_per_t, epsilon, rho = 70, sample_pairs = 100):
    uvst = sim_bc.convert_edges_uvst(edges)
    u,v,s,t = positive_edges = uvst[:, uvst[2,:] == 1]
    
    diff_X_positive = torch.abs(X[t,u] - X[t,v])

    log_likelihood_observed = torch.sum(torch.log(torch.sigmoid(rho * (epsilon - diff_X_positive))))

    u_sample, v_sample = torch.randint(N, (T, sample_pairs, 2)).T
    abs_diff_sample_X = torch.abs(torch.gather(X, 1, u_sample.T) - torch.gather(X, 1, v_sample.T))
    kappa_neg = torch.sigmoid(-rho * (epsilon - abs_diff_sample_X))[:-1]
    n_negative_interactions = edge_per_t - edges.sum(axis = 1)[:,2]
    
    log_likelihood_non_observed = torch.sum(torch.log(kappa_neg.mean(axis = 1)) * n_negative_interactions)
    
     
    neg_tot_log_likelihood = - log_likelihood_observed - log_likelihood_non_observed
    
    return neg_tot_log_likelihood    
"""

def tot_log_likelihood_observe_only_positive(X, diff_X_positive, n_negative_interactions, T, N, epsilon, rho = 70, sample_pairs = 100):
    #uvst = sim_bc.convert_edges_uvst(edges)
    #u,v,s,t = positive_edges = uvst[:, uvst[2,:] == 1]
    
    #diff_X_positive = torch.abs(X[t,u] - X[t,v])

    log_likelihood_observed = torch.sum(torch.log(torch.sigmoid(rho * (epsilon - diff_X_positive))))

    u_sample, v_sample = torch.randint(N, (T, sample_pairs, 2)).T
    abs_diff_sample_X = torch.abs(torch.gather(X, 1, u_sample.T) - torch.gather(X, 1, v_sample.T))
    kappa_neg = torch.sigmoid(-rho * (epsilon - abs_diff_sample_X))[:-1]
    #n_negative_interactions = edge_per_t - edges.sum(axis = 1)[:,2]
    
    log_likelihood_non_observed = torch.sum(torch.log(kappa_neg.mean(axis = 1)) * n_negative_interactions)
    
     
    neg_tot_log_likelihood = - log_likelihood_observed - log_likelihood_non_observed
    
    return neg_tot_log_likelihood    

def estimation_epsilon_torch(X, edges, mu, epsilon0 = 0.25, rho = 70,
                             num_epochs = 50, optimizer_name = "adam", 
                             lr = 0.01, hide_progress = False, sample_pairs = 50,
                             min_delta = 1e-5, patience = 5, return_time = False, min_epochs = 20):
    t0 = time()
    history = {"epsilon": [epsilon0], "loss": [], "grad": []} #return the lists of the epsilon estimates and losses 

    T, N = X.size()
    
    _, edge_per_t, _ = edges.size()
    
    early_stopping = EarlyStopping(patience, min_delta, min_epochs)
    model_class = BC_observed_positive
    model = model_class(epsilon0)
    
    optimizer = choose_optimizer(optimizer_name, lr, model) #input the optimizer and learning rate, among SGD, adam, adagrad, nadam
    
    #### added to compute the distances only once
    
    uvst = sim_bc.convert_edges_uvst(edges)
    u,v,s,t = positive_edges = uvst[:, uvst[2,:] == 1]
    
    diff_X_positive = torch.abs(X[t,u] - X[t,v])

    n_negative_interactions = edge_per_t - edges.sum(axis = 1)[:,2]
    
    ####
    
    
    for epoch in tqdm(range(num_epochs), disable = hide_progress):
        epsilon = model()
        #loss = tot_neg_log_likelihood(edges, epsilon, mu, rho, X, with_nodes_interactor = False)
        
        #loss = tot_log_likelihood_observe_only_positive(X, edges, T, N, edge_per_t, epsilon, rho, sample_pairs)
        
        ### modified to compute the distances only once
        loss = tot_log_likelihood_observe_only_positive(X, diff_X_positive, n_negative_interactions, T, N, epsilon, rho, sample_pairs)
        ###

        
        
        loss.backward()
        optimizer.step()
        
        
        history["epsilon"].append(sigmoid(model.theta.item()))
        history["loss"].append(loss.item())
        history["grad"].append(model.theta.grad.item())
        
        # early stopping
        
        early_stopping(history["epsilon"][-1], history["epsilon"][-2], epoch)
        if early_stopping.early_stop:
            break
        
        
        optimizer.zero_grad()
        
    t1 = time()
    if return_time:
        history["time"] = t1 - t0
    return history  
    
    
    
    
    