import torch
import numpy as np
import pandas as pd
import torch.nn as nn

from scipy.special import expit as sigmoid
from scipy.special import logit
from importlib import reload
from time import time
from tqdm import tqdm
from torcheval.metrics.functional import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from operator import itemgetter 

import sys
sys.path += ["../src"]
import simulator_opinion_dynamics as sod
from initialize_model import EarlyStopping,RandomizeEpsilon,choose_optimizer
from initialize_model import EarlyStopping,RandomizeEpsilon,choose_optimizer
from simulator_opinion_dynamics import kappa_from_epsilon


#########################################      Simple BC  ######################################################################
################################################################################################################################




class Simple_BC_Estimation(nn.Module):
    
    def __init__(self, parameters0, X, edges):
        
        super().__init__()
        
        epsilon0, rho = parameters0
        self.rho = rho
        u,v,s,t = uvst = sod.convert_edges_uvst(edges)
        self.diff_X = X[t,u] - X[t,v]
        theta = torch.tensor([logit(2 * epsilon0)], requires_grad = True)
        self.theta = nn.Parameter(theta)
        
    def forward(self):
        epsilon = torch.sigmoid(self.theta) / 2
        kappa = kappa_from_epsilon(epsilon, self.diff_X, self.rho)
        return kappa
    
    def neg_log_likelihood_function(kappa, s, t_minibatch):
        
        return -(torch.sum(torch.log((kappa * s) + ((1 - kappa) * (1 - s)))))
    
    def neg_log_likelihood_function_minibatch(kappa, s, t_minibatch):
        
        return -(torch.sum(torch.log((kappa * s) + ((1 - kappa) * (1 - s)))[t_minibatch]))
    


def gradient_descent_simple_BC(X, edges, rho, num_epochs, epsilon0 = 0.25, optimizer_name = "adam",
                               lr = 0.05, hide_progress = True, minibatch_size = 0, early_stopping_kw = {"patience": 20, "min_delta": 1e-5, "min_epochs": 20, "long_run_delta": 1e-5, "long_run_diff":10, "long_run_patience": 5}):
    u,v,s,t = uvst = sod.convert_edges_uvst(edges)
    
    T,N = X.shape
    
    model_class = Simple_BC_Estimation
    model = model_class((epsilon0, rho), X, edges)
    if minibatch_size == 0:
        loss_function = model_class.neg_log_likelihood_function
    if minibatch_size > 0:
        loss_function = model_class.neg_log_likelihood_function_minibatch
    
    
    early_stopping = EarlyStopping(**early_stopping_kw)
    optimizer = choose_optimizer(optimizer_name, lr, model)
    
    history = {"epsilon": [epsilon0], "loss": []}
    
    t0 = time()
    for epoch in tqdm(range(num_epochs), disable = hide_progress):
        t_minibatch = torch.randperm(T-1)[:minibatch_size]
        
        kappa = model()
        loss = loss_function(kappa, s, t_minibatch)
        
        loss.backward()
        optimizer.step()
        
        history["epsilon"].append(sigmoid(model.theta.item()) / 2)
        history["loss"].append(loss.item())
        
        optimizer.zero_grad()
        
        if epoch > early_stopping_kw["min_epochs"]:
            early_stopping(history["epsilon"][-3], history["epsilon"][-2], history["epsilon"][-1], epoch)
        if early_stopping.early_stop:
            break
            
    t1 = time()
    history["time"] = t1 - t0
    
    return history



#################################     BC observed positive  #####################################################################
#################################################################################################################################

def list_of_t_indices(t, T):
    l_indices = [[] for u in np.arange(T - 1)]
    for ind in np.arange(len(t)):
        l_indices[t[ind].item()].append(ind)
    return l_indices
  
  

    

class BC_Observe_positive_Estimation(nn.Module):
    
    def __init__(self, parameters0, X, edges, sample_pairs = 50):
        
        super().__init__()
        
        epsilon0, rho = parameters0
        self.rho = rho
        uvst = sod.convert_edges_uvst(edges)
        u,v,s,t = uvst[:, uvst[2,:] == 1]
        self.X = X
        self.diff_X = X[t,u] - X[t,v]
        theta = torch.tensor([logit(2 * epsilon0)], requires_grad = True)
        self.theta = nn.Parameter(theta)
        
        _, self.edge_per_t, _ = edges.size()
        self.T, self.N = X.size()
        self.sample_pairs = sample_pairs
        self.n_negative_interactions = self.edge_per_t - edges.sum(axis = 1)[:,2]
        
    def forward(self):
        epsilon = torch.sigmoid(self.theta) / 2
        kappa_pos = kappa_from_epsilon(epsilon, self.diff_X, self.rho)
        
        u_sample, v_sample = torch.tensor((np.random.rand((self.T - 1) * self.sample_pairs, self.N).argpartition(2,axis = 1)[:,:2]).T)
        u_sample, v_sample = u_sample.reshape(self.sample_pairs, self.T-1), v_sample.reshape(self.sample_pairs, self.T-1)
        diff_sample_X = (torch.gather(self.X, 1, u_sample.T) - torch.gather(self.X, 1, v_sample.T))
        kappa_neg = 1 - (kappa_from_epsilon(epsilon, diff_sample_X, self.rho)).mean(axis = 1)

        return kappa_pos, kappa_neg
    
    def neg_log_likelihood_function(kappa_pos, kappa_neg, n_negative_interactions, t_minibatch, t_pos_minibatch):
        log_likelihood_observed = torch.sum(torch.log(kappa_pos))
        log_likelihood_non_observed = torch.sum(torch.log(kappa_neg) * n_negative_interactions)
        
        neg_tot_log_likelihood = - log_likelihood_observed - log_likelihood_non_observed
        
        return neg_tot_log_likelihood

    def neg_log_likelihood_function_minibatch(kappa_pos, kappa_neg, n_negative_interactions, t_minibatch, t_pos_minibatch):
        log_likelihood_observed = torch.sum(torch.log(kappa_pos)[t_pos_minibatch])
        log_likelihood_non_observed = torch.sum((torch.log(kappa_neg) * n_negative_interactions)[t_minibatch])
        
        neg_tot_log_likelihood = - log_likelihood_observed - log_likelihood_non_observed
        
        return neg_tot_log_likelihood



def gradient_descent_BC_observe_positive(X, edges, rho, num_epochs, sample_pairs = 50,
                                         epsilon0 = 0.25, optimizer_name = "adam",
                                         lr = 0.05, hide_progress = True, minibatch_size = 0,
                                         early_stopping_kw = {"patience": 20, "min_delta": 1e-5, "min_epochs": 20, "long_run_delta": 1e-5, "long_run_diff":10, "long_run_patience": 5}
                                        ):
    u,v,s,t = uvst = sod.convert_edges_uvst(edges)
    
    model_class = BC_Observe_positive_Estimation
    model = model_class((epsilon0, rho), X, edges)
    if minibatch_size == 0:
        loss_function = model_class.neg_log_likelihood_function
    if minibatch_size > 0:
        loss_function = model_class.neg_log_likelihood_function_minibatch
    
    T,N = X.shape
    
    t_pos = uvst[3, uvst[2,:] == 1]
    indices_t = list_of_t_indices(t_pos, T)
    
    early_stopping = EarlyStopping(**early_stopping_kw)
    optimizer = choose_optimizer(optimizer_name, lr, model)
    
    history = {"epsilon": [epsilon0], "loss": []}
    t_minibatch, t_pos_minibatch = None, None
    t0 = time()
    
    for epoch in tqdm(range(num_epochs), disable = hide_progress):
        
        if minibatch_size > 0:
            t_minibatch = torch.randperm(T-1)[:minibatch_size]
            t_pos_minibatch = sum(list(itemgetter(*t_minibatch)(indices_t)), []) #use itemgetter to access multiple element of a list (same as [l[u] for u in ind], use sum(list, []) to flatten the list
        
        kappa_pos, kappa_neg = model()
        loss = loss_function(kappa_pos, kappa_neg, model.n_negative_interactions, t_minibatch, t_pos_minibatch)
        
        loss.backward()
        optimizer.step()
        
        history["epsilon"].append(sigmoid(model.theta.item()) / 2)
        history["loss"].append(loss.item())
        
        optimizer.zero_grad()
        
        if epoch > early_stopping_kw["min_epochs"]:
            early_stopping(history["epsilon"][-3], history["epsilon"][-2], history["epsilon"][-1], epoch)
        if early_stopping.early_stop:
            break
            
    t1 = time()
    history["time"] = t1 - t0
    
    return history

########################################    BC evidences X ######################################################################
#################################################################################################################################


def choose_optimizer_evidences(optimizer_name, lr, model, X0_lr_scale = 1):
    optimizer_list = ["adam", "SGD", "nadam", "adagrad", "RMSprop"]
    assert optimizer_name in optimizer_list, f"Optimizer must be in {optimizer_list}"
    
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam([
            {'params': model.logit_X0, 'lr': lr * X0_lr_scale},
            {'params': model.theta, 'lr': lr}
        ], lr = lr)
        
        optimizer = torch.optim.Adam([
            {'params': model.logit_X0, 'lr': lr * X0_lr_scale},
            {'params': model.theta, 'lr': lr}
        ], lr = lr) #define the optimizer with the input learning rate
    if optimizer_name == "SGD":
        optimizer = torch.optim.SGD([
            {'params': model.logit_X0, 'lr': lr * X0_lr_scale},
            {'params': model.theta, 'lr': lr}
        ], lr = lr, momentum = 0.9) #define the optimizer with the input learning rate
    if optimizer_name == "nadam":
        optimizer = torch.optim.NAdam([
            {'params': model.logit_X0, 'lr': lr * X0_lr_scale},
            {'params': model.theta, 'lr': lr}
        ], lr = lr) #define the optimizer with the input learning rate        
    if optimizer_name == "adagrad":
        optimizer = torch.optim.Adagrad([
            {'params': model.logit_X0, 'lr': lr * X0_lr_scale},
            {'params': model.theta, 'lr': lr}
        ], lr = lr) #define the optimizer with the input learning rate
    if optimizer_name == "RMSprop":
        optimizer = torch.optim.RMSprop([
            {'params': model.logit_X0, 'lr': lr * X0_lr_scale},
            {'params': model.theta, 'lr': lr}
        ], lr = lr)
    return optimizer




def count_s_from_edge(e):
    e_unique = e.unique(dim = 0, return_counts = True)
    e_unique[0][:,2] = e_unique[0][:,2] * e_unique[1]
    e_sum_s = e_unique[0]
    return e_sum_s[e_sum_s[:,2] > 0]


def edges_coo_mu(edges, mu, N):
        edges_count_s = [count_s_from_edge(edges[t]) for t in range(len(edges))]
        
        M = [torch.sparse_coo_tensor(indices = edges_count_s[t][:,:2].T, dtype = torch.float32,
                                     values = mu * edges_count_s[t][:,2], 
                                     size = [N, N]) for t in range(len(edges))]
        
        return M
    

def X_from_X0_coo_edges(X0, M, T, N):
    X = torch.zeros([T, N], dtype = torch.float32)
    
    X[0] = X0
    for t in range(T - 1):
        updates = ((X[t] * M[t].to_dense()).sum(dim = 1) - (X[t] * M[t].to_dense().T).sum(dim = 0) +\
                   (X[t] * M[t].to_dense().T).sum(dim = 1) - (X[t] * M[t].to_dense()).sum(dim = 0))
        X[t+1] = X[t] + updates
    
    return X
  
  
"""

###  much slower  ###

def X_from_X0_coo_edges(X0, indices_M, values_M, T, N):
    X = torch.zeros([T, N], dtype = torch.float32)
    
    X[0] = X0
    for t in range(T - 1):
        diff_Xt = torch.sparse_coo_tensor(indices = indices_M[t],
                                  values = values_M[t] * (X[t,indices_M[t][0]] - X[t, indices_M[t][1]]),
                                  size = [N,N])
        #updates = - torch.sparse.mm(diff_Xt, torch.ones(N,1)) + torch.sparse.mm(diff_Xt.T, torch.ones(N,1))
        updates = - torch.sparse.mm(diff_Xt, torch.ones(N,1))[:,0] + torch.sparse.mm(diff_Xt.T, torch.ones(N,1))[:,0]

        #updates = ((X[t] * M[t].to_dense()).sum(dim = 1) - (X[t] * M[t].to_dense().T).sum(dim = 0) +\
        #           (X[t] * M[t].to_dense().T).sum(dim = 1) - (X[t] * M[t].to_dense()).sum(dim = 0))
        X[t+1] = X[t] + updates
    
    return X
    
"""

def evidences_diff_from_X0_coo_edges(X0, indices_M, values_M, T, N, edges, evidences_indices):
    diff_X = torch.empty(T-1, len(edges[0]))
    X_ = X0.clone().detach()
    X_evidences = torch.empty(T, len(evidences_indices[0]))
    
    for t in range(T - 1):
        diff_Xt = torch.sparse_coo_tensor(indices = indices_M[t],
                                          values = values_M[t] * (X_[indices_M[t][0]] - X_[indices_M[t][1]]),
                                          size = [N,N])
        #updates = - torch.sparse.mm(diff_Xt, torch.ones(N,1)) + torch.sparse.mm(diff_Xt.T, torch.ones(N,1))
        updates = - torch.sparse.mm(diff_Xt, torch.ones(N,1))[:,0] + torch.sparse.mm(diff_Xt.T, torch.ones(N,1))[:,0]
        
        #updates = ((X[t] * M[t].to_dense()).sum(dim = 1) - (X[t] * M[t].to_dense().T).sum(dim = 0) +\
        #           (X[t] * M[t].to_dense().T).sum(dim = 1) - (X[t] * M[t].to_dense()).sum(dim = 0))
        
        
        u,v,_ = edges[t].T
        
        diff_X[t] = X_[u] - X_[v]
        X_evidences[t] = X_[evidences_indices[t]]
        
        X_ += updates
    X_evidences[T-1] = X_[evidences_indices[T-1]]
    
    return diff_X, X_evidences




class BC_evidence_X_Estimation(nn.Module):
    
    def __init__(self, parameters0, X, edges, evidences):
        
        super().__init__()
        
        epsilon0, mu, rho = parameters0
        self.rho = rho
        self.mu = mu
        
        self.u,self.v,self.s,self.t = uvst = sod.convert_edges_uvst(edges)

        self.X = X
        self.diff_X = X[self.t,self.u] - X[self.t,self.v]
        _, self.edge_per_t, _ = edges.size()
        self.T, self.N = X.size()
        
        self.bce = nn.BCELoss(reduction = "sum")
        
        self.M = edges_coo_mu(edges, mu, self.N)
                
        self.evidences_per_t = len(evidences[0][0])

        self.evidences_indices = torch.cat([evidences[k][0][None,:] for k in range(len(evidences))], dim = 0) #tensor with the indices of the users of which we know the evidence
        self.evidences_opinions = torch.cat([evidences[k][1][None,:] for k in range(len(evidences))], dim = 0).reshape(self.T * self.evidences_per_t).to(torch.float32) #only the evidences of these users
        self.evideneces = evidences
        
        X0 = torch.rand(self.N, dtype = torch.float32, requires_grad = True) # random initialization of the opinions
        self.logit_X0 = nn.Parameter(torch.logit(X0))   #define the parameters of the model
        
        #theta = torch.tensor([logit(epsilon0)], requires_grad = True)
        theta = torch.tensor([logit(2 * epsilon0)], requires_grad = True)
        self.theta = nn.Parameter(theta)
        
        
    def forward(self):
        epsilon = torch.sigmoid(self.theta) / 2
        #epsilon = torch.sigmoid(self.theta) 
        X0 = torch.sigmoid(self.logit_X0)   #at each step clip X0 in the interval [0,1]
        X = X_from_X0_coo_edges(X0, self.M, self.T, self.N)
        #X = X_from_X0_coo_edges(X0, self.indices_M, self.values_M, self.T, self.N)
        
        diff_X = X[self.t,self.u] - X[self.t, self.v] 
        kappa = kappa_from_epsilon(epsilon, diff_X, self.rho) # compute probability of interaction with current estimate of epsilon
        
        return X, kappa
    
    
    def neg_log_likelihood_function(self, kappa, s, evidences_indices, evidences_opinions, evidences_per_t, X, t_minibatch):
        T, _ = X.shape
        real_opinions = torch.cat([X[t, evidences_indices[t]] for t in range(T)])
        
        loss_edges = -torch.sum(torch.log(s * kappa + (1 - s) * (1 - kappa)))
        
        loss_evidences = -torch.sum(torch.log(evidences_opinions * real_opinions + (1 - evidences_opinions) * (1 - real_opinions)))
        loss_evidences_flip = -torch.sum(torch.log(evidences_opinions * (1 - real_opinions) + (1 - evidences_opinions) * (1 - (1 - real_opinions))))
        
        X0_flip = True if loss_evidences > loss_evidences_flip else False
        
        return torch.min(loss_edges + loss_evidences, loss_edges + loss_evidences_flip), X0_flip
    
    def neg_log_likelihood_function_minibatch(self, kappa, s, evidences_indices, evidences_opinions, evidences_per_t, X, t_minibatch):
        T, _ = X.shape
        real_opinions = torch.cat([X[t, evidences_indices[t]] for t in range(T)])
        
        loss_edges = -torch.sum(torch.log(s * kappa + (1 - s) * (1 - kappa))[t_minibatch])
        
        loss_evidences = -torch.sum(torch.log(evidences_opinions * real_opinions + (1 - evidences_opinions) * (1 - real_opinions))[t_minibatch])
        loss_evidences_flip = -torch.sum(torch.log(evidences_opinions * (1 - real_opinions) + (1 - evidences_opinions) * (1 - (1 - real_opinions)))[t_minibatch])
        
        X0_flip = True if loss_evidences > loss_evidences_flip else False
        
        return torch.min(loss_edges + loss_evidences, loss_edges + loss_evidences_flip), X0_flip
    

    
def gradient_descent_BC_evidence_X(X, edges, evidences, mu, rho, num_epochs,
                                   epsilon0 = 0.25, optimizer_name = "adam",
                                   lr = 0.05,  X0_lr_scale = 1,
                                   hide_progress = True, minibatch_size = 0,
                                   early_stopping_kw = {"patience": 20, "min_delta": 1e-5, "min_epochs": 20, "long_run_delta": 1e-5, "long_run_diff":10, "long_run_patience": 5}):
    u,v,s,t = uvst = sod.convert_edges_uvst(edges)
    T,_ = X.shape
    
    
    model_class = BC_evidence_X_Estimation
    model = model_class((epsilon0, mu, rho), X, edges, evidences)
    if minibatch_size == 0:
        loss_function = model_class.neg_log_likelihood_function
    if minibatch_size > 0:
        loss_function = model_class.neg_log_likelihood_function_minibatch
    
    early_stopping = EarlyStopping(**early_stopping_kw)
    #optimizer = choose_optimizer(optimizer_name, lr, model)
    optimizer = choose_optimizer_evidences(optimizer_name, lr, model, X0_lr_scale)
    
    history = {"epsilon": [epsilon0], "loss": [], "X0": []}
    
    t0 = time()
    for epoch in tqdm(range(num_epochs), disable = hide_progress):
        t_minibatch = torch.randperm(T-1)[:minibatch_size]
        
        X_, kappa = model()
        loss, X0_flip = loss_function(model, kappa, model.s, model.evidences_indices, model.evidences_opinions, 
                             model.evidences_per_t, X_, t_minibatch)
        
        loss.backward()
        optimizer.step()
        
        X0_estimate = sigmoid(model.logit_X0.detach()) + (1 - 2 * sigmoid(model.logit_X0.detach())) * X0_flip
        history["X0"].append(X0_estimate)
        #history["epsilon"].append(sigmoid(model.theta.item()))
        history["epsilon"].append(sigmoid(model.theta.item()) / 2)
        history["loss"].append(loss.item())
        
        if loss.item() == np.inf:
            
            model = model_class((np.random.rand() / 2, mu, rho), X, edges, evidences)
            optimizer = choose_optimizer_evidences(optimizer_name, lr, model, X0_lr_scale / 2)

        
        optimizer.zero_grad()
        
        if epoch > early_stopping_kw["min_epochs"]:
            
            early_stopping(history["epsilon"][-2], history["epsilon"][-1], history["epsilon"][-early_stopping_kw["long_run_diff"]], epoch)
            if early_stopping.early_stop:
                break
    
    
    t1 = time()
    history["time"] = t1 - t0
    
    return history


def X_from_X0_and_edges(X0, edges, mu):
    N = len(X0)
    T_, edge_per_t,_ = edges.shape
    T = T_ + 1
    
    X = torch.zeros([T, N], dtype = torch.float32)
    
    edges_count_s = [count_s_from_edge(edges[t]) for t in range(len(edges))]
    indices_M = [edges_count_s[t][:,:2].T.long() for t in range(T - 1)]
    values_M = [mu * edges_count_s[t][:,2] for t in range(T - 1)]
        
    
    X[0] = X0
    for t in range(T - 1):
        diff_Xt = torch.sparse_coo_tensor(indices = indices_M[t],
                                  values = values_M[t] * (X[t,indices_M[t][0]] - X[t, indices_M[t][1]]),
                                  size = [N,N])
        #updates = - torch.sparse.mm(diff_Xt, torch.ones(N,1)) + torch.sparse.mm(diff_Xt.T, torch.ones(N,1))
        updates = - torch.sparse.mm(diff_Xt, torch.ones(N,1))[:,0] + torch.sparse.mm(diff_Xt.T, torch.ones(N,1))[:,0]

        #updates = ((X[t] * M[t].to_dense()).sum(dim = 1) - (X[t] * M[t].to_dense().T).sum(dim = 0) +\
        #           (X[t] * M[t].to_dense().T).sum(dim = 1) - (X[t] * M[t].to_dense()).sum(dim = 0))
        X[t+1] = X[t] + updates
    
    return X
      
    
    
    
################################ BC with backfire effect and evidences ################################################
#################################################################################################################################



def X_from_X0_coo_edges_bf(X0, M_plus, M_minus, mu_plus, mu_minus, T, N):
    X = torch.zeros([T, N], dtype = torch.float32)
    
    X[0] = X0
    for t in range(T - 1):
        updates_plus = mu_plus * ((X[t] * M_plus[t].to_dense()).sum(dim = 1) - (X[t] * M_plus[t].to_dense().T).sum(dim = 0) +
(X[t] * M_plus[t].to_dense().T).sum(dim = 1) - (X[t] * M_plus[t].to_dense()).sum(dim = 0))
        updates_minus = mu_minus * ((X[t] * M_minus[t].to_dense()).sum(dim = 1) - (X[t] * M_minus[t].to_dense().T).sum(dim = 0) + (X[t] * M_minus[t].to_dense().T).sum(dim = 1) - (X[t] * M_minus[t].to_dense()).sum(dim = 0))
        X[t+1] = torch.clamp(X[t] + updates_plus - updates_minus, 1e-5, 1-1e-5)
    
    return X

def X_from_X0_coo_edges_without_M(X0, edges, mu_plus, mu_minus, T, N):
    X = torch.zeros([T, N], dtype = torch.float32)
    
    M_plus = edges_coo_mu(edges[:,:,[0,1,2]], 1, N)
    M_minus = edges_coo_mu(edges[:,:,[0,1,3]], 1, N)
    
    
    X[0] = X0
    for t in range(T - 1):
        updates = mu_plus * ((X[t] * M_plus[t].to_dense()).sum(dim = 1) - (X[t] * M_plus[t].to_dense().T).sum(dim = 0) +\
                   (X[t] * M_plus[t].to_dense().T).sum(dim = 1) - (X[t] * M_plus[t].to_dense()).sum(dim = 0))
        updates -= mu_minus * ((X[t] * M_minus[t].to_dense()).sum(dim = 1) - (X[t] * M_minus[t].to_dense().T).sum(dim = 0) +\
                   (X[t] * M_minus[t].to_dense().T).sum(dim = 1) - (X[t] * M_minus[t].to_dense()).sum(dim = 0))
        X[t+1] = torch.clamp(X[t] + updates, 0, 1)
        
    
    return X

class BC_evidence_X_Estimation_bf(nn.Module):
    
    def __init__(self, parameters0, T, N, #X, 
                 edges, evidences):
        
        super().__init__()
        
        epsilon0_plus, epsilon0_minus, mu_plus, mu_minus, rho = parameters0
        self.rho = rho
        self.mu_plus = mu_plus
        self.mu_minus = mu_minus
        
        self.u,self.v, self.s_plus, self.s_minus,self.t = uvst = sod.convert_edges_uvst(edges)

        #self.X = X
        #self.diff_X = X[self.t,self.u] - X[self.t,self.v]
        _, self.edge_per_t, _ = edges.size()
        #self.T, self.N = X.size()
        self.T, self.N = T, N
        
        self.M_plus = edges_coo_mu(edges[:,:,[0,1,2]], 1, self.N)
        self.M_minus = edges_coo_mu(edges[:,:,[0,1,3]], 1, self.N)
        #self.M =  M_plus + M_minus
        
        ########
        #edges_count_s = [count_s_from_edge(edges[t]) for t in range(self.T - 1)]
        #self.indices_M = [edges_count_s[t][:,:2].T.long() for t in range(self.T - 1)]
        #self.values_M = [mu * edges_count_s[t][:,2] for t in range(self.T - 1)]
        #######
        
        self.evidences_per_t = len(evidences[0][0])

        self.evidences_indices = torch.cat([evidences[k][0][None,:] for k in range(len(evidences))], dim = 0) #tensor with the indices of the users of which we know the evidence
        self.evidences_opinions = torch.cat([evidences[k][1][None,:] for k in range(len(evidences))], dim = 0).reshape(self.T * self.evidences_per_t).to(torch.float32) #only the evidences of these users
        self.evideneces = evidences
        
        X0 = torch.rand(self.N, dtype = torch.float32, requires_grad = True) # random initialization of the opinions
        self.logit_X0 = nn.Parameter(torch.logit(X0))   #define the parameters of the model
        
        theta = torch.tensor([logit(2*epsilon0_plus),logit(2*(epsilon0_minus-0.5))], requires_grad = True)
        self.theta = nn.Parameter(theta)
        
        
    def forward(self):
        #epsilon_plus, epsilon_minus = torch.sigmoid(self.theta)
        epsilon_plus = torch.sigmoid(self.theta[0]) / 2
        epsilon_minus = 0.5 + torch.sigmoid(self.theta[1]) / 2
        
        
        #if epsilon_plus > epsilon_minus:
        #    self.theta.data = self.theta.data.flip(0)
        #    epsilon_plus, epsilon_minus = epsilon_minus, epsilon_plus

        
        X0 = torch.sigmoid(self.logit_X0)   #at each step clip X0 in the interval [0,1]
        X = X_from_X0_coo_edges_bf(X0, self.M_plus, self.M_minus, self.mu_plus, self.mu_minus, self.T, self.N)
        
        #X = X_from_X0_coo_edges(X0, self.indices_M, self.values_M, self.T, self.N)
        
        diff_X = X[self.t,self.u] - X[self.t, self.v] 
        kappa_plus = kappa_from_epsilon(epsilon_plus, diff_X, self.rho)
        kappa_minus = kappa_from_epsilon(epsilon_minus, diff_X, -self.rho) # compute probability of interaction with current estimate of epsilon
        
        return X, kappa_plus, kappa_minus
    
    
    def neg_log_likelihood_function(self, kappa_plus, kappa_minus, s_plus, s_minus, evidences_indices, evidences_opinions, evidences_per_t, X, t_minibatch):
        T, _ = X.shape
        
        loss_edges_plus = -torch.sum(torch.log(s_plus * kappa_plus + (1 - s_plus) * (1 - kappa_plus)))
        loss_edges_minus = -torch.sum(torch.log(s_minus * kappa_minus + (1 - s_minus) * (1 - kappa_minus)))
        
        real_opinions = torch.cat([X[t, evidences_indices[t]] for t in range(T)])
        
        loss_evidences = -torch.sum(torch.log(evidences_opinions * real_opinions + (1 - evidences_opinions) * (1 - real_opinions)))
        loss_evidences_flip = -torch.sum(torch.log(evidences_opinions * (1 - real_opinions) + (1 - evidences_opinions) * (1 - (1 - real_opinions))))
        
        X0_flip = True if loss_evidences > loss_evidences_flip else False
        
        return torch.min(loss_edges_plus + loss_edges_minus + loss_evidences, loss_edges_plus + loss_edges_minus + loss_evidences_flip), X0_flip
    
        
    

    def neg_log_likelihood_function_minibatch(self, kappa_plus, kappa_minus, s_plus, s_minus, evidences_indices, evidences_opinions, evidences_per_t, X, t_minibatch):
        T, _ = X.shape
        
        loss_edges_plus = -torch.sum(torch.log(s_plus * kappa_plus + (1 - s_plus) * (1 - kappa_plus))[t_minibatch])
        loss_edges_minus = -torch.sum(torch.log(s_minus * kappa_minus + (1 - s_minus) * (1 - kappa_minus))[t_minibatch])
        
        real_opinions = torch.cat([X[t, evidences_indices[t]] for t in range(T)])
        
        loss_evidences = -torch.sum(torch.log(evidences_opinions * real_opinions + (1 - evidences_opinions) * (1 - real_opinions))[t_minibatch])
        loss_evidences_flip = -torch.sum(torch.log(evidences_opinions * (1 - real_opinions) + (1 - evidences_opinions) * (1 - (1 - real_opinions)))[t_minibatch])
        
        X0_flip = True if loss_evidences > loss_evidences_flip else False
        
        return torch.min(loss_edges_plus + loss_edges_minus + loss_evidences, loss_edges_plus + loss_edges_minus + loss_evidences_flip), X0_flip
    

    
def gradient_descent_BC_evidence_X_backfire(X, edges, evidences, mu_plus, mu_minus, rho, num_epochs,
                                   epsilon0_plus = 0.25, epsilon0_minus = 0.75, optimizer_name = "adam",
                                   lr = 0.05, hide_progress = True, minibatch_size = 0, X0_lr_scale = 50,early_stopping_kw = {"patience": 20, "min_delta": 1e-5, "min_epochs": 20, "long_run_delta": 1e-5, "long_run_diff":10, "long_run_patience": 5}):
    u,v, s_plus,s_minus, t = uvst = sod.convert_edges_uvst(edges)
    T,N = X.shape
    
    model_class = BC_evidence_X_Estimation_bf
    #model = model_class((epsilon0_plus, epsilon0_minus, mu_plus, mu_minus, rho), X, edges, evidences)
    model = model_class((epsilon0_plus, epsilon0_minus, mu_plus, mu_minus, rho), T, N, edges, evidences)
    if minibatch_size == 0:
        loss_function = model_class.neg_log_likelihood_function
    if minibatch_size > 0:
        loss_function = model_class.neg_log_likelihood_function_minibatch
    
    early_stopping = EarlyStopping(**early_stopping_kw)
    #optimizer = choose_optimizer(optimizer_name, lr, model)
    optimizer = choose_optimizer_evidences(optimizer_name, lr, model, X0_lr_scale = X0_lr_scale)
    
    history = {"epsilon_plus": [epsilon0_plus], "epsilon_minus": [epsilon0_minus], "loss": [], "X0": []}
    
    t0 = time()
    for epoch in tqdm(range(num_epochs), disable = hide_progress):
        t_minibatch = torch.randperm(T-1)[:minibatch_size]
        
        X_, kappa_plus, kappa_minus = model()
        
        loss, X0_flip = loss_function(model, kappa_plus, kappa_minus, model.s_plus, model.s_minus, model.evidences_indices, model.evidences_opinions, model.evidences_per_t, X_, t_minibatch)
        
        loss.backward()
        optimizer.step()
        
        X0_estimate = sigmoid(model.logit_X0.detach()) + (1 - 2 * sigmoid(model.logit_X0.detach())) * X0_flip
        
        history["X0"].append(X0_estimate)
        history["epsilon_plus"].append(sigmoid(model.theta[0].item()) / 2)
        history["epsilon_minus"].append(0.5 + sigmoid(model.theta[1].item()) / 2)
        history["loss"].append(loss.item())
        
        if loss.item() == np.inf:
            model = model_class((np.random.rand() / 2, 0.5 + np.random.rand() / 2, mu_plus, mu_minus, rho), T, N, edges, evidences)
            optimizer = choose_optimizer_evidences(optimizer_name, lr, model, X0_lr_scale / 2)

        optimizer.zero_grad()
        
        if epoch > min_epochs:
            early_stopping(history["epsilon_plus"][-3], history["epsilon_plus"][-2], history["epsilon_plus"][-1], epoch)
        if early_stopping.early_stop:
            break
            
    t1 = time()
    history["time"] = t1 - t0
    
    return history



##############################################################################################################################





def estimation_BC_model_from_data(X, edges, evidences, parameters,epsilon0 = 0.25, model_name = None,optimizer_name = "adam", lr = 0.01, num_epochs = 100, hide_progress = True, minibatch_size = 0, epsilon0_plus = 0.25, epsilon0_minus = 0.75, sample_pairs = 50, summarized = False, X0_lr_scale = 50, intermediate_estimates = True, intermediate_step = 50, early_stopping_kw = {"patience": 20, "min_delta": 1e-5, "min_epochs": 20, "long_run_delta": 1e-5, "long_run_diff":10, "long_run_patience": 5}):
    
    model_name_list = ["simple_BC", "BC_with_evidence", "BC_observed_positive", "BC_backfire"]
    assert model_name in model_name_list, f"Model must be in {model_name_list}"
    
    if len(parameters) == 3:
        epsilon, mu, rho = parameters
        
    
    
    if model_name == "simple_BC":
        
        history = gradient_descent_simple_BC(X = X, edges = edges, rho = rho, num_epochs = num_epochs, 
                                             epsilon0 = epsilon0, optimizer_name = optimizer_name,
                                             lr = lr, hide_progress = hide_progress, 
                                             minibatch_size = minibatch_size, early_stopping_kw = early_stopping_kw)
    if model_name == "BC_observed_positive":
        
        history = gradient_descent_BC_observe_positive(X = X, edges = edges, rho = rho, num_epochs = num_epochs, 
                                                       epsilon0 = epsilon0, optimizer_name = optimizer_name,
                                                       lr = lr, hide_progress = hide_progress, 
                                                       minibatch_size = minibatch_size, sample_pairs = sample_pairs, early_stopping_kw = early_stopping_kw)
    if model_name == "BC_with_evidence":

        history = gradient_descent_BC_evidence_X(X = X, edges = edges, evidences = evidences, mu = mu, rho = rho,
                                                 num_epochs = num_epochs, 
                                                 epsilon0 = epsilon0, optimizer_name = optimizer_name,
                                                 lr = lr, hide_progress = hide_progress, 
                                                 minibatch_size = minibatch_size, X0_lr_scale = X0_lr_scale, 
                                                 early_stopping_kw = early_stopping_kw)
    if model_name == "BC_backfire":
        epsilon_plus, epsilon_minus, mu_plus, mu_minus, rho = parameters
        
        history = gradient_descent_BC_evidence_X_backfire(X = X, edges = edges, evidences = evidences, rho = rho, 
                                                          num_epochs = num_epochs, mu_plus = mu_plus, mu_minus = mu_minus,
                                                          epsilon0_plus = epsilon0_plus, epsilon0_minus = epsilon0_minus, 
                                                          optimizer_name = optimizer_name, 
                                                          lr = lr, hide_progress = hide_progress, 
                                                          minibatch_size = minibatch_size, X0_lr_scale = X0_lr_scale, early_stopping_kw = early_stopping_kw)

    if summarized:
        summary = {"time": history["time"], "num_epochs": len(history["epsilon"]), "rho": rho}
        if model_name in ["simple_BC", "BC_observed_positive", "BC_with_evidence"]:
            summary.update({"epsilon_estimated": history["epsilon"][-1], "real_epsilon": epsilon, "mu": mu, "distance_epsilon": np.abs(epsilon - history["epsilon"][-1])})
            if intermediate_estimates:
                for epoch in np.arange(intermediate_step, len(history["epsilon"]) - 1, intermediate_step):
                    summary[f"epsilon_epoch_{epoch}"] = history["epsilon"][epoch] 
                    summary[f"distance_epsilon_epoch_{epoch}"] = np.abs(epsilon - history["epsilon"][epoch])                

        if model_name in ["BC_with_evidence", "BC_backfire"]:
            X0_r2 = r2_score(X[0], history["X0"][-1].detach()).item()
            X0_mae = nn.L1Loss()(X[0], history["X0"][-1].detach()).item()
            X0_mse = nn.MSELoss()(X[0], history["X0"][-1].detach()).item()
            summary.update({"X0_r2": X0_r2, "X0_mae": X0_mae, "X0_mse": X0_mse})
            if intermediate_estimates:
                for epoch in np.arange(intermediate_step, len(history["epsilon"]) - 1, intermediate_step):
                    summary[f"X0_r2_epoch_{epoch}"] = r2_score(X[0], history["X0"][epoch].detach()).item()
                    summary[f"X0_mae_epoch_{epoch}"] = nn.L1Loss()(X[0], history["X0"][epoch].detach()).item()
                    summary[f"X0_mse_epoch_{epoch}"] = nn.MSELoss()(X[0], history["X0"][epoch].detach()).item()

        if model_name == "BC_backfire":
            summary.update({"epsilon_plus_estimated": history["epsilon_plus"][-1], "epsilon_minus_estimated": history["epsilon_minus"][-1], "real_epsilon_plus": epsilon_plus, "real_epsilon_minus": epsilon_minus, "mu_plus": mu_plus, "mu_minus": mu_minus, "distance_epsilon_plus": np.abs(epsilon_plus - history["epsilon_plus"][-1]), "distance_epsilon_minus": np.abs(epsilon_minus - history["epsilon_minus"][-1])})
            if intermediate_estimates:
                for epoch in np.arange(intermediate_step, len(history["epsilon"]) - 1, intermediate_step):
                    summary[f"epsilon_plus_epoch_{epoch}"] = history["epsilon_plus"][epoch]
                    summary[f"epsilon_minus_epoch_{epoch}"] = history["epsilon_minus"][epoch]
                    summary[f"distance_epsilon_plus_epoch_{epoch}"] = np.abs(epsilon_plus - history["epsilon_plus"][epoch])
                    summary[f"distance_epsilon_minus_epoch_{epoch}"] = np.abs(epsilon_minus - history["epsilon_minus"][epoch])

        return summary
 

    return history




def simulate_trajectory_and_estimate(N,T,edge_per_t,parameters,model_name,evidences_per_t = 1,optimizer_name = "adam", lr = 0.01, num_epochs = 100,epsilon0 = 0.25, hide_progress = True, patience = 10, min_delta = 1e-6,
                                     min_epochs = 20, sample_pairs = 50, repetitions = 1, verbose = False):
    
    model_name_list = ["simple_BC", "BC_with_evidence", "BC_observed_positive"]
    assert model_name in model_name_list, f"Model must be in {model_name_list}"
        
    BC_simulator = sod.BC_simulator()
    BC_simulator.initialize_simulator(N,T,edge_per_t, evidences_per_t)
    
    X, edges, evidences = BC_simulator.simulate_trajectory(parameters)
    
    simulations_results = []
    
    for _ in range(repetitions):
        if verbose: 
            print(f"Repetition {_}")
        
        history = estimation_BC_model_from_data(X, edges, evidences, parameters, epsilon0, model_name, evidences_per_t, optimizer_name, lr, num_epochs, hide_progress, patience, min_delta,min_epochs, sample_pairs)
        if model_name == "BC_with_evidence":
            X0_r2 = r2_score(X[0], history["X0"][-1].detach()).item()
        else:
            X0_r2 = None
            
        simulations_results.append( {"epsilon_estimated":history["epsilon"][-1].item(), "X0_r2":X0_r2, "time":history["time"], "num_epochs": len(history["epsilon"]) ,"epsilon":parameters[0],"mu":parameters[1],"rho":parameters[2],"model_name":model_name, "optimizer_name":optimizer_name,"lr":lr,"N":N,"T":T,"edge_per_t":edge_per_t,"evidences_per_t":evidences_per_t})
        
    return simulations_results
        

def simulate_BC(N,T,edge_per_t,evidences_per_t,parameters, model = "simple"):
    if model == "simple":
        simulator = sod.BC_simulator()
    if model == "backfire":
        simulator = sod.BC_simulator_backfire()
    simulator.initialize_simulator(N,T,edge_per_t, evidences_per_t)
    
    X, edges, evidences = simulator.simulate_trajectory(parameters)
    
    return X, edges, evidences
    







