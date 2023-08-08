import torch
import numpy as np
import pandas as pd
import sys
import torch.nn as nn
sys.path += ["../src"]
from scipy.special import expit as sigmoid
from scipy.special import logit
import simulator_BC_coo as sim_coo
from simulator_BC_coo import opinion_update
from initialize_model import EarlyStopping,RandomizeEpsilon,choose_optimizer
from time import time
from tqdm import tqdm




class BC_evidences_X(nn.Module): #PyTorch model for gradient optimization, for estimating epsilon observing edges and x0
    
    def __init__(self, N, epsilon_0, mu, edges, rho):#, evidences_indices, evidences_opinions):
        
        super().__init__()
        
        self.edges = edges
        self.mu = mu
        self.rho = rho
        T_, self.edge_per_t, _ = edges.shape
        self.T = T_ + 1
        self.N = N
        self.u, self.v, s, self.t = uvst = sim_coo.convert_edges_uvst(edges.long())
        self.s = s.double()  #use s.float() for computing the BCE
        theta = torch.tensor([logit(epsilon_0)], requires_grad = True) # initialize weights with siepsilon0
        X0 = torch.rand(N, dtype = torch.float64, requires_grad = True) # random initialization of the opinions
        
        self.logit_X0 = nn.Parameter(torch.logit(X0))   #define the parameters of the model
        self.theta = nn.Parameter(theta)     #define the parameters of the model
        
    
    def forward(self):
        epsilon = torch.sigmoid(self.theta) #at each step clip epsilon in the interval [0,1]
        X0 = torch.sigmoid(self.logit_X0)   #at each step clip X0 in the interval [0,1]
        
        X = X_from_X0(X0, self.mu, self.edges, self.T, self.edge_per_t, self.N) # compute all X with current estimate of X0
        
        diff_X = X[self.t,self.u] - X[self.t,self.v] 
        kappa = kappa_from_epsilon(epsilon, diff_X, self.rho) # compute probability of interaction with current estimate of epsilon
        
        return X, kappa


#given mu, edges and X0, compute all X (that is a deterministic function of these variables)
def X_from_X0(X0, mu, edges, T, edge_per_t, N):
    X_t = X0.clone()
    
    
    X = []
    X.append(X_t)
    for t in range(T - 1):
        
        u,v,s = edges[t].long().T   # interacting edges and signs of interactions at time t
        
        #create a sparse_coo tensor for copmuting all the differences of the opinions of the interacting nodes
        #it will have dimension (N, edge_per_t), so that i store in each row one pair of nodes and the differences
        #of their opinions
        #these are multiplied by s, because if sign is 0, then the opinions are not updated
        diff_X_sparse = torch.sparse_coo_tensor(torch.cat([u[None, :],
                                                           torch.arange(edge_per_t)[None, :]], dim = 0), 
                                                - (X_t[u] - X_t[v]) * s, size = [N, edge_per_t]) +\
        torch.sparse_coo_tensor(torch.cat([v[None, :], 
                                           torch.arange(edge_per_t)[None, :]], dim = 0),
                            (X_t[u] - X_t[v]) * s, size = [N, edge_per_t])
        #multiply the differences by mu
        update = mu * (torch.sparse.sum(diff_X_sparse, dim = 1).to_dense())
        
        X_t = X_t.clone() + update   #update X_t
        X.append(X_t)    #create a tensor with X_t for each t
        
    
    return torch.cat([x[:, None] for x in X], dim = 1).T




#compute the loss of a single simulation, given the estimates of X0 and epsilon
def losses_one_simulation(X0, edges, evidences, mu, epsilon, rho = 70):
    T = len(evidences)
    N = len(X0)
    evidences_per_t = len(evidences[0][0])
    u,v,s,t = uvst = sim_coo.convert_edges_uvst(edges.long())
    s = s.float()
    
    X = X_from_X0(X0, mu, edges, T, edge_per_t, N).float()
    diff_X = X[t,u] - X[t,v]
    kappa = kappa_from_epsilon(epsilon, diff_X, rho)
    evidences_indices = torch.cat([evidences[k][0][None,:]
                                   for k in range(len(evidences))], dim = 0)
    evidences_opinions = torch.cat([evidences[k][1][None,:] 
                                    for k in range(len(evidences))], dim = 0).reshape(T * evidences_per_t)

    
    loss_edges = log_likelihood_edges(kappa, s)
    loss_evidences = log_likelihood_evidences(evidences_indices, evidences_opinions, X, T)
    
    return loss_edges.item(), loss_evidences.item()

def kappa_from_epsilon(epsilon, diff_X, rho):
    
    return torch.sigmoid(rho * (epsilon - torch.abs(diff_X)))




#compute the likelihood BCE between the signs of the interactions and the current estimate of the probability of interaction
def log_likelihood_edges(kappa, s, loss = nn.BCELoss(reduction = "sum")):
    
    sigmoid_kappa = torch.sigmoid(kappa)
    #l = -torch.sum(torch.log(s * kappa + (1 - s) * (1 - kappa)))
    l = -torch.sum(torch.log(s * sigmoid_kappa + (1 - s) * (1 - sigmoid_kappa)))
    #l = loss(kappa, s)
    
    return l #loss(kappa, s)




#compute the likelihood BCE between the signs of the evidences of the opinions and the current estimate of the opinions

def log_likelihood_evidences(evidences_indices, evidences_opinions, X, T, 
                             loss = nn.BCELoss(reduction = "sum")):
    
    real_opinions = torch.cat([X[t, evidences_indices[t]] for t in range(T)])
    #sigmoid_real_opinions = torch.sigmoid(real_opinions)
    
    l = -torch.sum(torch.log(evidences_opinions * real_opinions + (1 - evidences_opinions) * (1 - real_opinions)))
    #l = -torch.sum(torch.log(evidences_opinions * sigmoid_real_opinions + (1 - evidences_opinions) * (1 - sigmoid_real_opinions)))
    #l = loss(evidences_opinions.float(), real_opinions)
    
    return l #loss(evidences_opinions, real_opinions)



def estimation_epsilon_torch(X_, edges, evidences, mu, epsilon_0 = 0.25, rho = 70,
                             num_epochs = 50, optimizer_name = "adam", 
                             lr = 0.01, hide_progress = False, 
                             min_delta = 1e-5, patience = 5, return_time = False, min_epochs = 20,
                             step = "loss", multiple_restarts = 1, verbose = True
                            ):
    
    for _ in range(multiple_restarts): #use multiple restarts for choosing the best estimate 
        #print(f"Restart {_}")
        t0 = time()
        history = {"epsilon": [], "loss": [], "grad": [], "loss evidences": [], "loss edges": [], "X0": []} #return the lists of the epsilon estimates and losses 
        
        T, N = X_.size()

        _, edge_per_t, _ = edges.size()
        evidences_per_t = len(evidences[0][0])

        evidences_indices = torch.cat([evidences[k][0][None,:] for k in range(len(evidences))], dim = 0) #tensor with the indices of the users of which we know the evidence
        evidences_opinions = torch.cat([evidences[k][1][None,:] for k in range(len(evidences))], dim = 0).reshape(T * evidences_per_t).double() #only the evidences of these users
        


        early_stopping = EarlyStopping(patience, min_delta, min_epochs) #method for doing early stopping: after patience steps with less update than min_delta stop the estimation
        
        model_class = BC_evidences_X
        model = model_class(N, epsilon_0, mu, edges, rho)

        
        u,v,s,t = uvst = sim_coo.convert_edges_uvst(edges.long())
        s = s.double()
        
        

        if step == "parameters": # this is used if we want to optimize the two parameters separately 
            optimizer_epsilon = torch.optim.Adam([model.theta], lr = lr)
            optimizer_X0 = torch.optim.Adam([model.logit_X0], lr = lr)
        else: # this is used if we optimize the two parameters together
            optimizer = choose_optimizer(optimizer_name, lr, model) #input the optimizer and learning rate, among SGD, adam, adagrad, nadam



        for epoch in tqdm(range(num_epochs), disable = hide_progress):
            if step == "loss": #at each poch optimize the loss on the evidences, then the loss on the edges
                X, kappa = model()
                loss = loss_edges = log_likelihood_edges(kappa, s)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                X, kappa = model()
                loss = loss_evidences = log_likelihood_evidences(evidences_indices, evidences_opinions, X, T)
                
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            
            elif step == "parameters": #at each epoch optimize the loss wrt X0, then wrt epsilon
                for opt in [optimizer_X0, optimizer_epsilon]:
                    X, kappa = model()
                    loss_edges, loss_evidences = log_likelihood_edges(kappa, s), log_likelihood_evidences(evidences_indices, evidences_opinions, X, T)
                    loss = loss_edges + loss_evidences
                    loss.backward()
                    opt.step()
                    opt.zero_grad()
                    
            else: #optimize the two parameters together
                X, kappa = model()
                loss_edges, loss_evidences = log_likelihood_edges(kappa, s), log_likelihood_evidences(evidences_indices, evidences_opinions, X, T)

                loss = loss_edges + loss_evidences

                loss.backward()
                optimizer.step()

                optimizer.zero_grad()



            epsilon_ = sigmoid(model.theta.item())
            
            history["epsilon"].append(epsilon_)
            history["X0"].append(torch.sigmoid(model.logit_X0).detach())
            history["loss evidences"].append(loss_evidences.item())
            history["loss edges"].append(loss_edges.item())
            history["loss"].append(loss.item())
            #history["grad"].append(model.theta.grad.item())
            
            if verbose:
                if epoch % 20 == 0:
                    print(f"{epoch}/{num_epochs} Log-likelihood edges {round(loss_edges.item(),1)}")
                    print(f"{epoch}/{num_epochs} Log-likelihood evidences {round(loss_evidences.item(),1)}")
                    print(f"{epoch}/{num_epochs} epsilon {round(epsilon_.item(), 3)}")
                    print("")
            
            #early stopping
            if epoch > 5:
                early_stopping(history["epsilon"][-1], history["epsilon"][-2], epoch)
                if early_stopping.early_stop:
                    break
        
        
        t1 = time()
       
    return t1 - t0, history









