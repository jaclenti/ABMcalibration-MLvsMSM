import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from scipy.optimize import minimize, Bounds
from scipy.special import expit as sigmoid
from scipy.special import logit
import seaborn as sns
import matplotlib.pyplot as plt
from simulator_BC import simulator, simulator_stoch, convert_edges_uvst, differences_tensor
import pandas as pd
from tqdm import tqdm
from time import time


class BC_PGABM_forward_epsilon(nn.Module): #PyTorch model for gradient optimization, for estimating epsilon observing edges and x0
    
    def __init__(self, epsilon0 = 0.1):
        
        super().__init__()
        theta = torch.tensor([logit(epsilon0)], requires_grad = True) # initialize weights with epsilon0
        self.theta = nn.Parameter(theta)     #define the parameters of the model
    
    def forward(self):
        epsilon = torch.sigmoid(self.theta) #at each step clip epsilon in the interval [0,1]
        return epsilon
        
class BC_PGABM_forward_kappa(nn.Module): #PyTorch model for gradient optimization, for estimating epsilon observing edges and x0
    
    def __init__(self, epsilon0 = 0.1, rho = 10, mu = 0.1):
        
        super().__init__()
        theta = torch.tensor([logit(epsilon0)], requires_grad = True) # initialize weights with epsilon0
        rho = torch.tensor([rho], requires_grad = False) # initialize weights with epsilon0
        mu = torch.tensor([mu], requires_grad = False) # initialize weights with epsilon0
        self.theta = nn.Parameter(theta)     #define the parameters of the model
        self.rho = rho
        self.mu = mu
    
    def forward(self, diff_X):
        epsilon = torch.sigmoid(self.theta) #at each step clip epsilon in the interval [0,1]
        kappa = torch.sigmoid(self.rho * (epsilon - torch.abs(diff_X))) #probability of observing an edge, as the sigmoid of the absolute value of the difference of opinions    
        return kappa[u,v]
    

class EarlyStopping:
    def __init__(self, patience = 5, min_delta = 0, min_epochs = 100):

        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False
        self.min_epochs = min_epochs

    def __call__(self, previous_epsilon, current_epsilon, epoch):
        if self.min_delta > 0:
            if (epoch > self.min_epochs) & (np.abs(previous_epsilon - current_epsilon) < self.min_delta):
                self.counter +=1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.counter = 0

class RandomizeEpsilon:
    def __init__(self, patience = 5, min_delta = 0, max_epochs = 20):
        
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.max_epochs = max_epochs
        self.change_epsilon0 = False

    def __call__(self, previous_epsilon, current_epsilon, epoch):
        if self.min_delta > 0:
            if (epoch < self.max_epochs) & (np.abs(previous_epsilon - current_epsilon) < self.min_delta):
                self.counter +=1
                if self.counter >= self.patience:
                    self.change_epsilon0 = True
                else:
                    self.change_epsilon0 = False
       
        
    
#compute the negative log likelihood for all the edges
#as the binary cross entropy between the observed edges and the probabilities assigned by the sigmoid        
def tot_neg_log_likelihood(uvst, rho, epsilon, T, diff_X, loss = nn.BCEWithLogitsLoss(reduction = "sum")):
    #transform a tensor of dimension (T, 3, edge_per_t) into a tensor of dimensions (4, T*edge_per_t)
    #instead of having edges[t][0] = [u,v,s], we want edges[0] = [u,v,s,t]
    u,v,s,t = uvst
    #define the tensor of the matrices of the differences between opinions. diff_X[t,i,j] = X[t,i] - X[t,j]
    kappa = torch.sigmoid(rho * (epsilon - torch.abs(diff_X))) #probability of observing an edge, as the sigmoid of the absolute value of the difference of opinions

    return loss(kappa, s.double()) #return the BCE of probability of observing edges and s (sign of the edges)

def choose_optimizer(optimizer_name, lr, model):
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr = lr) #define the optimizer with the input learning rate
    if optimizer_name == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr = lr, momentum = 0.9) #define the optimizer with the input learning rate
    if optimizer_name == "nadam":
        optimizer = torch.optim.NAdam(model.parameters(), lr = lr) #define the optimizer with the input learning rate        
    if optimizer_name == "adagrad":
        optimizer = torch.optim.Adagrad(model.parameters(), lr = lr) #define the optimizer with the input learning rate
    return optimizer

def choose_model(forward, epsilon0, rho, mu):
    if forward == "epsilon":
        model_class = BC_PGABM_forward_epsilon
        model = model_class(epsilon0)
        
    if forward == "kappa":
        model_class = BC_PGABM_forward_kappa
        model = model_class(epsilon0, rho, mu)
    
    return model
    

#full optimization procedure
#X must be a torch tensor
#rho is the parameter of the sigmoid
#epsilon0 initialization of epsilon
#t_end is the last time to be considered for the trajectories
#lr is the learning rate
def estimation_epsilon_torch(rho, mu, X, edges, epsilon0,
                             num_epochs = 50, optimizer_name = "adam", lr = 0.01, forward = "epsilon", batch_size = 0, hide_progress = False, min_delta = 0, patience = 5, bce = nn.BCEWithLogitsLoss(reduction = "sum"), return_time = False, min_epochs = 20):
    t0 = time()
    history = {"epsilon": [epsilon0], "loss": [], "grad": []} #return the lists of the epsilon estimates and losses 

    T, N = X.size()
    _, edge_per_t, _ = edges.size()
    #for each epoch pick a sample of times to speed up the optimization (we can do it because we know X for each time)
    uvst = convert_edges_uvst(edges)
    u,v,s,t = uvst
    
    restarts = 5
    restart_loop = True
    
    while restart_loop & (restarts > 0):
        restarts -= 1
        restart_loop = False
        early_stopping = EarlyStopping(patience, min_delta, min_epochs)
        randomize_epsilon = RandomizeEpsilon(patience, min_delta)
        model = choose_model(forward, epsilon0, rho, mu) #the model can forward epsilon or kappa (the probability of interaction)
        optimizer = choose_optimizer(optimizer_name, lr, model) #input the optimizer and learning rate, among SGD, adam, adagrad, nadam
        diff_X = X[t,u] - X[t,v]
        #diff_X = differences_tensor(X, N, T)[t,u,v]

        for epoch in tqdm(range(num_epochs), disable = hide_progress):
            if batch_size < 1:
                batch_indices = np.arange(uvst.size()[1])
            else:
                batch_indices = np.random.choice(np.arange(uvst.size()[1]), batch_size)

            batch_uvst = uvst[:,batch_indices]
            batch_u, batch_v, batch_s, batch_t = batch_uvst

            if forward == "epsilon":
                epsilon = model()
                loss = tot_neg_log_likelihood(batch_uvst, rho, epsilon, T, diff_X[batch_indices], loss = bce)
            if forward == "kappa":
                s_pred = model(diff_X[batch_indices])
                loss = bce(s_pred, batch_s.double())

            loss.backward()
            optimizer.step()


            history["epsilon"].append(sigmoid(model.theta.item()))
            history["loss"].append(loss.item())
            history["grad"].append(model.theta.grad.item())

            # early stopping

            randomize_epsilon(history["epsilon"][-1], history["epsilon"][-2], epoch)
            if randomize_epsilon.change_epsilon0:
                epsilon0 = np.random.random()
                #restart_loop = True
                #break
                
            early_stopping(history["epsilon"][-1], history["epsilon"][-2], epoch)
            if early_stopping.early_stop:
                break

            
            optimizer.zero_grad()
        
    t1 = time()
    if return_time:
        history["time"] = t1 - t0
    return history


def losses_closure(edges, X, rho, epsilon_values = np.arange(0,1,0.05), loss = nn.BCEWithLogitsLoss(reduction = "sum")):
    T, N = X.size()
    _, edge_per_t, _ = edges.size()
    
    uvst = convert_edges_uvst(edges)
    u,v,s,t = uvst
    diff_X = X[t,u] - X[t,v]
    #diff_X = differences_tensor(X, N, T)[t,u,v]

    loss_epsilon = lambda x: tot_neg_log_likelihood(uvst, rho, x, T, diff_X, loss = loss)
    
    losses = [(x, loss_epsilon(x)) for x in epsilon_values]
    return np.array(losses).T


def print_history(history, epsilon, max_d = None, min_d = None):
    fig, (ax1, ax2, ax3) = plt.subplots(nrows = 1, ncols = 3, figsize = (12, 3))
    ax1.plot(history["epsilon"])
    
    ax2.plot(history["loss"])
    ax3.plot(history["grad"])
    
    ax1.hlines(epsilon, xmin = 0, xmax = len(history["loss"]), colors = "red", lw = 0.8)
    
    if max_d != None:
        ax1.hlines(max_d, xmin = 0, xmax = len(history["loss"]), colors = "red", linestyle = "dashed", lw = 0.5)
    if min_d != None:
        ax1.hlines(min_d, xmin = 0, xmax = len(history["loss"]), colors = "red", linestyle = "dashed", lw = 0.5)
    
    ax1.set_title("epsilon")
    ax2.set_title("loss")
    ax3.set_title("grad")
    
    fig.show()
    
    
def print_simulation_history(simulated_estimations, n):
    print_history(simulated_estimations[n]["history"], simulated_estimations[n]["epsilon"], 
                 simulated_estimations[n]["max_d"],simulated_estimations[n]["min_d"],)
    
    
          

def estimation_epsilon_scipy(rho, X, edges, epsilon0, bce = nn.BCEWithLogitsLoss(reduction = "sum")):
    T, N = X.size()
    
    uvst = convert_edges_uvst(edges)
    u,v,s,t = uvst
    
    diff_X = X[t,u] - X[t,v]
    #diff_X = differences_tensor(X, N, T)[t,u,v]

    loss_epsilon = lambda x: tot_neg_log_likelihood(uvst, rho, x[0], T, diff_X, bce)
    estimation_epsilon = minimize(loss_epsilon, epsilon0, bounds = Bounds(lb = 0.0, ub = 1.0))["x"]
    return estimation_epsilon[0]                   
  