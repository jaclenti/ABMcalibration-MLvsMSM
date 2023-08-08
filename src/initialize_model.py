# define method for initializing a model (nn.Module)
#   *  chooose among a set of optimizer (with relative learning rate)
#   *  define the routine for stopping the convergence
#   *  define the routine for restarting the convergence and randomize the initialization (in case of local minima reached early)

import torch
import torch.nn as nn
import numpy as np

def choose_optimizer(optimizer_name, lr, model):
    optimizer_list = ["adam", "SGD", "nadam", "adagrad", "RMSprop"]
    assert optimizer_name in optimizer_list, f"Optimizer must be in {optimizer_list}"
    
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr = lr) #define the optimizer with the input learning rate
    if optimizer_name == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr = lr, momentum = 0.9) #define the optimizer with the input learning rate
    if optimizer_name == "nadam":
        optimizer = torch.optim.NAdam(model.parameters(), lr = lr) #define the optimizer with the input learning rate        
    if optimizer_name == "adagrad":
        optimizer = torch.optim.Adagrad(model.parameters(), lr = lr) #define the optimizer with the input learning rate
    if optimizer_name == "RMSprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr = lr)
    return optimizer

class EarlyStopping:
    def __init__(self, patience = 5, min_delta = 0, min_epochs = 20, long_run_delta = 0, long_run_diff = 10, long_run_patience = 10):

        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.long_run_counter = 0
        self.early_stop = False
        self.min_epochs = min_epochs
        self.long_run_delta = long_run_delta
        self.long_run_patience = long_run_patience
        
    def __call__(self, previous_epsilon, current_epsilon, long_run_previous_epsilon, epoch):
        if self.min_delta > 0:
            if (epoch > self.min_epochs) & (np.abs(previous_epsilon + current_epsilon) < self.min_delta):
                self.counter +=1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.counter = 0

        if self.long_run_delta > 0:
            if (epoch > self.min_epochs) & ((np.abs(previous_epsilon - long_run_previous_epsilon) < self.long_run_delta) | (np.abs(current_epsilon - long_run_previous_epsilon) < self.long_run_delta)):
                self.long_run_counter +=1
                
                if self.long_run_counter >= self.long_run_patience:
                    self.early_stop = True

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
                