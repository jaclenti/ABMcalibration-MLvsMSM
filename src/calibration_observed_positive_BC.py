import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path += ["../src"]
import simulator_BC as sim_bc
import estimation_epsilon_BC as est_bc
import jax_estimation_BC as jest_bc
from calibrator_blackit import Calibrator #i just commented few lines of code to remove the output print
#from black_it.calibrator import Calibrator
import pandas as pd
from black_it.samplers.halton import HaltonSampler
from black_it.samplers.random_forest import RandomForestSampler
from black_it.samplers.best_batch import BestBatchSampler
from black_it.loss_functions.msm import MethodOfMomentsLoss
from black_it.samplers.random_uniform import RandomUniformSampler
from typing import Sequence
import os
import contextlib
from scipy.special import expit
import torch
import importlib
from time import time


#define a simulator class for simulating the BC model knowing the real edges and opinions
#the calibration will compare the arrays of signs: self.s and simulate_s([epsilon])
#i input theta = [epsilon], instead of epsilon, because the Calibrator of black-it is designed for calibrating a list of parameters
class BC_observe_positive_simulator_given_previous_time():
    def __init__(self, X, edges, mu, real_epsilon, rho = 100, seed = 1):
        if type(X) == torch.Tensor:
            X = X.numpy()
            edges = sim_bc.convert_edges_uvst(edges).T.numpy()
        self.T, self.N = X.shape
        self.edge_per_t = max(np.unique(edges[:,3], return_counts = True)[1])
        self.X = X
        self.edges = edges
        self.mu = mu
        self.real_epsilon = real_epsilon
        self.rho = rho
        self.seed = seed
        mean_s = np.array(pd.DataFrame(self.edges, columns = ["u", "v", "s", "t"]).groupby("t").mean()["s"])
        self.mean_s = mean_s[:,None]
        #self.mean_s = np.array(edges[:,:,2]).mean(axis = 1)[:,None]
    
    
    #this function return the simulated list of edges, from X, edges, and epsilon
    #it simulated edges[t]|X[t],u,v
    #so it knows the picked nodes and their opinions
    def simulate_edges(self, epsilon):
        edges_pred = []

        for t in range(self.T - 1):

            edges_t = self.edges[self.edges[:,3] == t]
            positive_t = len(edges_t)
            for edge in edges_t:
                u,v,s,_ = edge
                dist = np.abs(self.X[t, u] - self.X[t, v])            
                s = int(np.random.random() < expit(self.rho * (epsilon - dist)))
                edges_pred.append([u,v,s,t])
            for edge in range(self.edge_per_t - positive_t):
                u,v = np.random.choice(self.N, 2, replace = False)
                dist = np.abs(self.X[t, u] - self.X[t, v])
                s = int(np.random.random() < expit(self.rho * (epsilon - dist)))
                edges_pred.append([u,v,s,t])                

        return np.array(edges_pred)

    
    def simulate_mean_s(self, theta, _ = None, seed = None):
        epsilon, = theta
        
        if seed != None:
            np.random.seed(self.seed)
        
        pred_edges = self.simulate_edges(epsilon)
        mean_s_pred = np.array(pd.DataFrame(pred_edges, columns = ["u", "v", "s", "t"]).groupby("t").mean()["s"])
        
        return mean_s_pred[:, None] #Calibrator requires a time series of dim N, n_of_parameters"""
    
    

def calibrate_epsilon_BC(simulator_BC, plot_loss = True, calibration_batches = 100, 
                         sampler = HaltonSampler, loss = MethodOfMomentsLoss(), batch_size = 1,
                         epsilon_bounds = [[0.], [0.5]], precisions = [0.0001], return_time = False):
    t0 = time()
    
    
    real_data_s = simulator_BC.mean_s
    model_simulator_bc = simulator_BC.simulate_mean_s
    
    BC_cal = Calibrator(real_data = real_data_s,
                        samplers = [sampler(batch_size)],
                        loss_function = loss,
                        model = model_simulator_bc,
                        parameters_bounds = epsilon_bounds,
                        parameters_precision = precisions,
                        ensemble_size = 3,
                        saving_folder = None,
                        verbose = False,
                        n_jobs = 1#'initial_state'
                       )
    
    epsilon_BCcal, loss_BCcal = BC_cal.calibrate(calibration_batches)
    loss_epsilon_df = pd.DataFrame([{"epsilon": epsilon_BCcal[k,0], "loss": loss_BCcal[k]} for k in range(len(loss_BCcal))]).sort_values("epsilon").set_index("epsilon")
    
    
    if plot_loss:
        fig, ax = plt.subplots(nrows = 1, ncols = 1)
        ax.plot(loss_epsilon_df)
        ax.vlines(x = simulator_BC.real_epsilon, ymin = -0.5, ymax = loss_epsilon_df.max(), colors = "red")

    estimated_epsilon = loss_epsilon_df.sort_values("loss").index[0]
    t1 = time()
    calibration = {"loss": loss_epsilon_df, "epsilon": estimated_epsilon}
    
    if return_time:
        calibration["time"] = t1-t0
    
    return calibration
        
   