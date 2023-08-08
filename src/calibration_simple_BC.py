import torch
import numpy as np
from scipy.special import expit
from calibrator_blackit import Calibrator #i just commented few lines of code to remove the output print
from black_it.samplers.halton import HaltonSampler
from black_it.loss_functions.msm import MethodOfMomentsLoss
import pandas as pd
import matplotlib.pyplot as plt
from time import time

#define a simulator class for simulating the BC model knowing the real edges and opinions
#the calibration will compare the arrays of signs: self.s and simulate_s([epsilon])
#i input theta = [epsilon], instead of epsilon, because the Calibrator of black-it is designed for calibrating a list of parameters
class BC_simulator_given_previous_time():
    def __init__(self, X, edges, mu, real_epsilon, rho = 100, seed = 1):
        self.X = X
        self.edges = edges
        self.mu = mu
        self.real_epsilon = real_epsilon
        self.rho = rho
        self.seed = seed
        self.s = np.concatenate(np.array(edges[:,:,2]))[:, None] #Calibrator requires a time series of dim N, n_of_parameters
        self.mean_s = np.array(edges[:,:,2]).mean(axis = 1)[:,None]
    
    def simulate_s(self, theta, T = 200, seed = None):
        epsilon, = theta
        
        s_pred = np.array([])
        
        if seed != None:
            np.random.seed(self.seed)
        
        T, edge_per_t, _ = self.edges.size()
        
        for t in range(T):
            for new_edge in range(edge_per_t):
                
                u,v,s = self.edges[t, new_edge]
                dist = np.abs(self.X[t, u] - self.X[t, v])
                
                if np.random.random() < expit(self.rho * (epsilon - dist)):
                    s_pred = np.append(s_pred, 1)
                else:
                    s_pred = np.append(s_pred, 0)
        
        
        return s_pred[:, None] #Calibrator requires a time series of dim N, n_of_parameters
    
    def simulate_mean_s(self, theta, T = 200, seed = None):
        epsilon, = theta
        
        mean_s_pred = np.array([])
        
        if seed != None:
            np.random.seed(self.seed)
        
        T, edge_per_t, _ = self.edges.size()
        
        for t in range(T):
            s_t = np.array([])
            for new_edge in range(edge_per_t):
                
                u,v,s = self.edges[t, new_edge]
                dist = np.abs(self.X[t, u] - self.X[t, v])
                
                if np.random.random() < expit(self.rho * (epsilon - dist)):
                    s_t = np.append(s_t, 1)
                else:
                    s_t = np.append(s_t, 0)
                    
            mean_s_pred = np.append(mean_s_pred, np.mean(s_t))
        
        
        return mean_s_pred[:, None] #Calibrator requires a time series of dim N, n_of_parameters




#this function return the simulated list of edges, from X, edges, and epsilon
#it simulated edges[t]|X[t],u,v
#so it knows the picked nodes and their opinions
def simulate(X, edges, epsilon, rho = 100, seed = 1):
    edges_pred = edges.clone()
    
    np.random.seed(seed)
    
    T, edge_per_t, _ = edges.size()
    
    for t in range(T - 1):
        for new_edge in range(edge_per_t):

            u,v,s = edges[t, new_edge]
            dist = np.abs(X[t, u] - X[t, v])

            if np.random.random() < expit(rho * (epsilon - dist)):
                edges_pred[t][new_edge][2] = torch.tensor(1)
            else:
                edges_pred[t][new_edge][2] = torch.tensor(0)
                

    return edges_pred


def define_calibrator_BC(simulator_BC, samplers = [HaltonSampler(1)], loss = MethodOfMomentsLoss(),
                         epsilon_bounds = [[0.], [0.5]],precisions = [0.0001], compare_all_s = False):
    if compare_all_s:
        model_simulator_bc = simulator_BC.simulate_s
        real_data_s = simulator_BC.s
    else:
        model_simulator_bc = simulator_BC.simulate_mean_s
        real_data_s = simulator_BC.mean_s
    BC_cal = Calibrator(real_data = real_data_s,
                        samplers = samplers,
                        loss_function = loss,
                        model = model_simulator_bc,
                        parameters_bounds = epsilon_bounds,
                        parameters_precision = precisions,
                        ensemble_size = 3,
                        saving_folder = None,
                        verbose = False,
                        n_jobs = 1#'initial_state'
                       )
       
    return BC_cal

def calibrate_epsilon_BC(simulator_BC, plot_loss = True, calibration_batches = 100, 
                         sampler = HaltonSampler, loss = MethodOfMomentsLoss(), batch_size = 6, compare_all_s = False,
                         epsilon_bounds = [[0.], [0.5]], precisions = [0.0001], return_time = False):
    t0 = time()
    BC_cal = define_calibrator_BC(simulator_BC, samplers = [sampler(batch_size)], loss = loss,
                         epsilon_bounds = epsilon_bounds, precisions = precisions, compare_all_s = compare_all_s)
    
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
        
    
    
    









