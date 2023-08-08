import numpy as np
import sys
sys.path += ["/home/jacopo.lenti/Projects/learning-od-step-by-step/src"]
import simulator_BC_coo as sim_coo
from calibrator_blackit import Calibrator #i just commented few lines of code to remove the output print
#from black_it.calibrator import Calibrator
from black_it.samplers.halton import HaltonSampler
from black_it.loss_functions.msm import MethodOfMomentsLoss
from typing import Sequence
import os
import pandas as pd
import contextlib
import torch
from tqdm import tqdm
from time import time
from scipy.special import expit as sigmoid


def calibrate_params_ABM(simulator, parameters_names = ["epsilon"], calibration_batches = 100, ensemble_size = 3,
                         sampler = HaltonSampler, loss = MethodOfMomentsLoss(), batch_size = 1,
                         parameters_bounds = [[0.], [0.5]], precisions = [0.00001]):
    
    t0 = time()
    model = simulator.simulate_ts
    real_data = simulator.ts
    
    samplers = [sampler(batch_size)]


    BC_cal = Calibrator(real_data = real_data,
                        samplers = samplers,
                        loss_function = loss,
                        model = model,
                        parameters_bounds = parameters_bounds,
                        parameters_precision = precisions,
                        ensemble_size = ensemble_size,
                        saving_folder = None,
                        verbose = False,
                        n_jobs = 1#'initial_state'
                       )
    
    
    params, losses = BC_cal.calibrate(calibration_batches)
    t1 = time()
    
    output_dict = {"loss": losses}
    for i,param_i in enumerate(parameters_names):
        output_dict[param_i] = params.T[i]
        
    return pd.DataFrame(output_dict), t1 - t0

