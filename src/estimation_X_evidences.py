import sys
sys.path += ["/home/jacopo.lenti/Projects/learning-od-step-by-step/src"]

import simulator_BC_coo as sim_coo
import estimation_with_edges_and_evidences as ewee


inputs = []
for el in sys.argv[1:]:
    inputs.append(el)
    
N, T, edge_per_t, epsilon, mu = inputs

if __name__ == '__main__':
    X, edges, evidences = sim_coo.simulator_BC_np_coo(N, T, edge_per_t, epsilon, mu,
                                                      with_evidences = False, sum_ab = 1, distribution_evidences = "beta",
                                                      rho = 70, seed = None, X0 = [], evidences_per_t = None, as_torch = True)
    
    
    (time_sim, history) = ewee.estimation_epsilon_torch(X, edges, evidences, mu, epsilon_0 = 0.25, rho = 70,
                                                        num_epochs = 50, optimizer_name = "adam",
                                                        lr = 0.01, hide_progress = True,
                                                        min_delta = 1e-5, patience = 5, return_time = False, min_epochs = 20,
                                                        step = "loss", multiple_restarts = 1, verbose = False
                                                       )






