import torch
import torch.nn as nn
import numpy as np
from scipy.special import expit as sigmoid
import sys
sys.path += ['../src/']
import simulator_BC as sim_bc
import estimation_epsilon_BC as est_bc
import jax_estimation_BC as jest_bc
import jax.numpy as jnp
from importlib import reload
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from time import time
import jax
import jax.scipy.optimize as joptimize


#return only the positive edges (the ones that we observe in this case) from a np.array(u,v,s,t) format
def observe_positive(edges):
    return edges[edges[:,2] == 1]


#the likelihood function can be written as sum of two parts:
#1) for the positive interactions, sigmoid(epsilon - delta_x) for delta_x of each pair of interacting nodes
#2) for the negative interactions, sigmoid(epsilon - delta_x) for delta_x of all pairs nodes, since we do not know whic ones did interact

def tot_log_likelihood_oserve_only_positive(X, positive_edges, T, N, edge_per_t, epsilon, rho):

    u,v,s,t = positive_edges.T
    diff_X = np.abs(X[t,u] - X[t,v])
    
    #observed nodes
    log_likelihood_observed = np.sum(np.log(sigmoid(rho * (epsilon - diff_X))))

    #for each time count how many negative latent interactions occurred (edge_per_t - observed_interactions)
    #and compute the likelihood for all the possible pairs
    log_likelihood_non_observed = 0
    for t in range(T):
        X_t = X[t]
        edges_t = positive_edges[positive_edges[:,3] == t]
        if len(edges_t) < edge_per_t:
            u_diff,v_diff = np.triu_indices(n = len(X_t), k = 1)
            all_diff_X = np.abs(X_t[:,None] - X_t[None,:])[[u_diff,v_diff]]
            #the likelihood of non observing an interaction is observed (edge_per_t - observed_interactions) times
            log_likelihood_non_observed += (edge_per_t - len(edges_t)) * np.log((1 - sigmoid(rho * (epsilon - all_diff_X)).mean()))

    neg_tot_log_likelihood = - log_likelihood_observed - log_likelihood_non_observed
    
    return neg_tot_log_likelihood


def scipy_minimization(X, edges, edge_per_t, T, N, epsilon, rho = 70, x0 = 0.25, return_time = False):
    t0 = time()
    T, N = X.shape
    positive_edges = edges[edges[:,2] == 1]
    loss_closure = lambda x: tot_log_likelihood_oserve_only_positive(X, positive_edges, T, N, edge_per_t, x, rho)
    scipy_estimate = minimize(loss_closure, x0)
    t1 = time()
    if return_time:
        return (scipy_estimate.x, t1 - t0)
    else:
        return scipy_estimate.x



def jax_tot_log_likelihood_oserve_only_positive(X_jax, positive_edges_jax, edge_per_t, jepsilon, rho = 70):
    T, N = X_jax.shape
    u,v,s,t = positive_edges_jax.T
    diff_X = jnp.abs(X_jax[t,u] - X_jax[t,v])

    log_likelihood_observed = jnp.sum(jnp.log(jest_bc.jnp_sigmoid(rho * (jepsilon - diff_X))))

    log_likelihood_non_observed = 0
    for t in range(T):
        edges_t = positive_edges_jax[positive_edges_jax[:,3] == t]
        if len(edges_t) < edge_per_t:
            all_diff_X = jnp.abs(X_jax[:,None] - X_jax[None,:])
            all_diff_X_triu = jnp.array([all_diff_X[u,v] 
                                         for u in range(N)
                                         for v in np.arange(u + 1, N)])
            log_likelihood_non_observed += (edge_per_t - len(edges_t)) * jnp.log((1 - jest_bc.jnp_sigmoid(rho * (jepsilon - all_diff_X_triu)).mean()))

    neg_tot_log_likelihood = - log_likelihood_observed - log_likelihood_non_observed
    
    return neg_tot_log_likelihood


def estimate_jopt_optimize(X_jax, edges_jax, rho = 70, epsilon0 = 0.25, return_time = False):
    t0 = time()
    u,v,s,t = edges_jax.T
    edge_per_t = int(sum(edges_jax[:,3] == 0))
    T = int(max(t)) + 2
    positive_edges_jax = edges_jax[edges_jax[:,2] == 1]
    diff_X_jax = jnp.abs(X_jax[t,u] - X_jax[t,v])
    
    jepsilon0 = jnp.array([epsilon0])
    jtheta0 = jest_bc.jnp_logit(jepsilon0)
    
    #@jit
    def jit_compute_loss(jtheta):
        jepsilon = jest_bc.jnp_sigmoid(jtheta)
        return jax_tot_log_likelihood_oserve_only_positive(X_jax, positive_edges_jax, edge_per_t, jepsilon)
    
    opt_min = joptimize.minimize(jit_compute_loss, jtheta0, method = "BFGS")
    
    t1 = time()
    
    if return_time:
        return jest_bc.jnp_sigmoid(opt_min.x), t1 - t0
    else:
        return jest_bc.jnp_sigmoid(opt_min.x)
