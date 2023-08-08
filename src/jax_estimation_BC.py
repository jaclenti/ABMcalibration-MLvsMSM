import jax
from jax import jit
import numpy as np
import jax.numpy as jnp
from scipy.special import expit as sigmoid
from scipy.special import expit as logit
import optax
from time import time
from tqdm import tqdm
import jax.scipy.optimize as joptimize

def jnp_sigmoid(x):
    return 1 / (1 + jnp.exp(-x))

def jnp_logit(x):
    return jnp.log(x) - jnp.log(1 - x)

def jnp_bce_with_logits(x,y):
    xs = jnp_sigmoid(x)
    loss = -jnp.sum(y * jnp.log(xs) + (1-y) * jnp.log(1 - xs))
    return loss



def simulator_stoch_np(N, T, edge_per_t, epsilon, mu, rho = 70, seed = None, x0 = []):
    if seed != None:
        np.random.seed(seed)
    if len(x0) == 0:
        x0 = np.random.uniform(size = [1,N])
    
    edges = []
    t = 0
    x0 = np.random.uniform(size = [1,N])
    
    X = np.repeat(x0, T, axis = 0)
    
    for t in range(T-1):
        X[t+1] = X[t]
        """
        interacting_nodes = []
        for new_edge in range(edge_per_t):
            while True:
                u = np.random.randint(N) #pick a user
                if u not in interacting_nodes:
                    interacting_nodes.append(u)
                    break
            while True:
                v = np.random.randint(N)
                if v not in interacting_nodes:
                    interacting_nodes.append(v)
                    break
        """
        for new_edge in range(edge_per_t):
            u,v = np.random.choice(np.arange(N), size = 2, replace = False)
            
            dist = jnp.abs(X[t,u] - X[t,v])
            
            if np.random.random() < sigmoid(rho * (epsilon - dist)):
                X[t+1, v] += mu * (X[t,u] - X[t,v])
                X[t+1, u] += mu * (X[t,v] - X[t,u])
                edges.append(np.array([u,v,1,t]))
                #edges[t + new_edge] = np.array([u,v,1,t])
            else:
                edges.append(np.array([u,v,0,t]))
                #edges[t + new_edge] = np.array([u,v,0,t])
    return X, np.array(edges)

def simulator_stoch_jnp(N, T, edge_per_t, epsilon, mu, rho = 70, seed = None, x0 = []):
    X, edges = simulator_stoch_np(N, T, edge_per_t, epsilon, mu, rho, seed, x0)
    return jnp.array(X), jnp.array(edges)

def tot_neg_log_likelihood_jnp(edges_jnp, rho, epsilon, T, diff_X_jnp):
    u,v,s,t = edges_jnp.T
    kappa = jnp_sigmoid(rho * (epsilon - jnp.abs(diff_X_jnp))) #probability of observing an edge, as the sigmoid of the absolute value of the difference of opinions

    return jnp_bce_with_logits(kappa, s) #return the BCE of probability of observing edges and s (sign of the edges)


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
                

def estimator_BC_jax(rho, mu, X, edges, epsilon0, patience = 20, min_delta = 10e-6, min_epochs = 100,
                     num_epochs = 1000, lr = 0.0001, return_time = True, hide_progress = False):
    history = {"epsilon": [epsilon0], "loss": [], "grad": []}
    
    t0 = time()
    optimizer = optax.adam(learning_rate = lr)
    
    early_stopping = EarlyStopping(patience, min_delta, min_epochs)
    
    theta0 = jnp_sigmoid(epsilon0)
    
    params = {"theta": jnp.array([theta0])}
    opt_state = optimizer.init(params)
    
    u,v,s,t = edges.T
    T = int(max(t) + 2)
    edge_per_t = int(edges.shape[0] / (T - 1))
    
    diff_X = jnp.abs(X[t,u] - X[t,v])
    
    @jit
    def jit_compute_loss(params):
        return tot_neg_log_likelihood_jnp(edges, rho, jnp_logit(params["theta"]), T, diff_X)

    
    for epoch in tqdm(range(num_epochs), disable = hide_progress):
        grads = jax.grad(jit_compute_loss)(params)
        history["epsilon"].append(jnp_logit(params["theta"]))
        history["loss"].append(jit_compute_loss(params))
        history["grad"].append(grads["theta"])
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        
        
        
        early_stopping(history["epsilon"][-1], history["epsilon"][-2], epoch)
        
        if early_stopping.early_stop:
            break
    t1 = time()            
    if return_time:
        history["time"] = t1 - t0
        
    return history


def estimate_jopt_optimize(X_jax, edges_jax, rho = 70, epsilon0 = 0.25, return_time = False):
    t0 = time()
    u,v,s,t = edges_jax.T
    T = int(max(t)) + 2
    diff_X_jax = jnp.abs(X_jax[t,u] - X_jax[t,v])
    
    jepsilon0 = jnp.array([epsilon0])
    jtheta0 = jnp_logit(jepsilon0)
    
    @jit
    def jit_compute_loss(jtheta):
        jepsilon = jnp_sigmoid(jtheta)
        return tot_neg_log_likelihood_jnp(edges_jax, rho, jepsilon, T, diff_X_jax)
    
    opt_min = joptimize.minimize(jit_compute_loss, jtheta0, method = "BFGS")
    
    t1 = time()
    
    if return_time:
        return jnp_sigmoid(opt_min.x), t1 - t0
    else:
        return jnp_sigmoid(opt_min.x)


