# the method NodesInteractor is used to efficiently compute the probability of positive interaction for all the pairs of nodes, at each time.
# P(s|uv, epsilon, X) = sigma(rho * (epsilon - |X[t,u] - X[t,v]|)
#Instead of computing diff_X[t] = X[t,u] - X[t,v] for every t,
# - Compute diff_X[0]
# - Compute kappa = sigma(rho * (epsilon - |diff_X[0]|)
# - at each t > 0:
#    * Copy, kappa[t] = kappa[t - 1]
#    * Update only the rows and columns that have been modified (the nodes that had a positive interaction)

import torch

class NodesInteractor:
    def __init__(self, X0, epsilon, mu, rho = 70):
        self.rho = rho
        self.epsilon = epsilon.detach()
        self.mu = mu
        self.kappa = torch.sigmoid(rho * (epsilon - torch.abs(X0[None,:] - X0[:,None]))).detach()
    
    def update_interactions_probabilities(self, edges_t, X_t):
        positive_edges = edges_t[edges_t[:,2] == 1]
        u,v,s = positive_edges.T
        prev_X = X_t.detach().clone()
        new_X = X_t.detach().clone()
        new_X[u] += self.mu * (prev_X[v] - prev_X[u])
        new_X[v] += self.mu * (prev_X[u] - prev_X[v])
        
        all_uv = torch.cat([u,v])
        
        new_kappa = torch.sigmoid(self.rho * (self.epsilon - torch.abs(new_X[all_uv,None] - new_X[None,:])))
        
        self.kappa[all_uv,:] = new_kappa
        self.kappa[:,all_uv] = new_kappa.T