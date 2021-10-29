import numpy as np
import scipy.optimize as sopt
from scipy.optimize import leastsq


# Chi-Square Test
def chiSqrTerm(w,std, sim_map_m, exp_map):
    v = (np.sum(sim_map_m.T*w,1) - exp_map)/std
    chiSqr = 0.5 * np.sum(v*v)
    return chiSqr


# Function which reads sim data, scale them based on sf and calls ChiSqr to compare them to experimental values
def coeff_fit(sf, w_opt, std, sim_map_m, exp_map):
    sim_map_sf = sim_map_m * sf
    return chiSqrTerm(w_opt,std, sim_map_sf, exp_map)



# Returns proper weights from log-weights after normalisation
def getWeights(g):
    tmp = np.exp(g)
    s = tmp.sum()
    w = np.array(tmp / s)
    return w,s


# Log Prior base funtion
# Log Prior in the log-weights representation:
# theta * ((g.T * w) - (g0.T * w) + np.log(s0) - np.log(s))

def bioen_log_prior(w, s, g, g0, theta):
    w0,s0 = getWeights(g0)
    g_ave = np.sum(g * w)
    g0_ave = np.sum(g0 * w)
    log_prior = theta * (g_ave - g0_ave + np.log(s0) - np.log(s))
    return log_prior     


# Log Posterior base function

def bioen_log_posterior_base(g, g0, std, sim_map_m, exp_map, theta):
    w, s = getWeights(g)
    log_prior = bioen_log_prior(w, s, g, g0, theta)
    chiSqr = chiSqrTerm(w, std, sim_map_m, exp_map)
    log_posterior = chiSqr + log_prior
    return log_posterior

# Log Posterior base function with DIRICHLET
def bioen_log_posterior_dirichlet(g, g0, sim_map_m, exp_map, theta):
    w, s = getWeights(g)
    log_prior = bioen_log_prior_dirichlet(w, s, g, g0, theta)
    chiSqr = chiSqrTerm(w, sim_map_m, exp_map)
    log_posterior = chiSqr + log_prior
    return log_posterior


# Gradient of log Posterior base function in the log-weights representation

def grad_bioen_log_posterior_base(g, g0, std, sim_map_m, exp_map, theta):
    w, s = getWeights(g)
    tmp = np.zeros(w.shape[0])
    sim_ave = np.sum(sim_map_m.T * w,1).T
    for mu in range(w.shape[0]):
        w_mu = w[mu]
        diff1 = (sim_ave - exp_map)/std
        sim_mu = sim_map_m[mu]
        diff2 = (sim_mu - sim_ave)/std
        tmp[mu] = w_mu * np.sum(diff1 * diff2)
    gradient = w * theta * (g - np.sum(g*w) - g0 + np.sum(g0*w)) + tmp
    return gradient
