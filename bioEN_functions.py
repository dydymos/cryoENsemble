import numpy as np
import scipy.optimize as sopt
from scipy.optimize import leastsq
from kneed import KneeLocator


def chiSqrTerm(w,std, sim_map_m, exp_map):
    """
    Calculates Chi-Square Test
    """
    v = (np.sum(sim_map_m.T*w,1) - exp_map)/std
    chiSqr = 0.5 * np.sum(v*v)
    return chiSqr



def coeff_fit(sf, w_opt, std, sim_map_m, exp_map):
    """
    Reads simulated maps, scales them based on sf scalling factor and calls chiSqrTerm
    to compare them to experimental values
    """
    sim_map_sf = sim_map_m * sf
    return chiSqrTerm(w_opt,std, sim_map_sf, exp_map)



def getWeights(g):
    """
    Returns proper weights from log-weights after normalisation
    """
    tmp = np.exp(g)
    s = tmp.sum()
    w = np.array(tmp / s)
    return w,s


def bioen_log_prior(w, s, g, g0, theta):
    """
    Log Prior base funtion
    Log Prior in the log-weights representation:
    theta * ((g.T * w) - (g0.T * w) + np.log(s0) - np.log(s))
    """
    w0,s0 = getWeights(g0)
    g_ave = np.sum(g * w)
    g0_ave = np.sum(g0 * w)
    log_prior = theta * (g_ave - g0_ave + np.log(s0) - np.log(s))
    return log_prior


def bioen_log_posterior_base(g, g0, std, sim_map_m, exp_map, theta):
    """
    Log Posterior base function
    """
    w, s = getWeights(g)
    log_prior = bioen_log_prior(w, s, g, g0, theta)
    chiSqr = chiSqrTerm(w, std, sim_map_m, exp_map)
    log_posterior = chiSqr + log_prior
    return log_posterior


def grad_bioen_log_posterior_base(g, g0, std, sim_map_m, exp_map, theta):
    """
    Gradient of Log Posterior base function in the log-weights representation
    """
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

def bioen_init_uniform(N_models):
    """
    Initializes weight, log-weights and parameters
    """
    # Reference weights for models [UNIFORM]
    w0 = np.ones(N_models)/N_models
    # Initial weights for models to start optimization [UNIFORM]
    w_init = np.ones(N_models)/N_models
    # Reference log-weights, with one vaue set to 0
    # Actually for uniform reference weights this sets the whole log-weight vector to zero - is this a problem?
    g0 = np.log(w0)
    g0 -= g0[-1]
    # Log-weights for initialization of the optimization protocol
    g_init = np.log(w_init)
    g_init -= g_init[-1]
    # Initialize log-weights
    g = g_init
    # Initial scalling factor
    sf_init = 1.0
    return w0,w_init,g0,g_init,sf_init

def bioen(sim_em_v_data,exp_em_mask,std,thetas,g0,g_init,sf_start,n_iter,epsilon,pgtol,maxiter):
    """
    "" ITERATIONS THROUGHT Thetas
    """
    w_opt_array = []
    f_min_array = []
    sf_opt_array = []
    for theta in thetas:
        print("THETA = "+str(theta))
        g = g_init
        sim_em_v=sim_em_v_data*sf_start
        sf_opt = sf_start
        for i in range(0,n_iter):
            if (i == 0):
                print("ITERATION "+str(i+1))
                # Getting optimal weight
                res=sopt.fmin_l_bfgs_b(bioen_log_posterior_base,g,args = (g0, std, sim_em_v, exp_em_mask, theta),fprime = grad_bioen_log_posterior_base, epsilon = epsilon, pgtol = pgtol, maxiter = maxiter, disp = False)
                # new weights
                w_opt = getWeights(res[0])[0]
                # final energy
                fmin = res[1]
                fmin_old = fmin
                print("fmin    = ", fmin)
                # Using new weights to get new scalling factor
                sf_ = leastsq(coeff_fit, sf_opt, args=(w_opt,std, sim_em_v_data,exp_em_mask))[0]
                sf_opt = sf_
                # geting new sim data with new nuisance parameter
                sim_em_v = sim_em_v_data * sf_opt
            else:
                print("ITERATION "+str(i+1))
                # Getting optimal weight
                res=sopt.fmin_l_bfgs_b(bioen_log_posterior_base,g,args = (g0, std, sim_em_v, exp_em_mask, theta),fprime = grad_bioen_log_posterior_base, epsilon = epsilon, pgtol = pgtol, maxiter = maxiter, disp = False)
                # new weights
                w_opt = getWeights(res[0])[0]
                # final energy
                fmin = res[1]
                if (fmin == fmin_old):
                    print("fmin    = ", fmin)
                    print("CONVERGENCE")
                    break
                else:
                    fmin_old = fmin
                    print("fmin    = ", fmin)
                    # Using new weights to get new scalling factor
                    sf_ = leastsq(coeff_fit, sf_opt, args=(w_opt,std, sim_em_v_data,exp_em_mask))[0]
                    sf_opt = sf_
                    # geting new sim data with new nuisance parameter
                    sim_em_v = sim_em_v_data * sf_opt
        sf_opt_array.append(sf_opt)
        w_opt_array.append(w_opt)
        f_min_array.append(fmin)
    S_array = [get_entropy(w0,i) for i in w_opt_array]
    chisqrt_array = [chiSqrTerm(w_opt_array[i],std,sim_em_v_data*sf_opt_array[i],exp_em_mask) for i in range(0,len(thetas))]
    return w_opt_array, S_array, chisqrt_array

def get_entropy(w0, weights):
    """
    Calculate entropy - Kullback-Leibler divergence
    """
    s = - np.sum(weights * np.log(weights / w0))
    return s

def knee_loc(chisqrt_array,S_array):
    """
    "" Finding knee in the curve
    """
    yy=np.array(chisqrt_array)
    xx=-1*np.array(S_array)
    kneedle = KneeLocator(xx,yy, S=1.0, curve="convex", direction="decreasing")
    theta_index = np.where(xx==kneedle.knee)[0][0]
    return theta_index
