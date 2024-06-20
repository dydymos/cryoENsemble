import numpy as np
import scipy.optimize as sopt
from scipy.optimize import leastsq
from scipy.optimize import minimize,Bounds
import kneed
from kneed import KneeLocator
from scipy.ndimage import gaussian_filter

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


def sigma_fit(sigma, w_opt, std, sim_map_m, exp_map,mask):
    """
    Reads simulated maps, applies gaussian filter with certain sigma
    to compare them to experimental values
    """
    sim_map_sf = [gaussian_filter(map,sigma[0]) for map in sim_map_m]
    sim_map_sf_masked = np.array([map[mask] for map in sim_map_sf])
    v = (np.sum(sim_map_sf_masked.T*w_opt,1) - exp_map)/std
    chiSqr = 0.5 * np.sum(v*v)
    return chiSqr

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


def bioen_sigma(sim_map_array,exp_map_norm,mask,std,theta,g0,g_init,sf_start,sigma0,resolution,n_iter,epsilon,pgtol,maxiter):
    """
    "" Single run of BioEN - with one theta value
    """
    print("THETA = "+str(theta))
    g = g_init
    # Initial sigma
    sigma_min = 1 * 0.225
    sigma_max = resolution * 0.225
    bounds = Bounds([sigma_min], [sigma_max])
    sim_map_array_g = gaussian_filter(sim_map_array,sigma0)
    N_models = np.shape(sim_map_array)[0]
    N_voxels = N_voxels=np.shape(mask)[1]
    sim_map_array_mask=np.zeros([N_models,N_voxels])
    for i in range(0,N_models):
        sim_map_array_mask[i] = sim_map_array_g[i][mask]
    sim_map_array_mask_sf = sim_map_array_mask * sf_start
    exp_map_mask = exp_map_norm[mask]
    sf_opt = sf_start
    chisqr0 = chiSqr(np.ones(N_models)/N_models,std,sim_map_array_g*sf_start,exp_map_norm,mask)
    print("ChiSqr0    = ", chisqr0)
    for i in range(0,n_iter):
        if (i == 0):
            print("ITERATION "+str(i+1))
            # Getting optimal weight
            res=sopt.fmin_l_bfgs_b(bioen_log_posterior_base,g,args = (g0, std, sim_map_array_mask_sf, exp_map_mask, theta),fprime = grad_bioen_log_posterior_base, epsilon = epsilon, pgtol = pgtol, maxiter = maxiter, disp = False)
            # new weights
            w_opt = getWeights(res[0])[0]
            # final energy
            fmin = res[1]
            fmin_old = fmin
            print("fmin    = ", fmin)
            # Using new weights to get new scalling factor
            sf_ = leastsq(coeff_fit, sf_opt, args=(w_opt,std, sim_map_array_mask,exp_map_mask))[0]
            sf_opt = sf_
            # Using new weights to get new sigma
            sigma_ = minimize(sigma_fit, [sigma_min], args=(w_opt,std,sim_map_array*sf_opt,exp_map_norm,mask),method="L-BFGS-B",bounds=bounds).x[0]
            sigma_opt = sigma_
            sim_map_array_g = gaussian_filter(sim_map_array,sigma_opt)
            for i in range(0,N_models):
                sim_map_array_mask[i] = sim_map_array_g[i][mask]
            sim_map_array_mask_sf = sim_map_array_mask * sf_opt
            chisqr_iter = chiSqr(w_opt,std,sim_map_array_g*sf_opt,exp_map_norm,mask)
            print("ChiSqr    = ", chisqr_iter)
        else:
            print("ITERATION "+str(i+1))
            # Getting optimal weight
            res=sopt.fmin_l_bfgs_b(bioen_log_posterior_base,g,args = (g0, std, sim_map_array_mask_sf, exp_map_mask, theta),fprime = grad_bioen_log_posterior_base, epsilon = epsilon, pgtol = pgtol, maxiter = maxiter, disp = False)
            # new weights
            # final energy
            fmin = res[1]
            if (fmin == fmin_old):
                print("fmin    = ", fmin)
                print("ChiSqr    = ", chisqr_iter)
                print("CONVERGENCE\n")
                break
            if (fmin > fmin_old):
                print("fmin    = ", fmin)
                print("ChiSqr    = ", chisqr_iter)
                print("ALREADY MINIMUM\n")
                break
            else:
                w_opt = getWeights(res[0])[0]
                fmin_old = fmin
                print("fmin    = ", fmin)
                # Using new weights to get new scalling factor
                sf_ = leastsq(coeff_fit, sf_opt, args=(w_opt,std, sim_map_array_mask,exp_map_mask))[0]
                sf_opt = sf_
                # Using new weights to get new sigma
                sigma_ = minimize(sigma_fit, [sigma_min], args=(w_opt,std,sim_map_array*sf_opt,exp_map_norm,mask),method="L-BFGS-B",bounds=bounds).x[0]
                sigma_opt = sigma_
                sim_map_array_g = gaussian_filter(sim_map_array,sigma_opt)
                for i in range(0,N_models):
                    sim_map_array_mask[i] = sim_map_array_g[i][mask]
                sim_map_array_mask_sf = sim_map_array_mask * sf_opt
                chisqr_iter = chiSqr(w_opt,std,sim_map_array_g*sf_opt,exp_map_norm,mask)
                print("ChiSqr    = ", chisqr_iter)
    return w_opt, sf_opt, sigma_opt



def bioen_sigma_one(sim_map_g_mask,exp_map_mask,std,theta,g0,g_init,epsilon,pgtol,maxiter):
    """
    "" Single run of BioEN - with one theta value
    """
    print("THETA = "+str(theta))
    g = g_init
    N = np.shape(sim_map_g_mask)[0]
    w0 = np.ones(N)/N
    chisqrt = chiSqrTerm(w0,std,sim_map_g_mask,exp_map_mask)
    print("Starting ChiSqrt = "+str(chisqrt))
    # Getting optimal weight
    res=sopt.fmin_l_bfgs_b(bioen_log_posterior_base,g,args = (g0, std, sim_map_g_mask, exp_map_mask, theta),fprime = grad_bioen_log_posterior_base, epsilon = epsilon, pgtol = pgtol, maxiter = maxiter, disp = False)
    # new weights
    w_opt = getWeights(res[0])[0]
    # final energy
    fmin = res[1]
    fmin_old = fmin
    print("fmin    = ", fmin)
    # geting new sim data with new nuisance parameter
    chisqrt = chiSqrTerm(w_opt,std,sim_map_g_mask,exp_map_mask)
    print("ChiSqrt = "+str(chisqrt))
    return w_opt


def get_entropy(w0, weights):
    """
    Calculate entropy - Kullback-Leibler divergence
    """
    s = -np.sum(weights * np.log(weights / w0))
    return s

def knee_loc(S_array,chisqrt_array):
    """
    "" Finding knee in the curve
    """
    yy=np.sort(np.array(chisqrt_array))[::-1]
    xx=-1*np.sort(np.array(S_array))[::-1]
    kneedle = KneeLocator(xx,yy, S=1.0, curve="convex", direction="decreasing")
    theta_index = np.where(-1*np.array(S_array)==kneedle.knee)[0][0]
    return theta_index
