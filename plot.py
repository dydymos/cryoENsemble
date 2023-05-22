import numpy as np
import matplotlib.pyplot as plt



def plot_lcurve(s_dict,chisqrt_d,theta_new,N_voxels,name):
    cmap = plt.cm.get_cmap('rainbow')
    plt.figure(figsize=[6,6])
    N = len(s_dict)
    s_array = np.array(list(s_dict.values()))
    chisqrt_array = np.array(list(chisqrt_d.values()))
    for i in range(0,N):
        key = np.sort(list(s_dict.keys()))[::-1][i]
        plt.scatter(-1*s_dict[key],chisqrt_d[key]/N_voxels,color=cmap((i+1)/N),label=str(key),s=80)
    plt.scatter(-1*s_dict[theta_new],chisqrt_d[theta_new]/N_voxels,facecolor="none",edgecolor="black",linewidth=3,s=100)
    plt.legend(loc=(1.0,0))
    plt.xlabel("Entropy")
    plt.ylabel("Reduced Chi2")
    x_plot_delta = np.max(-1*s_array)-np.min(-1*s_array)
    y_plot_delta = np.max(chisqrt_array)/N_voxels-np.min(chisqrt_array)/N_voxels
    plt.ylim(np.min(chisqrt_array)/N_voxels-0.05*y_plot_delta,np.max(chisqrt_array)/N_voxels+0.05*y_plot_delta)
    plt.xlim(np.min(-1*s_array)-0.05*x_plot_delta,np.max(-1*s_array)+0.05*x_plot_delta)
    plt.savefig(name)
    plt.clf()

def plot_Neff(s_dict,chisqrt_d,theta_new,N_voxels,name):
    cmap = plt.cm.get_cmap('rainbow')
    plt.figure(figsize=[6,6])
    N = len(s_dict)
    s_array = np.array(list(s_dict.values()))
    chisqrt_array = np.array(list(chisqrt_d.values()))
    for i in range(0,N):
        key = np.sort(list(s_dict.keys()))[::-1][i]
        plt.scatter(np.exp(s_dict[key]),chisqrt_d[key]/N_voxels,color=cmap((i+1)/N),label=str(key),s=80)
    plt.scatter(np.exp(s_dict[theta_new]),chisqrt_d[theta_new]/N_voxels,facecolor="none",edgecolor="black",linewidth=3,s=100)
    plt.legend(loc=(1.0,0))
    plt.xlabel("Neff")
    plt.ylabel("Reduced Chi2")
    x_plot_delta = np.max(np.exp(s_array))-np.min(np.exp(s_array))
    y_plot_delta = np.max(chisqrt_array)/N_voxels-np.min(chisqrt_array)/N_voxels
    plt.ylim(np.min(chisqrt_array)/N_voxels-0.05*y_plot_delta,np.max(chisqrt_array)/N_voxels+0.05*y_plot_delta)
    plt.xlim(np.min(np.exp(s_array))-0.05*x_plot_delta,np.max(np.exp(s_array))+0.05*x_plot_delta)
    plt.savefig(name)
    plt.clf()


def plot_weights(w_opt_d,sel,theta_new,N,name):
    cmap = plt.cm.get_cmap('rainbow')
    x = np.zeros(N)
    plt.figure(figsize=[6,6])
    N = len(w_opt_d)
    for i in range(0,N):
        key = np.sort(list(w_opt_d.keys()))[::-1][i]
        x[sel] = w_opt_d[key]
        plt.plot(x,'o-',color=cmap((i+1)/N),label=str(key))
    x[sel] = w_opt_d[theta_new]
    plt.plot(x,'o-',color="black")
    plt.legend(loc=(1.0,0))
    plt.xlabel("Models")
    plt.ylabel("Weights")
    plt.savefig(name)
    plt.clf()


def plot_fsc(fsc_0,fsc_1,name):
    plt.plot(fsc_0[:,0],fsc_0[:,1],label="FSC prior")
    plt.plot(fsc_1[:,0],fsc_1[:,1],label="FSC posterior")
    plt.xlabel('1/resolution [A-1]')
    plt.ylabel('Fourier Shell Correlation')
    plt.legend()
    plt.savefig(name)
    plt.clf()

def plot_smoc(smoc_prior,smoc_poster,name):
    plt.plot(smoc_prior,label="Prior")
    plt.plot(smoc_poster,label="Posterior")
    plt.ylim(0,1.1)
    plt.ylabel("SMOC score")
    plt.xlabel("Sequence")
    plt.legend()
    plt.savefig(name)
    plt.clf()


