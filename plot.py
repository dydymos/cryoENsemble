import numpy as np
import matplotlib.pyplot as plt



def plot_lcurve(s_dict,chisqrt_d,theta_index,N_voxels):
    N = len(s_dict)
    cmap = plt.cm.get_cmap('rainbow')
    plt.figure(figsize=[6,6])
    for i in range(0,N):
        key = list(s_dict.keys())[i]
        plt.scatter(-1*s_dict[key],chisqrt_d[key]/N_voxels,color=cmap((i+1)/N),label=str(key),s=80)
    plt.scatter(-1*s_dict[theta_index],chisqrt_array[theta_index]/N_voxels,facecolor="none",edgecolor="black",linewidth=3,s=100)
    plt.legend(loc=(1.0,0))
    plt.xlabel("Entropy")
    plt.ylabel("Reduced Chi2")
    x_plot_delta = np.max(-np.array(S_array))-np.min(-np.array(S_array))
    y_plot_delta = np.max(chisqrt_array)/N_voxels-np.min(chisqrt_array)/N_voxels
    plt.ylim(np.min(chisqrt_array)/N_voxels-0.05*y_plot_delta,np.max(chisqrt_array)/N_voxels+0.05*y_plot_delta)
    plt.xlim(np.min(-np.array(S_array))-0.05*x_plot_delta,np.max(-np.array(S_array))+0.05*x_plot_delta)
    plt.savefig("l-curve.svg")
