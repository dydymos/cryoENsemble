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


def plot_weights(w_opt_d,sel,theta_new,N,pdb_list,max_weights,name):
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
    plt.scatter(pdb_list,x[pdb_list],s=600, facecolors='none', edgecolors='grey')
    plt.legend(loc=(1.0,0))
    plt.xlabel("Models")
    plt.ylabel("Weights")
    plt.ylim(-0.005,max_weights+0.005)
    plt.savefig(name)
    plt.clf()


def plot_weights_missing(w_opt_d,sel,theta_new,N,name):
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
    plt.plot(fsc_0[:,0],fsc_0[:,1],label="FSC prior",lw=2)
    plt.plot(fsc_1[:,0],fsc_1[:,1],label="FSC posterior",lw=2)
    plt.xlabel('1/resolution [A-1]')
    plt.ylabel('Fourier Shell Correlation')
    plt.legend()
    plt.savefig(name)
    plt.clf()

def plot_fsc_sel(fsc_0,fsc_1,fsc_2,name):
    plt.plot(fsc_0[:,0],fsc_0[:,1],label="FSC prior",lw=2)
    plt.plot(fsc_1[:,0],fsc_1[:,1],label="FSC initial reweighting",lw=2)
    plt.plot(fsc_1[:,0],fsc_2[:,1],label="FSC final reweighting",lw=2)
    plt.xlabel('1/resolution [A-1]')
    plt.ylabel('Fourier Shell Correlation')
    plt.legend()
    plt.savefig(name)
    plt.clf()

def plot_smoc(smoc_prior,smoc_poster,name):
    plt.plot(smoc_prior,label="Prior",lw=2)
    plt.plot(smoc_poster,label="Posterior",lw=2)
    plt.ylim(0,1.1)
    plt.ylabel("SMOC score")
    plt.xlabel("Sequence")
    plt.legend()
    plt.savefig(name)
    plt.clf()

def plot_smoc_sel(smoc_prior,smoc_poster,smoc_poster_sel,name):
    plt.plot(smoc_prior,label="Prior",lw=2)
    plt.plot(smoc_poster,label="Initial reweighting",lw=2)
    plt.plot(smoc_poster_sel,label="Final reweighting",lw=2)
    plt.ylim(0,1.1)
    plt.ylabel("SMOC score")
    plt.xlabel("Sequence")
    plt.legend()
    plt.savefig(name)
    plt.clf()


def plot_rmsf(rmsf_target, rmsf_prior,rmsf_post,name):
    plt.plot(rmsf_target,label="target ensemble",ls='--',color="black",lw=2)
    plt.plot(rmsf_prior,label="prior ensemble",color="C0",lw=3,alpha = 0.8)
    plt.plot(rmsf_post,label="post ensemble",color="C1",lw=3,alpha = 0.8)
    plt.xlabel('Residue')
    plt.ylabel('RMSF (Å)')
    plt.title('RMSF of Backbone Atoms')
    plt.legend()
    plt.savefig(name)
    plt.clf()

def plot_rmsf_sel(rmsf_target, rmsf_prior,rmsf_post,rmsf_post_sel, name):
    plt.plot(rmsf_target,label="Target ensemble",ls='--',color="black",lw=2)
    plt.plot(rmsf_prior,label="Prior",color="C0",lw=3,alpha=0.8)
    plt.plot(rmsf_post,label="Initial reweighting",color="C1",lw=3,alpha = 0.8)
    plt.plot(rmsf_post_sel,label="Final reweighting",color="C2",lw=3,alpha = 0.8)
    plt.xlabel('Residue')
    plt.ylabel('RMSF (Å)')
    plt.title('RMSF of Backbone Atoms')
    plt.legend()
    plt.savefig(name)
    plt.clf()

def plot_pca(dot_products_prior,dot_products_post,name):
    plt.figure(figsize=(12, 6))
    plt.subplot(1,2,1)
    # Plot open state dot product matrix
    plt.imshow(np.abs(dot_products_prior), cmap='coolwarm_r', aspect='auto',vmax=1.0)
    # Set x and y ticks to correspond to 1, 2, 3
    plt.title("Target vs Prior")
    plt.xticks(ticks=[0, 1, 2], labels=['1', '2', '3'])
    plt.yticks(ticks=[0, 1, 2], labels=['1', '2', '3'])
    # Add colorbar
    plt.colorbar()
    # Display the plot
    plt.xlabel('Prior Main Eigenvectors')
    plt.ylabel('Target Main Eigenvectors')
    plt.subplot(1,2,2)
    # Plot close state dot product matrix
    plt.imshow(np.abs(dot_products_post), cmap='coolwarm_r', aspect='auto',vmax=1.0)
    # Set x and y ticks to correspond to 1, 2, 3
    plt.title("Target vs Posterior")
    plt.xticks(ticks=[0, 1, 2], labels=['1', '2', '3'])
    plt.yticks(ticks=[0, 1, 2], labels=['1', '2', '3'])
    # Add colorbar
    plt.colorbar()
    # Display the plot
    plt.xlabel('Posterior Main Eigenvectors')
    plt.ylabel('Target Main Eigenvectors')
    plt.savefig(name)
    plt.clf()

def plot_pca_sel(dot_products_prior,dot_products_post,dot_products_post_sel,name):
    plt.figure(figsize=(16, 6))
    plt.subplot(1,3,1)
    # Plot open state dot product matrix
    plt.imshow(np.abs(dot_products_prior), cmap='coolwarm_r', aspect='auto',vmax=1.0)
    # Set x and y ticks to correspond to 1, 2, 3
    plt.title("Target vs Prior")
    plt.xticks(ticks=[0, 1, 2], labels=['1', '2', '3'])
    plt.yticks(ticks=[0, 1, 2], labels=['1', '2', '3'])
    # Add colorbar
    plt.colorbar()
    # Display the plot
    plt.xlabel('Prior Main Eigenvectors')
    plt.ylabel('Target Main Eigenvectors')
    plt.subplot(1,3,2)
    # Plot close state dot product matrix
    plt.imshow(np.abs(dot_products_post), cmap='coolwarm_r', aspect='auto',vmax=1.0)
    # Set x and y ticks to correspond to 1, 2, 3
    plt.title("Target vs Initial reweighting")
    plt.xticks(ticks=[0, 1, 2], labels=['1', '2', '3'])
    plt.yticks(ticks=[0, 1, 2], labels=['1', '2', '3'])
    # Add colorbar
    plt.colorbar()
    # Display the plot
    plt.xlabel('Posterior Main Eigenvectors')
    plt.ylabel('Target Main Eigenvectors')
    plt.subplot(1,3,3)
    # Plot close state dot product matrix
    plt.imshow(np.abs(dot_products_post_sel), cmap='coolwarm_r', aspect='auto',vmax=1.0)
    # Set x and y ticks to correspond to 1, 2, 3
    plt.title("Target vs Final reweighting")
    plt.xticks(ticks=[0, 1, 2], labels=['1', '2', '3'])
    plt.yticks(ticks=[0, 1, 2], labels=['1', '2', '3'])
    # Add colorbar
    plt.colorbar()
    # Display the plot
    plt.xlabel('Posterior Main Eigenvectors')
    plt.ylabel('Target Main Eigenvectors')
    plt.savefig(name)
    plt.clf()
