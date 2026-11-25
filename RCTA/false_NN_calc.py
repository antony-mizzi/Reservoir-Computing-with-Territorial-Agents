import numpy as np 
import maps_and_systems as mapnsys
import random as rand
import time
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scienceplots
plt.style.use(['science', 'no-latex'])

def embedd_TS_2D(TS,lag=1,horizon=1):
    return np.array([[TS[t],TS[t+lag]] for t in range(len(TS)-2*lag-horizon)])

def embedd_TS_3D(TS,lag=1,horizon=1):
    return np.array([[TS[t],TS[t+lag],TS[t+2*lag]] for t in range(len(TS)-2*lag-horizon)])

def get_error_bound(lst,log=False):
    samp_means = [np.mean(np.random.choice(lst,size=len(lst))) for i in range(200)]
    return np.std(samp_means)

def false_nearest_neighbours_in_R2(TS,lag=7,epsilon=0.1,window=20):
    TS2d = embedd_TS_2D(TS,lag)
    TS3d = embedd_TS_3D(TS,lag)
    N = len(TS2d)
    all_NNs = []; true_NNs=[]
    for i in range(N):
        if i % 100 == 0:
            print(i)
        all_NNs.append([j for j in range(N) if np.linalg.norm(TS2d[i]-TS2d[j])<epsilon and abs(i-j) > window])
        true_NNs.append([j for j in range(N) if np.linalg.norm(TS3d[i]-TS3d[j])<np.sqrt(1.5)*epsilon and 
                         abs(i-j) > window])

    Total_NNs = np.sum([len(nns) for nns in all_NNs])
    True_NNs = np.sum([len(nns) for nns in true_NNs])
    print('Total NNs: '+str(Total_NNs))
    print('True NNs: '+str(True_NNs))
    Fnnpercent = (Total_NNs-True_NNs)/Total_NNs
    
    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax1.plot(TS2d[:,0],TS2d[:,1],color='k',alpha=0.2,linewidth=1)
    ax2.plot(TS3d[:,0],TS3d[:,1],TS3d[:,2],color='k',alpha=0.2,linewidth=1)
    for i in range(N):
        ax1.plot([TS2d[i,0]],[TS2d[i,1]],'o',markersize=2,color='k')
        ax2.plot([TS3d[i,0]],[TS3d[i,1]],[TS3d[i,2]],'o',markersize=2,color='k')
    for i in range(N):
        for j in all_NNs[i]:
            if j in true_NNs[i]:
                ax1.plot([TS2d[i,0],TS2d[j,0]],[TS2d[i,1],TS2d[j,1]],color='C1',alpha=0.1)
                ax2.plot([TS3d[i,0],TS3d[j,0]],[TS3d[i,1],TS3d[j,1]],[TS3d[i,2],TS3d[j,2]], 
                         color='green',alpha=0.1)
            else:
                ax1.plot([TS2d[i,0],TS2d[j,0]],[TS2d[i,1],TS2d[j,1]],color='C2',alpha=0.1)
                ax2.plot([TS3d[i,0],TS3d[j,0]],[TS3d[i,1],TS3d[j,1]],[TS3d[i,2],TS3d[j,2]], 
                         color='red',alpha=0.1)
    ax1.set_title('Lorenz',fontsize=20)
    ax2.set_title('FNN = '+str(round(Fnnpercent,2)),fontsize=20)
    ax2.xaxis.pane.set_visible(False)
    ax2.yaxis.pane.set_visible(False)
    ax2.zaxis.pane.set_visible(False)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('z')
    ax2.view_init(elev=15, azim=15)
    plt.tight_layout()
    plt.show()
    return Fnnpercent

def FNN_4plot(epsilon=0.1,window=20,title_means=[None,None,None,None],title_stds=[None,None,None,None]):
    TSnames = ['Lorenz','Rössler','Mackey Glass','Halvorsen']
    TSs = [mapnsys.runge_Lorenz(1000),mapnsys.runge_Rossler(1000),
           mapnsys.Mackey_glass(1000),mapnsys.runge_Halvorsen(1000)]
    lags = [7,5,6,7]; FNNvals = []
    fig,axs=plt.subplots(2,2,figsize=(10,10),dpi=120)
    for ind in range(4):
        print('Running '+str(TSnames[ind]))
        TS = TSs[ind]; lag = lags[ind]
        
        TS2d = embedd_TS_2D(TS,lag)
        TS3d = embedd_TS_3D(TS,lag)
        N = len(TS2d)
        all_NNs = []; true_NNs=[]
        for i in range(N):
            all_NNs.append([j for j in range(N) if np.linalg.norm(TS2d[i]-TS2d[j])<epsilon and abs(i-j) > window])
            true_NNs.append([j for j in range(N) if np.linalg.norm(TS3d[i]-TS3d[j])<np.sqrt(1.5)*epsilon and 
                             abs(i-j) > window])
    
        Total_NNs = np.sum([len(nns) for nns in all_NNs])
        True_NNs = np.sum([len(nns) for nns in true_NNs])
        Fnnpercent = (Total_NNs-True_NNs)/Total_NNs
        FNNvals.append(Fnnpercent)
        
        axs[ind//2,ind%2].plot(TS2d[:,0],TS2d[:,1],color='k',alpha=0.2,linewidth=1)
       
        for i in range(N):
            axs[ind//2,ind%2].plot([TS2d[i,0]],[TS2d[i,1]],'o',markersize=2,color='k')
        for i in range(N):
            for j in all_NNs[i]:
                if j in true_NNs[i]:
                    axs[ind//2,ind%2].plot([TS2d[i,0],TS2d[j,0]],[TS2d[i,1],TS2d[j,1]],color='C1',alpha=0.1)
                else:
                    axs[ind//2,ind%2].plot([TS2d[i,0],TS2d[j,0]],[TS2d[i,1],TS2d[j,1]],color='C2',alpha=0.1)

        axs[ind//2,ind%2].set_title(TSnames[ind]+r' (FNN % = $'+str(round(100*title_means[ind],3))+
                                    '\pm'+str(round(300*title_stds[ind],3))+'$)',fontsize=20)
        axs[ind//2,ind%2].set_xticks([])
        axs[ind//2,ind%2].set_yticks([])
       
    plt.tight_layout()
    plt.show()
    return FNNvals
    

def get_FNN_bounds(runs=30):
    FNNs = [[],[],[],[]]
    #title_means = [0.2856,0.1816,0.4966,0.6494] # epsilon = 0.1
    #title_stds = [0.0163,0.0055,0.0054,0.0099] # epsilon = 0.1
    title_means = [0.2359,0.1468,0.3839,0.6091] # epsilon = 0.2
    title_stds = [0.0132,0.0018,0.0018,0.0058] # epsilon = 0.2
    for run in range(runs):
        sample_fnns = FNN_4plot(epsilon=0.2,title_means=title_means,title_stds=title_stds)
        FNNs[0].append(sample_fnns[0]); FNNs[1].append(sample_fnns[1])
        FNNs[2].append(sample_fnns[2]); FNNs[3].append(sample_fnns[3])
        print('')
        print('---------------------------------------------------------------------')
        print('after '+str(run+1)+' samples: ')
        print('Lorenz: '+str(round(np.mean(FNNs[0]),4))+' +- '+str(round(get_error_bound(FNNs[0]),4)))
        print('Rossler: '+str(round(np.mean(FNNs[1]),4))+' +- '+str(round(get_error_bound(FNNs[1]),4)))
        print('Mackey Glass: '+str(round(np.mean(FNNs[2]),4))+' +- '+str(round(get_error_bound(FNNs[2]),4)))
        print('Halvorsen: '+str(round(np.mean(FNNs[3]),4))+' +- '+str(round(get_error_bound(FNNs[3]),4)))
        print('---------------------------------------------------------------------')
        print('')
        time.sleep(15)
        
 
    
title_means = [0.2359,0.1468,0.3839,0.6091] # epsilon = 0.2
title_stds = [0.0132,0.0018,0.0018,0.0058]
FNN_4plot(epsilon=0.2,title_means=title_means,title_stds=title_stds)
get_FNN_bounds(runs=10)
#FNN_4plot(title_means=[0.2359,0.1468,0.3839,0.6091],title_stds=[0.0]) 
#TS = mapnsys.Mackey_glass(600)
#TS = mapnsys.runge_Lorenz(1000)
#TS = mapnsys.runge_Rossler(1000)
#TS = mapnsys.runge_Halvorsen(1000)
#FNN = false_nearest_neighbours_in_R2(TS, lag=7)
#print(FNN)