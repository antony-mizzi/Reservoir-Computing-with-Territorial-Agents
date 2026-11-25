import numpy as np 
import matplotlib.pyplot as plt
import random as rand
import maps_and_systems as mapnsys
import Terps_2D as terp2d
import ESNs 
import math
import Description_Lengther as deslo


def component_embedding(X):
   # for j in range(4):
    for j in [0,100]:
        i = rand.randint(0,X.shape[1]-1)
        plt.plot(X[1:1001,i],X[6:1006,i],alpha=0.3)
    plt.show()

N = 50; alpha = 10**(-12)

TS = mapnsys.runge_Lorenz(5000,dt=0.02)
TS2 = mapnsys.runge_Lorenz(5000,dt=0.02)
mini = min(TS);maxi=max(TS)
mini = -2; maxi = 2

verticies = np.array([np.random.normal(0,1,size=2) for i in range(N)])
verticies = np.array([[mini+(maxi-mini)*rand.random(),mini+(maxi-mini)*rand.random()] for i in range(N)])
noisyTS = TS+np.random.normal(0,0.5,size=len(TS))
noisyemb = terp2d.embedd_TS(noisyTS,lag=5)
points = [ noisyemb[i,:] for i in range(noisyemb.shape[0]) ]
verticies = np.array(rand.sample(points,N))
print(verticies)
print(verticies.shape)
print(1/0)

terps = terp2d.Terp_land2d(verticies,Fp=10,Fh=60,Ff=7,Fa=1)
esn = ESNs.ESN(n=4*N)

TerpMSEs = []; EsnMSEs = []
for horizon in range(1,105,4):
    normaliser = np.mean(np.square(TS))
    if horizon == 1:
        plot = True
    else:
        plot = False
    TSemb,Y = terps.embedd_TS(TS,lag=5,horizon=horizon,plot=plot)
    TSemb2,Y2 = terps.embedd_TS(TS2,lag=5,horizon=horizon,plot=plot)
    mask = terps.mask_ts(TSemb)
    states,terpX=terps.process(TSemb)
    states2,terpX2=terps.process(TSemb2)
    
    if False:
        means = terpX.mean(axis=0)
        terpX = terpX - means
        terpX2 = terpX2 - means
        stds = np.reciprocal(np.sqrt(np.var(terpX,axis=0)+0.00001))
        terpX = np.multiply(stds,terpX)
        terpX = terpX[:,:-1]
        terpX2 = np.multiply(stds,terpX2)
        terpX2 = terpX2[:,:-1]
        ones_column = np.ones((terpX.shape[0], 1), dtype=terpX.dtype)
        terpX = np.hstack((terpX, ones_column))
        terpX2 = np.hstack((terpX2, ones_column))
    terpcoefs = terps.fit_coefs(terpX,Y,alpha=alpha,transient=200)
    DL = deslo.get_total_DL(terpX,Y,alpha)
    component_embedding(terpX)
    
    esnX = esn.generate_state_matrix(TS[5:],leak=1,transient=0)
    esnX2 = esn.generate_state_matrix(TS2[5:],leak=1,transient=0)
    if False:
        means = esnX.mean(axis=0)
        esnX = esnX - means
        esnX2 = esnX2 - means
        stds = np.reciprocal(np.sqrt(np.var(esnX,axis=0)+0.00001))
        esnX = np.multiply(stds,esnX)
        esnX2 = np.multiply(stds,esnX2)
        esnX = esnX[:,:-1]
        esnX2 = esnX2[:,:-1]
        ones_column = np.ones((esnX.shape[0], 1), dtype=terpX.dtype)
        esnX = np.hstack((esnX, ones_column))
        esnX2 = np.hstack((esnX2, ones_column))
    #X = X*math.sqrt(1/physV)
    #X = X[:,:-1]

    if horizon > 0:
        esnX = esnX[:-horizon]
        esnX2 = esnX2[:-horizon]

    esncoefs = terp2d.fit_coefs(esnX[200:,:], Y[200:], alpha=alpha)
    
    terp_pred = np.matmul(terpX2[200:,:],terpcoefs)
    esn_pred = np.matmul(esnX2[200:,:],esncoefs)
    plt.title('Horizon = '+str(horizon))
    plt.plot(Y2[400:1000],color='k')  
    plt.plot(terp_pred[200:800],color='r',label='terps',alpha=0.3)
    plt.plot(esn_pred[200:800],color='C1',label='ESN',alpha=0.3)
    plt.legend()
    plt.show()
    terpMSE = math.log10(np.mean(np.square(terp_pred-Y2[200:]))/normaliser)
    esnMSE = math.log10(np.mean(np.square(esn_pred-Y2[200:]))/normaliser)
    TerpMSEs.append(terpMSE); EsnMSEs.append(esnMSE)
    print('Log Terp MSE is '+str(round(terpMSE,3)))
    print('Log ESN MSE is '+str(round(esnMSE,3)))

plt.title('Log MSE')
plt.plot(0.02*np.array(list(range(0,104,4))),TerpMSEs,label='Terps')
plt.plot(0.02*np.array(list(range(0,104,4))),EsnMSEs,label='ESN')
plt.legend()
plt.xlabel('Forecasting Horizon (s)')
plt.show()

