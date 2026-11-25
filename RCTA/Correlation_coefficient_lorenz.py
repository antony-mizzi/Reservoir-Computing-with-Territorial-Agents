import numpy as np 
import maps_and_systems as mapnsys
import genetic_and_other_optimisers as chaz 
import Terps_2D as terpo 
import random as rand
import matplotlib.pyplot as plt
import helper_funcs as helpo
import graphing as grapo
import time
import concap as conco
import Description_Lengther as deslo

  
        
def do_run(system = 'Lorenz', LenTS = 8000, territory_method = 'from_TS', N_vertex = 20, alpha = 10**(-12),
              lag = 1,Ff=4,Fh=30,Fp=10,Fa=0,horizon=25):
    
    if system == 'Lorenz':
        TS = mapnsys.runge_Lorenz(LenTS+400,Sr=0.02)
        TS2 = mapnsys.runge_Lorenz(LenTS,Sr=0.02)
        lag = 7; benchmark_err = 0.7
        suffix = '_Lorenz'
    elif system == 'Ikeda':
        TS = mapnsys.Ikeda(LenTS+400)
        TS2 = mapnsys.Ikeda(LenTS)
        suffix = '_Ikeda'
    elif system == 'Thomas':
        TS = mapnsys.runge_Thomas(LenTS+400)
        TS2 = mapnsys.runge_Thomas(LenTS)
        lag = 7
        suffix = '_Thomas'
    elif system == 'Halvorsen':
        TS = mapnsys.runge_Halvorsen(LenTS+400)
        TS2 = mapnsys.runge_Halvorsen(LenTS)
        lag = 7; benchmark_err = 0.7
        suffix = '_Halvorsen'
    elif system == 'Rossler':
        TS = mapnsys.runge_Rossler(LenTS+400)
        TS2 = mapnsys.runge_Rossler(LenTS)
        lag = 5; benchmark_err = 0.2
        suffix = '_Rossler'
    elif system == 'Lorenz96':
        TS = mapnsys.Lorenz96(LenTS+400)
        TS2 = mapnsys.Lorenz96(LenTS)
        lag = 8
        suffix = '_Lorenz96'
    elif system == 'Mackey_G':
        TS = mapnsys.Mackey_glass(LenTS+400,Sr=1)
        TS2 = mapnsys.Mackey_glass(LenTS,Sr=1)
        lag=6; benchmark_err = 0.15
        suffix = '_Mackey_G'
    if territory_method == 'MDL':
        #scoretrack = chaz.naturally_select(TS, TS, alpha=alpha,Ff=Ff,Fh=Fh,Fp=Fp,Fa=Fa,record_each_itt=True,
        #                                   setup={'system':system,'method':territory_method})
        #bs = chaz.review_scores_by_crit(scoretrack, LenTS, method = 'DL')
        #verticies = bs['homes']
        verticies = chaz.Terpstien_island(TS, TS2, alpha=alpha,Ff=Ff,Fh=Fh,Fp=Fp,Fa=Fa,method='DL',record_each_itt=True,
                                            setup={'system':system,'method':territory_method},lag=lag)
        
    elif territory_method == 'MDL_NSGA2':
        verticies = chaz.NSGA2(TS, alpha=alpha,Ff=Ff,Fh=Fh,Fp=Fp,Fa=Fa,method='DL',lag=lag,system=system)
        home_file = 'homes_nsga.csv'
        recon_file = 'recons_nsga.csv'
        gen_file = 'gen_stats_nsga.csv'
        
    elif territory_method == 'MDL_breathing':
        scoretrack = chaz.naturally_select(TS, TS, alpha=alpha,Ff=Ff,Fh=Fh,Fp=Fp,Fa=Fa,record_each_itt=True,
                                           setup={'system':system,'method':territory_method},lag=lag)
        bs = chaz.review_scores_by_crit(scoretrack, LenTS, method = 'DL')
        verticies = bs['homes']
        #verticies = chaz.Terpstien_island(TS, TS2, alpha=alpha,Ff=Ff,Fh=Fh,Fp=Fp,Fa=Fa,method='DL',record_each_itt=True,
        #                                    setup={'system':system,'method':territory_method})
    elif territory_method == 'AIC_breathing':
        scoretrack = chaz.naturally_select(TS, TS, alpha=alpha,Ff=Ff,Fh=Fh,Fp=Fp,Fa=Fa,record_each_itt=True,
                                           setup={'system':system,'method':territory_method},lag=lag)
        bs = chaz.review_scores_by_crit(scoretrack, LenTS, method = 'AIC')
        verticies = bs['homes']
        #verticies = chaz.Terpstien_island(TS, TS2, alpha=alpha,Ff=Ff,Fh=Fh,Fp=Fp,Fa=Fa,method='DL',record_each_itt=True,
        #                                    setup={'system':system,'method':territory_method})
            
    elif territory_method == 'MDL_brute':
        #scoretrack = chaz.naturally_select(TS, TS, alpha=alpha,Ff=Ff,Fh=Fh,Fp=Fp,Fa=Fa,record_each_itt=True,
        #                                   setup={'system':system,'method':territory_method})
        #bs = chaz.review_scores_by_crit(scoretrack, LenTS, method = 'DL')
        #verticies = bs['homes']
        verticies = chaz.Randomly_generation_selection(TS, TS2, alpha=alpha,Ff=Ff,Fh=Fh,Fp=Fp,Fa=Fa,method='DL',record_each_itt=True,
                                            setup={'system':system,'method':territory_method},lag=lag)

    elif territory_method == 'AIC':
        #scoretrack = chaz.naturally_select(TS, TS, alpha=alpha,Ff=Ff,Fh=Fh,Fp=Fp,Fa=Fa,method='AIC',record_each_itt=True,
        #                                   setup={'system':system,'method':territory_method})
        #bs = chaz.review_scores_by_crit(scoretrack, LenTS, method = 'AIC')
        #verticies = bs['homes']
        verticies = chaz.Terpstien_island(TS, TS2, alpha=alpha,Ff=Ff,Fh=Fh,Fp=Fp,Fa=Fa,method='AIC',record_each_itt=True,
                                            setup={'system':system,'method':territory_method},lag=lag)

    elif territory_method == 'from_TS':
        noisyTS = TS+np.random.normal(0,0.1,size=len(TS))
        noisyemb = terpo.embedd_TS(noisyTS,lag=lag)
        points = [ noisyemb[i,:] for i in range(noisyemb.shape[0]) ]
        verticies = np.array(rand.sample(points,N_vertex))
        #helpo.log_homes(verticies, 'homes_random'+suffix+'.csv')
        home_file = 'homes.csv'
        recon_file = 'recons.csv'
        gen_file = 'gen_stats.csv'
        
    elif territory_method == 'random_uniform':
        if system == 'Ikeda':
            points = [ np.random.uniform(-0.5,2,2) for i in range(N_vertex) ]
        else:
            points = [ np.random.uniform(-2.5,2.5,2) for i in range(N_vertex) ]
        verticies = np.array(points)
        home_file = 'homes.csv'
        recon_file = 'recons.csv'
        gen_file = 'gen_stats.csv'

    
   # -------------------- running once to find useless agents
    terps = terpo.Terp_land2d(verticies,Fp=Fp,Fh=Fh,Ff=Ff,Fa=Fa,fit_inp = False)
    TSemb,Y = terps.embedd_TS(TS,lag=lag,horizon=horizon,plot=False)
    states,terpX=terps.process(TSemb,save_time=-300,dt=0.02)
    
    # --------------------- deleting useless agents
    verts = terps.homes
    usefull = np.var(terpX,axis=0)[:-1]
    n = len(usefull)//4
    usefull = usefull[:2*n].reshape(n, 2).prod(axis=1) * usefull[2*n:].reshape(n, 2).prod(axis=1)

    for delind in np.where(usefull == 0)[0][::-1]:
        verts = 1*np.delete(verts, delind, axis=0)
    
    
    # ----------------- Running to determine coefficients
    terps = terpo.Terp_land2d(verts,Fp=Fp,Fh=Fh,Ff=Ff,Fa=Fa,fit_inp = False)
    TSemb,Y = terps.embedd_TS(TS,lag=lag,horizon=horizon,plot=False)
    states,terpX=terps.process(TSemb,save_time=-300,dt=0.02)
    terpXend = terpX[-300:,:]; terpX = terpX[:-300,:]; Yend = Y[-300-lag:]; Y = Y[:-300]
    coefs = terps.fit_coefs(terpX,Y,alpha=alpha)
    


    #static_preds = np.matmul(terpXend[-300:,:],np.transpose(coefs)) 
    #MSE = np.mean(np.square(np.matmul(terpX,np.transpose(coefs))-Y))
    #NRMSE = np.sqrt(MSE)/np.sqrt(np.mean(np.square(Y)))
    #R2 = 1-len(Y)*MSE/(np.sum(np.square(Y-np.mean(Y))))
    
    #terps.last_inp = np.array([Y[-20-lag],Y[-20]])
    

    terps.state = terps.homes+np.random.normal(0,0.3,size=terps.homes.shape)
    terps.velocities = np.random.normal(0,10,size=terps.homes.shape)
    #states,terpX2=terps.process(TSemb,save_time=-200)

    TSemb,Y = terps.embedd_TS(TS2,lag=lag,horizon=horizon,plot=False)
    states,X=terps.process(TSemb,save_time=-200,dt=0.02)
    predictions = np.matmul(X,np.transpose(coefs))
    correlation = np.corrcoef(predictions[200:],Y[200:])
    print(correlation[0,1])
    correlation = np.dot(predictions[200:],Y[200:])/np.sqrt(np.sum(np.square(Y[200:]))*np.sum(np.square(predictions[200:])))
    print('correlation: '+str(correlation))
    
    MSE = np.mean(np.square(predictions[200:]-Y[200:]))
    NRMSE = np.sqrt(MSE)/np.std(Y)#(np.max(Y[200:])-np.min(Y[200:]))
    print('NRMSE: '+str(NRMSE))
    R2 = 1-len(Y)*MSE/(np.sum(np.square(Y[200:]-np.mean(Y[200:]))))
    print('R2: '+str(R2))
     
    return correlation, NRMSE, R2




corrs = []; R2s = []; NRMSEs = []
for go in range(5):
    print(go)
    corr,nrmse,R2 = do_run(territory_method = 'random_uniform', N_vertex = 50, alpha = 10**(-8),
                  lag = 7,Ff=20,Fh=80,Fp=60,Fa=0,system='Mackey_G',horizon=1)
    corrs.append(corr); NRMSEs.append(nrmse); R2s.append(R2)
    

    print('')
    print('----------------------------- Correlation')
    print(np.mean(corrs))
    #print(' = 1-'+str(1-np.mean(corrs)))
    uncert = grapo.get_error_bound(corrs)
    print('+- '+str(uncert))
    print('best is '+str(max(corrs)))
    print('')
    
    print('')
    print('----------------------------- R2')
    print(np.mean(R2s))
    uncert = grapo.get_error_bound(R2s)
    print('+- '+str(uncert))
    print('best is '+str(max(R2s)))
    print('')
    
    print('')
    print('----------------------------- NRMSE')
    print(np.mean(NRMSEs))
    uncert = grapo.get_error_bound(NRMSEs)
    print('+- '+str(uncert))
    print('best is '+str(min(NRMSEs)))
    print('')



