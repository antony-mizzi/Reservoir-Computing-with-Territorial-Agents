import numpy as np 
import maps_and_systems as mapnsys
import genetic_and_other_optimisers as chaz 
import Terps_2D as terpo 
import random as rand
import matplotlib.pyplot as plt
import helper_funcs as helpo
import time
import csv
plt.style.use(['science', 'no-latex'])

def get_error_bound(lst,log=False):
    samp_means = [np.mean(np.random.choice(lst,size=len(lst))) for i in range(len(lst))]
    #stds = np.array([np.std(s) for s in samp_means])
    return np.std(samp_means)

def get_errors_by_horizon(system = 'Lorenz', LenTS = 3000, territory_method = 'from_TS', N_vertex = 20, alpha = 10**(-12),
              lag = 1,Ff=4,Fh=30,Fp=10,Fa=0):
    
    if system == 'Lorenz':
        TS = mapnsys.runge_Lorenz(LenTS+400,Sr=0.02)
        TS2 = mapnsys.runge_Lorenz(5000,Sr=0.02)
        lag = 7
        suffix = '_Lorenz'
    elif system == 'Ikeda':
        TS = mapnsys.Ikeda(LenTS+400)
        TS2 = mapnsys.Ikeda(LenTS)
        suffix = '_Ikeda'
    elif system == 'Thomas':
        TS = mapnsys.runge_Thomas(LenTS+400)
        TS2 = mapnsys.runge_Thomas(5000)
        lag = 7
        suffix = '_Thomas'
    elif system == 'Halvorsen':
        TS = mapnsys.runge_Halvorsen(LenTS+400)
        TS2 = mapnsys.runge_Halvorsen(5000)
        lag = 7
        suffix = '_Halvorsen'
    elif system == 'Rossler':
        TS = mapnsys.runge_Rossler(LenTS+400)
        TS2 = mapnsys.runge_Rossler(5000)
        lag = 5
        suffix = '_Rossler'
    elif system == 'Lorenz96':
        TS = mapnsys.Lorenz96(LenTS+400)
        TS2 = mapnsys.Lorenz96(5000)
        lag = 8
        suffix = '_Lorenz96'
    elif system == 'Mackey_G':
        TS = mapnsys.Mackey_glass(LenTS+400)
        TS2 = mapnsys.Mackey_glass(5000)
        lag=6
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
        helpo.log_homes(verticies, 'homes_random'+suffix+'.csv')
    elif territory_method == 'random_uniform':
        if system == 'Ikeda':
            points = [ np.random.uniform(-0.5,2,2) for i in range(N_vertex) ]
        else:
            points = [ np.random.uniform(-2.5,2.5,2) for i in range(N_vertex) ]
        verticies = np.array(points)

    
    terps = terpo.Terp_land2d(verticies,Fp=Fp,Fh=Fh,Ff=Ff,Fa=Fa,fit_inp = False)
    TSemb,Y = terps.embedd_TS(TS,lag=lag,horizon=1,plot=False)
    states,terpX=terps.process(TSemb,save_time=-300)
    
    
    terpXend = terpX[-300:,:]; terpX = terpX[:-300,:]; Yend = Y[-300-lag:]; Y = Y[:-300]
    coefs = terps.fit_coefs(terpX,Y,alpha=alpha)
    

    static_preds = np.matmul(terpXend[-300:,:],np.transpose(coefs)) 
    #MSE = np.mean(np.square(np.matmul(terpX,np.transpose(coefs))-Y))
    #print('log Training MSE is '+str(np.log(MSE)))
    terps.last_inp = np.array([Y[-20-lag],Y[-20]])
    recon,recostart = terps.auto_recon(coefs,Yend[:-200].tolist(),Nsteps=200,lag=lag)
    
    #cap = min(2000,len(recon) - lag - 1)
    #fig,axs=plt.subplots(1,1,figsize=(8,8),dpi=120)
    #axs.plot(recon[:cap],recon[lag:cap+lag],linewidth=1,alpha=0.5,color='k',label='recon')
    #    #axs.plot(recostart[:-lag],recostart[lag:],linewidth=1,alpha=0.5,color='C0',label='recostart')
    #    #axs.plot([recostart[-lag]],[recostart[-1]],'o',color='r',label='recostart[-1]')
    #axs.plot([recon[0]],[recon[lag]],'o',color='k')
    #    #axs.plot(Y[-20:-10-lag],Y[-20+lag:-10],linewidth=1,alpha=0.3,color='C1',label='lead-in')
    #    #axs.plot(Y[-10-lag:-lag],Y[-10:],linewidth=1,alpha=0.6,color='C0')
    #axs.plot(Yend[lag:-lag],Yend[2*lag:],linewidth=1,alpha=0.2,color='C2',label='True trajectory')
    #    #axs.plot(static_preds[:-lag],static_preds[lag:],alpha=0.2,color='C1',linestyle='dashed')
    #axs.legend(fontsize=15)
    #plt.tight_layout()
    #plt.show()
    
    
    
    #plt.plot(Yend[lag:],color='C1',label='Yend[lag:]')
    #plt.plot(recostart,color='C0',label='recostart',linestyle='dashed')
        #plt.plot(static_preds,color='C2',linestyle='dashed',label='static_preds')
    #plt.legend()
    #plt.vlines(100,ymin=-2,ymax=2,color='k',linestyle='dotted')
    #plt.show()
    
    #errors = np.abs(np.array(Yend[lag:])-np.array(recostart[:-lag]))
    #plt.plot(np.log(errors[99:]))
    #plt.hlines(np.log(0.5),xmin=0,xmax=len(errors)-100,color='k',linestyle='dashed')
    #plt.show()
    
    errors = terpo.get_average_absolute_forecast_errors(terps, coefs, TS2, lag=lag)
    #plt.plot(np.log(errors))
    #plt.hlines(np.log(0.5),xmin=0,xmax=len(errors),color='k',linestyle='dashed')
    #plt.hlines(np.log(0.2),xmin=0,xmax=len(errors),color='k',linestyle='dashed')
    #plt.hlines(np.log(0.1),xmin=0,xmax=len(errors),color='k',linestyle='dashed')
    #plt.show()
    
    
    helpo.log_recon(errors.tolist(), 'recon_errors.csv', method=territory_method, TS=system,size=len(terps.homes))
    return errors


def make_plot(fname='recon_errors.csv'):
    
    TSs = ['Lorenz','Rossler','Mackey_G','Halvorsen']
    Titles = ['Lorenz','Rössler','Mackey Glass','Halvorsen']
    steps_per_Ltime = [45.45,39.22,50,42.74]
    methods = ['from_TS','random_uniform']
    fig,axs = plt.subplots(2,2,figsize=(14,8),dpi=120)
    colors = ['saddlebrown','magenta']
    epsilons = [0.7,0.2,0.15,0.7]
    labels = ['Territories Sampled from Time Series','Territories Sampled Uniformly']
    steps_per_Ltime = {'Lorenz':45.45,'Rossler':39.22,'Mackey_G':50,'Halvorsen':42.74}
    for tsind in range(4):
        for methind in range(2):
            Errs = []; Label = True
            with open('logs/'+fname, 'r', newline='') as file:
                reader = csv.reader(file)
                for row in reader:
                    if row[0] == methods[methind] and row[1] == TSs[tsind]:
                        Errs.append(np.log(np.array([float(r) for r in row[3:]])))
            x = np.array(list(range(len(Errs[0]))))/steps_per_Ltime[TSs[tsind]]
            #for err in Errs:
            #    if Label:
            #        axs[tsind//2,tsind%2].plot(x,err,color=colors[methind],alpha=0.5,label=labels[methind])
            #        Label = False
            #    else:
            #        axs[tsind//2,tsind%2].plot(x,err,color=colors[methind],alpha=0.2)
            Errs = [[e[i] for e in Errs] for i in range(len(x))]
            unc = np.array([3*get_error_bound(e) for e in Errs])
            means = np.array([np.mean(e) for e in Errs])
            axs[tsind//2,tsind%2].plot(x,means,color=colors[methind],alpha=1,
                                       label=labels[methind])
            axs[tsind//2,tsind%2].fill_between(x,means-unc,means+unc,color=colors[methind],alpha=0.3)
        axs[tsind//2,tsind%2].hlines(np.log(epsilons[tsind]),xmin=0,xmax=np.max(x),color='k',linestyle='dashed')
        axs[tsind//2,tsind%2].tick_params(axis='both', labelsize=16)
        axs[tsind//2,tsind%2].set_title(Titles[tsind],fontsize=22)
        xpos = 5*np.max(x)/7; ypos = np.log(epsilons[tsind])-0.5
        axs[tsind//2,tsind%2].text(xpos,ypos,'$\\varepsilon_{BM}='+str(epsilons[tsind])+'$',fontsize=24)
        axs[tsind//2,tsind%2].set_ylim(None,1)
        
    print('Number of samples is '+str(len(Errs[0])))
    axs[0,0].legend(fontsize=19)
    #axs[0,0].set_xticks([])
    #axs[0,1].set_xticks([])
    axs[1,0].set_xlabel('Lyapunov Times',fontsize=22)
    axs[1,1].set_xlabel('Lyapunov Times',fontsize=22)
    axs[0,0].set_ylabel('Log absolute error',fontsize=22)
    axs[1,0].set_ylabel('Log absolute error',fontsize=22)
    plt.tight_layout()
    plt.show()
                    
        
    


#make_plot(fname='recon_errors60.csv')
#print(1/0)
Ff=20; Fh=80; Fp=60
Es = []
for go in range(50):
    errs = get_errors_by_horizon(lag=5,alpha=10**(-8), system = 'Lorenz', territory_method = 'random_uniform',Ff=Ff, Fh=Fh, Fp=Fp, N_vertex=60)
    errs = get_errors_by_horizon(lag=5,alpha=10**(-8), system = 'Lorenz', territory_method = 'from_TS',Ff=Ff, Fh=Fh, Fp=Fp, N_vertex=60)
    
    errs = get_errors_by_horizon(lag=5,alpha=10**(-8), system = 'Rossler', territory_method = 'random_uniform',Ff=Ff, Fh=Fh, Fp=Fp, N_vertex=60)
    errs = get_errors_by_horizon(lag=5,alpha=10**(-8), system = 'Rossler', territory_method = 'from_TS',Ff=Ff, Fh=Fh, Fp=Fp, N_vertex=60)
    
    errs = get_errors_by_horizon(lag=5,alpha=10**(-8), system = 'Halvorsen', territory_method = 'random_uniform',Ff=Ff, Fh=Fh, Fp=Fp, N_vertex=60)
    errs = get_errors_by_horizon(lag=5,alpha=10**(-8), system = 'Halvorsen', territory_method = 'from_TS',Ff=Ff, Fh=Fh, Fp=Fp, N_vertex=60)
    
    errs = get_errors_by_horizon(lag=5,alpha=10**(-8), system = 'Mackey_G', territory_method = 'random_uniform',Ff=Ff, Fh=Fh, Fp=Fp, N_vertex=60)
    errs = get_errors_by_horizon(lag=5,alpha=10**(-8), system = 'Mackey_G', territory_method = 'from_TS',Ff=Ff, Fh=Fh, Fp=Fp, N_vertex=60)
    
    make_plot()
    time.sleep(5)




























