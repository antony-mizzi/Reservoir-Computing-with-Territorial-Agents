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


def test_hypers(points, TS, TS2 ,system = 'Lorenz', N_vertex = 20, alpha = 10**(-6),
              lag = 5,Ff=4,Fh=30,Fp=10,Fa=0,make_plots=False,predforce='Step'):
    
    #points = [ np.random.uniform(-2.5,2.5,2) for i in range(N_vertex) ]
    verticies = np.array(points)
    terps = terpo.Terp_land2d(verticies,Fp=Fp,Fh=Fh,Ff=Ff,Fa=Fa,fit_inp = False)
    TSemb,Y = terps.embedd_TS(TS,lag=lag,horizon=1,plot=False)
    states,terpX=terps.process(TSemb,save_time=-300)
    terpXend = terpX[-300:,:]; terpX = terpX[:-300,:]; Yend = Y[-300-lag:]; Y = Y[:-300]
    coefs = terps.fit_coefs(terpX,Y,alpha=alpha,transient=200)
    static_preds = np.matmul(terpXend[-300:,:],np.transpose(coefs)) 
    MSE = np.mean(np.square(np.matmul(terpX,np.transpose(coefs))-Y))
    #print('log Trining MSE is '+str(np.log(MSE)))
    MSE = np.mean(np.square(static_preds-Yend[lag:]))
    #print('log Unseen MSE is '+str(np.log(MSE)))
    #terps.last_inp = np.array([Y[-20-lag],Y[-20]])
    recon,recostart = terps.auto_recon(coefs,Yend[:-200].tolist(),Nsteps=200,lag=lag)
    
    if make_plots:
        plt.plot(recostart,color='r')
        plt.plot(Yend[lag:],color='C0')
        plt.plot(static_preds,color='C1',linestyle='dashed')
        plt.vlines(100,ymin=-2,ymax=2,color='k',linestyle='dotted')
        plt.show()
        cap = min(2000,len(recon) - lag - 1)
        fig,axs=plt.subplots(1,1,figsize=(8,8))
        axs.plot(recon[:cap],recon[lag:cap+lag],linewidth=1,alpha=0.5,color='k')
        #axs.plot(recostart[:-lag],recostart[lag:],linewidth=1,alpha=0.5,color='r')
        #axs.plot([recostart[-lag]],[recostart[-1]],'o',color='r')
        #axs.plot([recon[0]],[recon[lag]],'o',color='k')
        #axs.plot(Y[-20:-10-lag],Y[-20+lag:-10],linewidth=1,alpha=0.3,color='C0')
        #axs.plot(Y[-10-lag:-lag],Y[-10:],linewidth=1,alpha=0.6,color='C0')
        axs.plot(Yend[lag:-lag],Yend[2*lag:],linewidth=1,alpha=0.2,color='C0')
        axs.plot(static_preds[:-lag],static_preds[lag:],alpha=0.2,color='C1',linestyle='dashed')
        plt.tight_layout()
        plt.show()
    
    terps.state = terps.homes+np.random.normal(0,0.3,size=terps.homes.shape)
    terps.velocities = np.random.normal(0,10,size=terps.homes.shape)
    states,terpX2=terps.process(TSemb,save_time=-200)
    try:
        conc = conco.get_concap(terpX[1000:,:], terpX2[1000:-300,:])
        #print('Consistency Cap is '+str(conc)+' of '+str(4*len(terps.homes)+1))
        FH = [i for i in range(len(Yend[100+lag:])-1) if abs(recostart[100+i]-Yend[100+lag+i])>0.5]
        if len(FH) > 0:
            FH = min(FH)
        else:
            FH = 200
        print('Forcasting Horizon from 1 avg: '+str(FH))
        FH = terpo.get_avg_forecasting_horizon(terps, coefs, TS2)
        print('Forcasting Horizon with avging : '+str(FH))
        
        TSemb,Y = terps.embedd_TS(TS2,lag=lag,horizon=1,plot=False)
        states,X=terps.process(TSemb,save_time=-200)
        NMSE = np.mean(np.square(np.matmul(X,np.transpose(coefs))-Y))/np.mean(np.square(Y))
        
        details = {'system':system,'Ff':Ff,'Fh':Fh,'Fp':Fp,'Fa':Fa,'N_terps':len(terps.homes),
                   'Concap':conc,'FH':FH,'NMSE_unseen':NMSE,'Predforce':predforce}
        helpo.log_data(details, 'param_scan.csv')
    except:
        print(1/0)
        print('FAILED GETTING STATS')
    return recon


def test_hypers_just_recon(points, TS, system = 'Lorenz', N_vertex = 20, alpha = 10**(-6),
              lag = 5,Ff=4,Fh=30,Fp=10,Fa=0,make_plots=False,predforce='Step'):
    
    #points = [ np.random.uniform(-2.5,2.5,2) for i in range(N_vertex) ]
    verticies = np.array(points)
    terps = terpo.Terp_land2d(verticies,Fp=Fp,Fh=Fh,Ff=Ff,Fa=Fa,fit_inp = False)
    TSemb,Y = terps.embedd_TS(TS,lag=lag,horizon=1,plot=False)
    states,terpX=terps.process(TSemb,save_time=-300)
    terpXend = terpX[-300:,:]; terpX = terpX[:-300,:]; Yend = Y[-300-lag:]; Y = Y[:-300]
    coefs = terps.fit_coefs(terpX,Y,alpha=alpha)
    static_preds = np.matmul(terpXend[-300:,:],np.transpose(coefs)) 
    MSE = np.mean(np.square(np.matmul(terpX,np.transpose(coefs))-Y))
    #print('log Trining MSE is '+str(np.log(MSE)))
    MSE = np.mean(np.square(static_preds-Yend[lag:]))
    #print('log Unseen MSE is '+str(np.log(MSE)))
    #terps.last_inp = np.array([Y[-20-lag],Y[-20]])
    recon,recostart = terps.auto_recon(coefs,Yend[:-200].tolist(),Nsteps=200,lag=lag)
    
    if make_plots:
        plt.plot(recostart,color='r')
        plt.plot(Yend[lag:],color='C0')
        plt.plot(static_preds,color='C1',linestyle='dashed')
        plt.vlines(100,ymin=-2,ymax=2,color='k',linestyle='dotted')
        plt.show()
        cap = min(2000,len(recon) - lag - 1)
        fig,axs=plt.subplots(1,1,figsize=(8,8))
        axs.plot(recon[:cap],recon[lag:cap+lag],linewidth=1,alpha=0.5,color='k')
        #axs.plot(recostart[:-lag],recostart[lag:],linewidth=1,alpha=0.5,color='r')
        #axs.plot([recostart[-lag]],[recostart[-1]],'o',color='r')
        #axs.plot([recon[0]],[recon[lag]],'o',color='k')
        #axs.plot(Y[-20:-10-lag],Y[-20+lag:-10],linewidth=1,alpha=0.3,color='C0')
        #axs.plot(Y[-10-lag:-lag],Y[-10:],linewidth=1,alpha=0.6,color='C0')
        axs.plot(Yend[lag:-lag],Yend[2*lag:],linewidth=1,alpha=0.2,color='C0')
        axs.plot(static_preds[:-lag],static_preds[lag:],alpha=0.2,color='C1',linestyle='dashed')
        plt.tight_layout()
        plt.show()
    
    return recon


def full_scan(system='Lorenz',LenTS=3000,lag=5):
    points = np.random.uniform(-2.5,2.5,size=(rand.randint(16,24),2))

    #FHs = [10**i for i in np.linspace(1,2,11)][2:-2]; FPs = [10**i for i in np.linspace(0,1.4,11)][:-3]
    #FFs = [10**i for i in np.linspace(0.75,1.25,5)][:-1]
    
    FHs = [10**i for i in [1.8,1.9,2.0]]
    FFs = [10**i for i in np.linspace(0.7,1.9,13)]
    FPs =  [10**i for i in np.linspace(0.8,2.0,13)]
    #print('Fhs: '+str(np.log10(FHs)))
    #print('Fps: '+str(np.log10(FPs)))
    #print('Ffs: '+str(np.log10(FFs)))
    #time.sleep(10)
    if system == 'Lorenz':
        TS = mapnsys.runge_Lorenz(LenTS+400,Sr=0.02)
        TS2 = mapnsys.runge_Lorenz(LenTS+300,Sr=0.02)

    elif system == 'Ikeda':
        TS = mapnsys.Ikeda(LenTS+400)
        TS2 = mapnsys.Ikeda(LenTS+300)

    elif system == 'Thomas':
        TS = mapnsys.runge_Thomas(LenTS+400)
        TS2 = mapnsys.runge_Thomas(LenTS+300)

    elif system == 'Halvorsen':
        TS = mapnsys.runge_Halvorsen(LenTS+400)
        TS2 = mapnsys.runge_Halvorsen(LenTS+300)

    elif system == 'Rossler':
        TS = mapnsys.runge_Rossler(LenTS+400)
        TS2 = mapnsys.runge_Rossler(LenTS+300)
    
    #verticies = np.array([[0,0]])
    terps = terpo.Terp_land2d(points,Fp=1,Fh=1,Ff=1,Fa=1)
    TSemb,Y = terps.embedd_TS(TS,lag=lag,horizon=1,plot=False)
    points = [ (TSemb+np.random.normal(0,0.2,size=TSemb.shape))[i,:] for i in range(TSemb.shape[0]) ]
    points = np.array(rand.sample(points,rand.randint(16,24)))
    
    count = 0
    for Fh in FHs:
        for Fp in FPs:
            for Ff in FFs:
                test_hypers(points, TS, TS2, Fh = Fh, Fp = Fp, Ff = Ff, system=system)
                count += 1
                print(str(count)+'/'+str(len(FHs)*len(FFs)*len(FPs))+' : (h,p,f) = ('+str(np.log10(Fh))+','+
                      str(np.log10(Fp))+','+str(np.log10(Ff))+')')
        time.sleep(15)
     
        
def test_recon_by_hypers(system='Lorenz',LenTS=3000, Ff = 1, Fp = 1, Fh = 1, points_provided=True, pointset = []):
    recons = []; lag = 5
    if points_provided:
        nGos = len(pointset)
    else:
        nGos = 5
    for go in range(nGos):
        print(go)
        if points_provided:
            points = pointset[go]
        else:
            points = np.random.uniform(-2.5,2.5,size=(rand.randint(15,20),2))
        if system == 'Lorenz':
            TS = mapnsys.runge_Lorenz(LenTS+400,Sr=0.02)
        elif system == 'Thomas':
            TS = mapnsys.runge_Thomas(LenTS+400)
        elif system == 'Halvorsen':
            TS = mapnsys.runge_Halvorsen(LenTS+400)
        elif system == 'Rossler':
            TS = mapnsys.runge_Rossler(LenTS+400)
        elif system == 'Lorenz96':
            TS = mapnsys.Lorenz96(LenTS+400)
        elif system == 'Mackey_G':
            TS = mapnsys.Mackey_glass(LenTS+400); lag = 8
        recon = test_hypers_just_recon(points, TS, Ff=Ff,Fh=Fh,Fp=Fp)
        recons.append(recon)
    fig,axs = plt.subplots(1,2,figsize=(10,5))
    axs[0].plot(TS[-1000:-lag],TS[-(1000-lag):],alpha=0.5,color='k')
    for recon in recons:
        axs[1].plot(recon[:-lag],recon[lag:],alpha=0.1)
    
    axs[1].set_title('(Ff,Fh,Fp) = ('+str(round(Ff,1))+','+str(round(Fh,1))+','+str(round(Fp,1))+')',fontsize=25)
    for i in range(2):
        axs[i].set_xlim(-2.5,2.5)
        axs[i].set_ylim(-2.5,2.5)
    plt.tight_layout()
    plt.show()
        

 

pointset = [np.random.uniform(-2.5,2.5,size=(rand.randint(15,24),2)) for i in range(10)]


test_recon_by_hypers(system='Lorenz',Ff=10**(1.1),Fh=63,Fp=10**(1.8),pointset=pointset) 
test_recon_by_hypers(system='Lorenz',Ff=10**(1.1),Fh=63,Fp=10**(1.7),pointset=pointset) 
test_recon_by_hypers(system='Lorenz',Ff=10**(1.1),Fh=63,Fp=10**(1.6),pointset=pointset) 
test_recon_by_hypers(system='Lorenz',Ff=10**(1.1),Fh=63,Fp=10**(1.5),pointset=pointset) 






print(1/0)
#test_recon_by_hypers(system='Lorenz',Ff=10**(1.1),Fh=63,Fp=10**(1.6),pointset=pointset) 
#test_recon_by_hypers(system='Rossler',Ff=10**(1.2),Fh=80,Fp=10**(1.9),pointset=pointset)
#test_recon_by_hypers(system='Halvorsen',Ff=10**(1.5),Fh=100,Fp=10**(2.0),pointset=pointset)
#test_recon_by_hypers(system='Thomas',Ff=10**(1.7),Fh=100,Fp=10**(1.1),pointset=pointset) 
#
#



for go in range(3):

    full_scan(system='Lorenz')
    grapo.param_scan(conditions={'system':'Lorenz'})     
    time.sleep(30)
    full_scan(system='Halvorsen')
    grapo.param_scan(conditions={'system':'Halvorsen'})   
    time.sleep(30)
    full_scan(system='Thomas')
    grapo.param_scan(conditions={'system':'Thomas'})   
    time.sleep(30)
    full_scan(system='Rossler')
    grapo.param_scan(conditions={'system':'Rossler'})   
    time.sleep(60)


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    