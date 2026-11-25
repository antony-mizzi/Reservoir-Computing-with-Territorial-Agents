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

#Ff=7,Fh=60,Fp=10
def just_make_homes(system = 'Lorenz', LenTS = 1000, territory_method = 'from_TS', N_vertex = 20, alpha = 10**(-12),
              lag = 5,Ff=7,Fh=60,Fp=10,Fa=0):
    if system == 'Lorenz':
        TS = mapnsys.runge_Lorenz(LenTS+400,Sr=0.02)
        TS2 = mapnsys.runge_Lorenz(LenTS,Sr=0.02)
    elif system == 'Ikeda':
        TS = mapnsys.Ikeda(LenTS+400)
    if territory_method == 'MDL':
        scoretrack = chaz.naturally_select(TS, TS, alpha=alpha,Ff=Ff,Fh=Fh,Fp=Fp,Fa=Fa)
        bs = chaz.review_scores_by_crit(scoretrack, LenTS, method = 'DL')
        verticies = bs['homes']
        helpo.log_homes(verticies, 'homes_MDL.csv')
    elif territory_method == 'AIC':
        scoretrack = chaz.naturally_select(TS, TS, alpha=alpha,Ff=Ff,Fh=Fh,Fp=Fp,Fa=Fa,method='AIC')
        bs = chaz.review_scores_by_crit(scoretrack, LenTS, method = 'AIC')
        verticies = bs['homes']
        helpo.log_homes(verticies, 'homes_AIC.csv')
    elif territory_method == 'from_TS':
        noisyTS = TS+np.random.normal(0,0.1,size=len(TS))
        noisyemb = terpo.embedd_TS(noisyTS,lag=lag)
        points = [ noisyemb[i,:] for i in range(noisyemb.shape[0]) ]
        verticies = np.array(rand.sample(points,N_vertex))
        helpo.log_homes(verticies, 'homes_random.csv')
    elif territory_method == 'random_uniform':
        points = [ np.random.uniform(-2.5,2.5,2) for i in range(N_vertex) ]
        verticies = np.array(rand.sample(points,N_vertex))
        helpo.log_homes(verticies, 'homes_uniform.csv')
        
def save_ideal_recon(system = 'Lorenz', LenTS = 1200):
    if system == 'Lorenz':
        TS = mapnsys.runge_Lorenz(LenTS+400,Sr=0.02)
        lag = 7; benchmark_err = 0.7
        suffix = '_Lorenz'
    elif system == 'Ikeda':
        TS = mapnsys.Ikeda(LenTS+400)
        suffix = '_Ikeda'
    elif system == 'Thomas':
        TS = mapnsys.runge_Thomas(LenTS+400)
        lag = 7; benchmark_err = None
        suffix = '_Thomas'
    elif system == 'Halvorsen':
        TS = mapnsys.runge_Halvorsen(LenTS+400)
        lag = 7; benchmark_err = 0.7
        suffix = '_Halvorsen'
    elif system == 'Rossler':
        TS = mapnsys.runge_Rossler(LenTS+400)
        lag = 5; benchmark_err = 0.15
        suffix = '_Rossler'
    elif system == 'Lorenz96':
        TS = mapnsys.Lorenz96(LenTS+400)
        lag = 8
        suffix = '_Lorenz96'
    elif system == 'Mackey_G':
        TS = mapnsys.Mackey_glass(LenTS+400)
        lag=6; benchmark_err = 0.15
        suffix = '_Mackey_G'
    
    recon = TS[-200:]
    helpo.log_recon(recon.tolist(), 'recons_ideal.csv', method='ideal', TS=system)
    
    
        
def get_recon(system = 'Lorenz', LenTS = 3000, territory_method = 'from_TS', N_vertex = 20, alpha = 10**(-12),
              lag = 1,Ff=4,Fh=30,Fp=10,Fa=0):
    
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
        TS = mapnsys.Mackey_glass(LenTS+400)
        TS2 = mapnsys.Mackey_glass(LenTS)
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
        home_file = 'homes2.csv'
        recon_file = 'recons2.csv'
        gen_file = 'gen_stats2.csv'
        
    elif territory_method == 'random_uniform':
        if system == 'Ikeda':
            points = [ np.random.uniform(-0.5,2,2) for i in range(N_vertex) ]
        else:
            points = [ np.random.uniform(-2.5,2.5,2) for i in range(N_vertex) ]
        verticies = np.array(points)
        home_file = 'homes2.csv'
        recon_file = 'recons2.csv'
        gen_file = 'gen_stats2.csv'

    
   
    terps = terpo.Terp_land2d(verticies,Fp=Fp,Fh=Fh,Ff=Ff,Fa=Fa,fit_inp = False)
    TSemb,Y = terps.embedd_TS(TS,lag=lag,horizon=1,plot=False)
    states,terpX=terps.process(TSemb,save_time=-300)
    
    
    usefull = np.var(terpX,axis=0)[:-1]
    n = len(usefull)//4
    usefull = usefull[:2*n].reshape(n, 2).prod(axis=1) * usefull[2*n:].reshape(n, 2).prod(axis=1)
    usless_number = len(np.where(usefull == 0)[0])
    
    terpXend = terpX[-300:,:]; terpX = terpX[:-300,:]; Yend = Y[-300-lag:]; Y = Y[:-300]
    coefs = terps.fit_coefs(terpX,Y,alpha=alpha)
    
    #grapo.component_embedding_and_territory_vis(terpX,Y,verticies,lag=lag)
    #terpo.terp_agent_plot(terpX, Y, 0, coefs)
    #terpo.terp_agent_plot(terpX, Y, 1, coefs)

    DL, MSE, Lpar, MSE_unb = deslo.get_total_DL(terpX, Y, alpha, epsilon=0.01,method='DL')
    #static_preds = np.matmul(terpXend[-300:,:],np.transpose(coefs)) 
    MSE = np.mean(np.square(np.matmul(terpX,np.transpose(coefs))-Y))
    print('log Training MSE is '+str(np.log(MSE)))
    #terps.last_inp = np.array([Y[-20-lag],Y[-20]])
    recon,recostart = terps.auto_recon(coefs,Yend[:-200].tolist(),Nsteps=200,lag=lag)
    #plt.plot(recostart,color='r')
    #plt.plot(Yend[lag:],color='C0')
    #plt.plot(static_preds,color='C1',linestyle='dashed')
    #plt.vlines(100,ymin=-2,ymax=2,color='k',linestyle='dotted')
    #plt.show()
    #cap = min(2000,len(recon) - lag - 1)
    #fig,axs=plt.subplots(1,1,figsize=(8,8),dpi=120)
    #axs.plot(recon[:cap],recon[lag:cap+lag],linewidth=1,alpha=0.5,color='k')
    #axs.plot(recostart[:-lag],recostart[lag:],linewidth=1,alpha=0.5,color='r')
    #axs.plot([recostart[-lag]],[recostart[-1]],'o',color='r')
    #axs.plot([recon[0]],[recon[lag]],'o',color='k')
    #axs.plot(Y[-20:-10-lag],Y[-20+lag:-10],linewidth=1,alpha=0.3,color='C0')
    #axs.plot(Y[-10-lag:-lag],Y[-10:],linewidth=1,alpha=0.6,color='C0')
    #axs.plot(Yend[lag:-lag],Yend[2*lag:],linewidth=1,alpha=0.2,color='C0')
    #axs.plot(static_preds[:-lag],static_preds[lag:],alpha=0.2,color='C1',linestyle='dashed')
    #plt.tight_layout()
    #plt.show()
    #print(1/0)
    terps.state = terps.homes+np.random.normal(0,0.3,size=terps.homes.shape)
    terps.velocities = np.random.normal(0,10,size=terps.homes.shape)
    states,terpX2=terps.process(TSemb,save_time=-200)
    try:
        FH = terpo.get_avg_forecasting_horizon(terps, coefs, TS2, lag=lag, benchmark_err=benchmark_err)
        conc = conco.get_concap(terpX[1000:,:], terpX2[1000:-300,:])
        print('Consistency Cap is '+str(conc)+' of '+str(4*len(terps.homes)+1))
        #FH = [i for i in range(len(Yend[100+lag:])-1) if abs(recostart[100+i]-Yend[100+lag+i])>0.8]
        #if len(FH) > 0:
        #    FH = min(FH)
        #else:
        #    FH = 200
        print('Forcasting Horizon is '+str(FH))
        #print('FORGOT TO CHANGE FH CALCULATION METHOD')
        
        TSemb,Y = terps.embedd_TS(TS2,lag=lag,horizon=1,plot=False)
        states,X=terps.process(TSemb,save_time=-200)
        NMSE = np.mean(np.square(np.matmul(X,np.transpose(coefs))-Y))/np.mean(np.square(Y))
        
        details = {'system':system,'method':territory_method,'N_terps':len(terps.homes),'Concap':conc,
                   'FH':FH,'NMSE_unseen':NMSE,'SC_penalty':Lpar,'MSE_unbiased_est':MSE_unb,
                   'useless_number':usless_number}
        helpo.log_data(details, gen_file)
    except:
        print('FAILED GETTING STATS')
    helpo.log_recon(recon.tolist(), recon_file, method=territory_method, TS=system,size=len(terps.homes))
    helpo.log_homes(verticies, home_file, method=territory_method, TS=system)
    print('FINISHED RUN')
    return verticies.shape[0]
    

#for trial in range(41):
#    print(trial)
#    for sys in ['Lorenz','Rossler','Mackey_G','Lorenz96','Halvorsen']:
#        save_ideal_recon(system=sys)
    

def run_random_territories_by_size():
    for verts in [5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]:
    
        #################################################### Lorenz
        Ff=20; Fh=80; Fp=60
        ####################################################
        
        v = get_recon(lag=5,alpha=10**(-8), system = 'Lorenz', territory_method = 'from_TS',Ff=Ff, Fh=Fh, Fp=Fp, N_vertex=verts)
        v = get_recon(lag=5,alpha=10**(-8), system = 'Lorenz', territory_method = 'random_uniform',Ff=Ff, Fh=Fh, Fp=Fp, N_vertex=verts)
       
        v = get_recon(lag=5,alpha=10**(-8), system = 'Halvorsen', territory_method = 'from_TS',Ff=Ff, Fh=Fh, Fp=Fp, N_vertex=verts)
        v = get_recon(lag=5,alpha=10**(-8), system = 'Halvorsen', territory_method = 'random_uniform',Ff=Ff, Fh=Fh, Fp=Fp, N_vertex=verts)
        
        v = get_recon(lag=5,alpha=10**(-8), system = 'Rossler', territory_method = 'from_TS',Ff=Ff, Fh=Fh, Fp=Fp, N_vertex=verts)
        v = get_recon(lag=5,alpha=10**(-8), system = 'Rossler', territory_method = 'random_uniform',Ff=Ff, Fh=Fh, Fp=Fp, N_vertex=verts)
        
        v = get_recon(lag=5,alpha=10**(-8), system = 'Mackey_G', territory_method = 'from_TS',Ff=Ff, Fh=Fh, Fp=Fp, N_vertex=verts)
        v = get_recon(lag=5,alpha=10**(-8), system = 'Mackey_G', territory_method = 'random_uniform',Ff=Ff, Fh=Fh, Fp=Fp, N_vertex=verts)
       
        print('---------------------------- FINISHED CYCLE with verts = '+str(verts)+' --------------')
        time.sleep(10)
        
    #grapo.recon_composites_by_method(Method='random_uniform',fname='recons.csv',f2='gen_stats.csv',
    #                               size=50,filter_by_size=True)
    grapo.FH_by_number_of_agents(fname='logs/gen_stats2.csv')
    time.sleep(180)
   

def run_evolved_territories():
    #################################################### Lorenz
    #Ff=12; Fh=63; Fp=40
    Ff=20; Fh=80; Fp=60
    ####################################################

    verts = get_recon(lag=5,alpha=10**(-8), system = 'Lorenz', territory_method = 'MDL_NSGA2',Ff=Ff, Fh=Fh, Fp=Fp)
    time.sleep(30)

    verts = get_recon(lag=5,alpha=10**(-8), system = 'Halvorsen', territory_method = 'MDL_NSGA2',Ff=Ff, Fh=Fh, Fp=Fp)
    time.sleep(30)

    verts = get_recon(lag=5,alpha=10**(-8), system = 'Rossler', territory_method = 'MDL_NSGA2',Ff=Ff, Fh=Fh, Fp=Fp)
    time.sleep(30)
    
    verts = get_recon(lag=5,alpha=10**(-8), system = 'Mackey_G', territory_method = 'MDL_NSGA2',Ff=Ff, Fh=Fh, Fp=Fp)
    grapo.recon_composites_by_method(Method='MDL_NSGA2',fname='recons_nsga.csv',f2='gen_stats_nsga.csv')
    grapo.NSGAplot(fname='logs/NSGA2log.csv')
    print('---------------------------- FINISHED CYCLE with verts = '+str(verts)+' --------------')
    time.sleep(120)
    



for go in range(49):
    run_random_territories_by_size()

#grapo.recon_composites_by_method(Method='random_uniform',fname='recons.csv',f2='gen_stats.csv',
#                                size=50,filter_by_size=True)
    
#grapo.recon_composites_by_method(Method='random_uniform',fname='recons.csv',f2='gen_stats.csv',
#                                size=60,filter_by_size=True)
#grapo.recon_composites_by_method(Method='from_TS',fname='recons.csv',f2='gen_stats.csv',
#                               size=30,filter_by_size=True)
for go in range(50):
    print(go)
    save_ideal_recon(system='Lorenz')
    save_ideal_recon(system='Mackey_G')
    save_ideal_recon(system='Rossler')
    save_ideal_recon(system='Halvorsen')

print(1/0)
grapo.recon_composites_by_method(Method='MDL_NSGA2',fname='recons_nsga.csv',f2='gen_stats_nsga.csv')
#grapo.FH_by_number_of_agents()
#run_evolved_territories()
print(1/0)
#grapo.recon_composites_by_method(Method='MDL_NSGA2',fname='recons_nsga.csv',f2='gen_stats_nsga.csv')


#for go in range(17):
#    run_evolved_territories() 
#    run_random_territories_by_size()    

for go in range(49):
    run_evolved_territories()  
    
    

    #run_evolved_territories()
    
    #################################################### Lorenz96
    #Ff=20; Fh=80; Fp=25
    ####################################################
    
    #################################################### Thomas 
    #Ff=50; Fh=100; Fp=13 
    ####################################################
    
    #################################################### Halvorsen
    #Ff=28; Fh=100; Fp=80
    ####################################################
    
    #################################################### Rossler
    #Ff=16; Fh=80; Fp=80
    ####################################################
    
    #################################################### Mackey_G
    #Ff=32; Fh=100; Fp=80
    ####################################################
    
    
    
    
    
    
