import numpy as np 
import Terps_2D as terpo
import Description_Lengther as deslo 
import maps_and_systems as mapnsys
import matplotlib.pyplot as plt
import helper_funcs as helpo
import time
import concap as conco
import random as rand
import agent_deleter as deleto
import scienceplots
plt.style.use(['science', 'no-latex'])

def delete_add_2(homes):
    if homes.shape[0] > 2:
        reduced = np.delete(homes, rand.randint(0,homes.shape[0]-1), axis=0)
        reduced = np.append(reduced, np.random.uniform(-2.5,2.5,size=(1,2)), axis=0)
        reduced = np.append(reduced, np.random.uniform(-2.5,2.5,size=(1,2)), axis=0)
    else:
        reduced = 1*homes
        reduced = np.append(reduced, np.random.uniform(-2.5,2.5,size=(1,2)), axis=0)
    return reduced
    

def naturally_select(TS,TS2,max_N = 51,sample_size=32,horizon=1,alpha=10**(-12),
                     Ff=7,Fh=60,Fp=10,Fa=0,method='DL',record_each_itt=False,setup={},lag=5):
    Vertex_list = [] 
    verticies = np.random.uniform(-2.5,2.5,size=(1,2))#np.array([[0,0]])
    terps = terpo.Terp_land2d(verticies,Fp=Fp,Fh=Fh,Ff=Ff,Fa=Fa)
    TSemb,Y = terps.embedd_TS(TS,lag=lag,horizon=horizon,plot=False)
    TSemb2,Y2 = terps.embedd_TS(TS2,lag=lag,horizon=horizon,plot=False)
    TSmax = np.max(TS); TSmin = np.min(TS)
    epsi = 10**(-1)
    score_track = []; can_get_paret = True
    for N in range(1,max_N):
        print('Running Epoch '+str(N)+' #terps = '+str(terps.homes.shape[0]))
        homes = terps.homes
        if terps.homes.shape[0] == 0:
            potentials = [np.random.uniform(-2.5,2.5,size=(1,2)) for v in range(sample_size)]
        else:
            potentials = [np.append(homes, v.reshape(1, 2), axis=0) for v in np.random.uniform(-2.5,2.5,size=(sample_size,2))]
        #potentials += [delete_add_2(homes) for mut in range(sample_size//2)]
        DLs = []
        for pot in potentials:
            terps = terpo.Terp_land2d(pot,Ff=Ff,Fh=Fh,Fp=Fp,Fa=Fa)
            #mask = terps.mask_ts(TSemb)
            states,X=terps.process(TSemb)
            DL, MSE, Lpar, MSE_unb = deslo.get_total_DL(X, Y, alpha, epsilon=epsi,method=method)
            DLs.append(DL)
        homes = potentials[DLs.index(min(DLs))]
        Vertex_list.append(homes)
        terps = terpo.Terp_land2d(homes,Ff=Ff,Fh=Fh,Fp=Fp,Fa=Fa)
        #try:
        #    terps.plot_homes(Y)
        #except:
        #    pass
        #mask = terps.mask_ts(TSemb)
        states,X2=terps.process(TSemb2)
        
        
        states,X=terps.process(TSemb,save_time=-300)
        MSE_gen = deslo.get_gen_err(X, Y, X2, Y2, alpha)

        
        new_epsi = deslo.get_epsi(X, Y)
        Rank = np.linalg.matrix_rank(X); Nvar = 4*homes.shape[0]
        
        #print('RANK X = '+str(Rank)+', size = '+str(4*homes.shape[0]))
        coefs = terps.fit_coefs(X,Y,alpha=alpha,transient=100)
        print('MSE GEN IS '+str(MSE_gen))
        
        SHmse = terps.short_recon_MSE(coefs,Y,lag=lag)
        print('Short horizon MSE is '+str(SHmse))
        MSE = np.mean(np.square(np.matmul(X[100:,:],np.transpose(coefs))-Y[100:]))
        Lpar = deslo.L_model_class(X, alpha,method=method)

        score_track.append({'MSE':MSE,'Lpar':Lpar,'N':homes.shape[0],'homes':homes,'MSEunseen':MSE_gen,'epsilon':new_epsi,
                            'SHmse':SHmse})
        print('Epoch '+str(N)+' #terps = '+str(homes.shape[0])+', log MSE = '+str(np.log(MSE)))
        if record_each_itt:
            Yend = Y[-300-lag:]
            recon,recostart = terps.auto_recon(coefs,Yend[:-200].tolist(),Nsteps=200,lag=lag)
            FH = terpo.get_avg_forecasting_horizon(terps, coefs, TS2, lag=lag)
            
            #plt.title('N = '+str((len(coefs)-1)/4)+', FH = '+str(FH)+', MSE = '+str(round(np.log(MSE),2))+' Lparam = '+str(round(Lpar,1)))
            #plt.plot(recostart,color='r')
            #plt.plot(Yend[lag:],color='C0')
            #plt.show()
            #plt.title('Short Forecast MSE = '+str(np.log(SHmse)))
            #plt.plot(recostart[100:-5],recostart[105:],color='k')
            #plt.plot(Yend[100:-5],Yend[105:],color='r')
            #plt.show()
            #try:
            #    conc = conco.get_concap(X[1000:-300:,:], X2[1000:-300,:])
            #except:
            #    conc = None
                
            details = {'system':setup['system'],'method':setup['method']+'_breathing','N_terps':len(terps.homes),
                   'Concap':None,'FH':FH,'NMSE_unseen':MSE_gen,'itteration':N}
            helpo.log_data(details, 'Island_stats.csv')

                
        if Rank < Nvar-1 and Nvar > 5:
            print('Rank Deficiency')
            homes = terps.homes[:-1,:]
            #terps = terpo.Terp_land2d(homes,Ff=Ff,Fh=Fh,Fp=Fp,Fa=Fa)
            # Delete least usefull coefficient
            #states,X=terps.process(TSemb)
            #homes = deleto.delete_by_parallel_crossings(TSemb, homes)
            if homes.shape[0] > 5:
                homes = deleto.delete_by_false_near_neighbour_crossings(TSemb, homes)
            else:
                absco = np.abs(deslo.fit_coefs(X, Y, alpha = alpha))
            #linv = np.linalg.inv(np.matmul(np.transpose(X),X)+np.diag(np.ones(X.shape[1]))*alpha)
                n = homes.shape[0]
                usefullness = np.square(absco)[:4*n]
            #usefullness = np.multiply(np.square(absco),np.reciprocal(np.diag(linv)))[:4*n]
            #usefullness = np.var(X[:,:-1],axis=0)
                usefullness = usefullness[:2*n].reshape(n, 2).sum(axis=1) + usefullness[2*n:].reshape(n, 2).sum(axis=1)
                least_usefull = np.argmin(usefullness)
                homes = np.delete(homes, least_usefull, axis=0)
            terps = terpo.Terp_land2d(homes,Ff=Ff,Fh=Fh,Fp=Fp,Fa=Fa)
          
        else:
            # Deleting any territories that recieve no driving signal
            usefull = np.var(X,axis=0)[:-1]
            n = len(usefull)//4
            usefull = usefull[:2*n].reshape(n, 2).prod(axis=1) * usefull[2*n:].reshape(n, 2).prod(axis=1)
            if homes.shape[0] > 3:
                if len(np.where(usefull == 0)[0]) > 0:
                    print('DELTED USELESS TERRITORY')
                    delete = np.where(usefull == 0)[0][0]
                    homes = np.delete(homes, delete, axis=0)
                    
            best_homes = homes
            MSE = deslo.get_gen_err(X, Y, X, Y, alpha)
            Lpar = deslo.L_model_class(X, alpha,method=method)
            paretod = gets_Paretod(score_track, Lpar, MSE, method='MDL', lenTS=len(TS))
            if paretod and can_get_paret:
                print('Got Paretod')
                #homes = terps.homes#[:-1,:]
                #homes = deleto.delete_by_parallel_crossings(TSemb, homes)
                if homes.shape[0] > 5:
                    homes = deleto.delete_by_false_near_neighbour_crossings(TSemb, homes)
                else:
                    usefullness = np.var(X[:,:-1],axis=0)

                    #usefullness = np.multiply(np.square(absco),np.reciprocal(np.diag(linv)))[:4*n]
                    usefullness = usefullness[:2*n].reshape(n, 2).sum(axis=1) + usefullness[2*n:].reshape(n, 2).sum(axis=1)
                    least_usefull = np.argmin(usefullness)
                    homes = np.delete(homes, least_usefull, axis=0)
                    
                #terps = terpo.Terp_land2d(homes,Ff=Ff,Fh=Fh,Fp=Fp,Fa=Fa)
                # Delete least usefull coefficient
                #states,X=terps.process(TSemb)
                
                #absco = np.abs(deslo.fit_coefs(X, Y, alpha = alpha))
                #linv = np.linalg.inv(np.matmul(np.transpose(X),X)+np.diag(np.ones(X.shape[1]))*alpha)
                #n = homes.shape[0]
                
                terps = terpo.Terp_land2d(homes,Ff=Ff,Fh=Fh,Fp=Fp,Fa=Fa)
                can_get_paret = False
            else:
                can_get_paret = True
                terps = terpo.Terp_land2d(homes,Ff=Ff,Fh=Fh,Fp=Fp,Fa=Fa)
            # Can drop all scores in score track that are not pareto optimal 
           #     states,X2=terps.process(TSemb2)
           #     MSE_gen = deslo.get_gen_err(X, Y, X2, Y2, alpha)
           #     score_track.append({'MSE':MSE,'Lpar':Lpar,'N':homes.shape[0],'homes':homes,'MSEunseen':MSE_gen})
           #     print('Epoch '+str(N)+' #terps = '+str(homes.shape[0])+', log generalisation MSE = '+str(np.log(MSE_gen)))
        #try:
        #    terps.plot_homes(TS,title='DL = '+str(int(min(DLs)))+', N = '+str(4*homes.shape[0])+', rank = '+str(np.linalg.matrix_rank(X)))
        #except:
        #    print('Couldnt plot')
        if new_epsi < epsi:
            epsi = 1*new_epsi
        #print('Epsi = '+str(epsi))
        if N % 20 == 0:
            #plt.plot([np.log(s['MSE']) for s in score_track],[s['Lpar'] for s in score_track],'o')
            #plt.show()
            #score_track = Pareto_clean(score_track)
            #plt.plot([np.log(s['MSE']) for s in score_track],[s['Lpar'] for s in score_track],'o')
            #plt.show()
            time.sleep(20)
    #retrospectively set epsilon to the smallest value of epsilon and re-calculate DLs 
    #pick the set of home locations from score_track that produced the minimum DL 
    #score_track = Pareto_clean(score_track)
    return score_track

def mutate(homes,TSemb,del1=0.1,del2=0.3,del3=0.1,add=0.5,do_pert=0.5,exp_N_perts=1,pert_strength=0.2,lag=5):
    not_changed = True
    if homes.shape[0] > 5:
        ranval = rand.random()
        if ranval < del1:
            homes = np.delete(homes, rand.randint(0,homes.shape[0]-1), axis=0)
        elif ranval < del1+del2:
            #homes = deleto.delete_by_false_near_neighbour_crossings(TSemb, homes, lag=lag)
            homes = deleto.delete_by_first_false_crossings(TSemb, homes, lag=lag)
        elif ranval < del1+del2+del3:
            homes = deleto.merge_random_neighbours(homes)
        not_changed = False
    if rand.random() < add:
        homes = np.append(homes, np.random.uniform(-2.5,2.5,size=(1,2)), axis=0)
        not_changed = False
    if rand.random() < do_pert or not_changed:
        Ngenes = homes.shape[0]
        pert = np.random.normal(0,pert_strength,size=(Ngenes,2))
        N_to_pert = np.random.binomial(Ngenes-1,exp_N_perts/(Ngenes-1))+1
        idx = np.random.choice(Ngenes, size=N_to_pert, replace=False)
        mask = np.zeros(Ngenes, dtype=bool)
        mask[idx] = True
        pert_masked = pert.copy()
        pert_masked[~mask] = 0
        homes += pert_masked
        #homes+= np.random.normal(0,pert_strength,size=homes.shape)
        homes = np.clip(homes, -2.499, 2.499)
    return homes

def softmax(X):
    X = np.exp(X)
    return X/np.sum(X)

def Randomly_generation_selection(TS,TS2,horizon=1,alpha=10**(-12),Ff=7,Fh=60,Fp=10,Fa=0,method='DL',
                                  record_each_itt=True,setup={},lag=5,N_samples=50,sample_size=42):
    verticies = np.random.uniform(-2.5,2.5,size=(8,2))
    terps = terpo.Terp_land2d(verticies,Fp=Fp,Fh=Fh,Ff=Ff,Fa=Fa)
    TSemb,Y = terps.embedd_TS(TS,lag=lag,horizon=horizon,plot=False)
    #points = [ (TSemb+np.random.normal(0,0.2,size=TSemb.shape))[i,:] for i in range(TSemb.shape[0]) ]
    LenTS = len(Y)
    epsi = 0.01; scores = []
    TSemb2,Y2 = terps.embedd_TS(TS2,lag=lag,horizon=horizon,plot=False)
    for sample_num in range(N_samples):
        if sample_num == 0:
            sampsize = 49
        else:
            sampsize = sample_size
        sample = [np.random.uniform(-2.5,2.5,size=(rand.randint(6,20),2)) for j in range(sampsize)]
        for homes in sample:
            terps = terpo.Terp_land2d(homes,Ff=Ff,Fh=Fh,Fp=Fp,Fa=Fa)
            states,X=terps.process(TSemb)
            DL,MSE,Lpar,MSEunb = deslo.get_total_DL(X, Y, alpha, epsilon=epsi,method=method)
            scores.append({'DL':DL,'MSE':MSE,'MSEunb':MSEunb,'Lpar':Lpar,'homes':homes})
        if min([s['MSEunb'] for s in scores]) < epsi:
            epsi = 1*min([s['MSEunb'] for s in scores])
        DLs = [(LenTS/2-50)*s['MSE']/epsi + s['Lpar'] for s in scores]
        #elif method == 'AIC':
        #    DLs = [LenTS*np.log(s['MSE']) + s['Lpar'] for s in allscores]
        best_ind = DLs.index(min(DLs))
        best_homes = scores[best_ind]['homes']
        print('# of terps is '+str(best_homes.shape[0]))
        if record_each_itt:
            MSE = scores[best_ind]['MSE']
            terps = terpo.Terp_land2d(best_homes,Ff=Ff,Fh=Fh,Fp=Fp,Fa=Fa)
            states,X2=terps.process(TSemb2)
            states,X=terps.process(TSemb)
            coefs = terps.fit_coefs(X,Y,alpha=alpha)
            MSE_gen = np.mean(np.square(np.matmul(X2[100:,:],np.transpose(coefs))-Y2[100:]))
            Yend = Y[-300-lag:]
            FH = terpo.get_avg_forecasting_horizon(terps, coefs, TS2, lag=lag)
            recon,recostart = terps.auto_recon(coefs,Yend[:-200].tolist(),Nsteps=200,lag=lag)
            
                
            details = {'system':setup['system'],'method':setup['method']+' (brute force)','N_terps':len(terps.homes),
                   'Concap':None,'FH':FH,'NMSE_unseen':MSE_gen,'itteration':sample_num}
            helpo.log_data(details, 'Island_stats.csv')
    return best_homes
            
def NSGA2(TS,generations=50,pop_size=50,horizon=1,alpha=10**(-12),lag=5,method='DL',Ff=7,Fh=60,Fp=10,
          Fa=0,del1=0.1,del2=0.3,del3=0.1,add=0.5,do_pert=0.2,exp_N_perts=2,pert_strength=0.2,starting_pop=50,
          system='Lorenz'):
    
    #EMBEDDING TIME SERIES
    verticies = np.random.uniform(-2.5,2.5,size=(8,2))
    terps = terpo.Terp_land2d(verticies,Fp=Fp,Fh=Fh,Ff=Ff,Fa=Fa)
    TSemb,Y = terps.embedd_TS(TS,lag=lag,horizon=horizon,plot=False)
    
    #INITIALISING POPULATION
    population = [np.random.uniform(-2.5,2.5,size=(rand.randint(15,70),2)) for j in range(starting_pop)]
    MSEs = []; penalties = []; pareto_front = []
    
    for generation in range(generations):
        
        #EVALUATING PERFORMANCE AND COMPEXITY PENALTY
        if generation == 0:
            for h in range(starting_pop):
                homes = population[h]
                terps = terpo.Terp_land2d(homes,Ff=Ff,Fh=Fh,Fp=Fp,Fa=Fa)
                states,X=terps.process(TSemb)
                
                # Deleting any territories that recieve no driving signal
                usefull = np.var(X,axis=0)[:-1]
                n = len(usefull)//4
                usefull = usefull[:2*n].reshape(n, 2).prod(axis=1) * usefull[2*n:].reshape(n, 2).prod(axis=1)
 
                for delind in np.where(usefull == 0)[0][::-1]:
                    if len(homes) > 5:
                        homes = 1*np.delete(homes, delind, axis=0)
                population[h] = homes
                    
                DL,MSE,Lpar,MSEunb = deslo.get_total_DL(X, Y, alpha, epsilon=0.1 ,method=method)
                MSEs.append(np.log(MSE)); penalties.append(Lpar)
        else:
            for h in range(pop_size//2):
                homes = population[h]
                terps = terpo.Terp_land2d(homes,Ff=Ff,Fh=Fh,Fp=Fp,Fa=Fa)
                states,X=terps.process(TSemb)
                
                # Deleting any territories that recieve no driving signal
                usefull = np.var(X,axis=0)[:-1]
                n = len(usefull)//4
                usefull = usefull[:2*n].reshape(n, 2).prod(axis=1) * usefull[2*n:].reshape(n, 2).prod(axis=1)
                for delind in np.where(usefull == 0)[0][::-1]:
                    if len(homes) > 5:
                        homes = 1*np.delete(homes, delind, axis=0)
                population[h] = homes
                    
                DL,MSE,Lpar,MSEunb = deslo.get_total_DL(X, Y, alpha, epsilon=0.1 ,method=method)
                MSEs.append(np.log(MSE)); penalties.append(Lpar)
        
        #GETTING FINAL RANKING SCORE (combines front number and manhattan distance)
        scores,front1 = pareto_front_sorter(MSEs, penalties)
        if generation == 0:
            gen1C1 = MSEs; gen1C2 = penalties
            first_front = front1
        
        #DISINCENTIVISING REPRODUCTION OF EFFICITVE DUPLICATES
        for s in range(1,len(scores)):
            if scores[s] in scores[:s]:
                scores[s] += 1000
            

        #SAVING POSSIBLE PARETO OPTIMAL SOLUTIONS
        for ind in front1:  #!!!!!!!!
            pareto_front.append({'MSE':MSEs[ind],'pen':penalties[ind],'homes':population[ind]})
        pareto_front = Pareto_clean2(pareto_front)

        finalC1front = [p['MSE'] for p in pareto_front]
        finalC2front = [p['pen'] for p in pareto_front]
        
        #good_front = Pareto_subsetter(finalC1front, finalC2front)
        #print('Size of possible min DL solns is '+str(len(good_front))) 
        
        #TOURNAMENT TO SELECT PARENTS
        #if generation != generations-1:
        #    shuf = list(range(pop_size))
        #    rand.shuffle(shuf)
        #    matchups = [[shuf[2*i],shuf[2*i+1]] for i in range(pop_size//2)]
        #    winners = []
        #    for pair in matchups:
        #        pair_scores = [scores[pair[0]],scores[pair[1]]]
        #        winners.append( pair[pair_scores.index(min(pair_scores))] )
        #    newpopulation = [population[w] for w in winners]
        #    MSEs = [MSEs[w] for w in winners]; penalties = [penalties[w] for w in winners]
        #    for winner in winners:
        #        newpopulation.append(mutate(population[winner],TSemb))
        #    population = newpopulation
        #time.sleep(10)
        
        #SELECTION BY RANKING 
        if generation != generations-1:
            sorted_scores = sorted(scores)
            
            #winners = [i for i in range(pop_size) if scores[i] <= sorted_scores[pop_size//2-1]] #!!!
            winners = [scores.index(sorted_scores[i]) for i in range(pop_size//2)]

            newpopulation = [population[w] for w in winners]
            MSEs = [MSEs[w] for w in winners]; penalties = [penalties[w] for w in winners]
            for winner in winners:
                newpopulation.append(mutate(population[winner],TSemb,del1=del1,del2=del2,del3=del3,
                                            add=add,do_pert=do_pert,exp_N_perts=exp_N_perts,
                                            pert_strength=pert_strength, lag=lag))
            population = newpopulation + []
            print(str(generation)+':'+str(len(population)))

        
        
    print('Size of Pareto front before cleaning '+str(len(pareto_front)))    
    pareto_front = Pareto_clean2(pareto_front)
    print('Size of Pareto front is '+str(len(pareto_front))) 
    finalC1front = [p['MSE'] for p in pareto_front]
    finalC2front = [p['pen'] for p in pareto_front]
    
    good_front = Pareto_subsetter(finalC1front, finalC2front)
    print('Size of possible min DL solns is '+str(len(good_front))) 
    
    try:
        make_Pareto_front_plot(gen1C1, gen1C2, first_front, finalC1front, finalC2front, good_front)
    except:
        print('PLOT FAILED')

    good_front = [pareto_front[i] for i in good_front]
    
    #FINAL SELECTION FROM PARETO FRONT:
    
    epstimates = []
    for soln in good_front:#pareto_front:
        terps = terpo.Terp_land2d(soln['homes'],Ff=Ff,Fh=Fh,Fp=Fp,Fa=Fa)
        states,X=terps.process(TSemb)
        coefs = terpo.fit_coefs_pinv(X,Y,alpha=10**(-16))
        MSE = np.mean(np.square(np.matmul(X,np.transpose(coefs))-Y))
        epstimate = MSE*(len(Y)/(len(Y)-4*len(soln['homes'])))
        epstimates.append(epstimate)
    epsi_unb = min(epstimates)
    print('epsi = '+str(np.log(epsi_unb)))
    if False:
        SCs = [len(Y)*soln['MSE']/(2*epsi_unb)+soln['pen'] for soln in good_front]
        mind = SCs.index(min(SCs))
        final_model_homes = good_front[mind]['homes']
    if True:
        longMSEs = {}; lengths = list(range(2,31)); picks = {}
        C1s = [p['MSE'] for p in good_front]; C2s = [p['pen'] for p in good_front]
        coeflist = []
        for soln in good_front:#pareto_front:
            terps = terpo.Terp_land2d(soln['homes'],Ff=Ff,Fh=Fh,Fp=Fp,Fa=Fa)
            states,X=terps.process(TSemb)
            coefs = terps.fit_coefs(X,Y,alpha=alpha)
            coeflist.append(coefs)
            soln['FH'] = terpo.get_avg_forecasting_horizon(terps, coefs, TS, lag=lag)
        for length in lengths:
            temp_LMSE = []
            for s in range(len(good_front)):#pareto_front:
                soln = good_front[s]
                terps = terpo.Terp_land2d(soln['homes'],Ff=Ff,Fh=Fh,Fp=Fp,Fa=Fa)
                coefs = coeflist[s]
                longMSE = terps.short_recon_MSE(coefs, Y, lag=lag,length=length)
                temp_LMSE.append(longMSE)
            optind = temp_LMSE.index(min(temp_LMSE))
            #print('Optind is '+str(optind))
            effect_eps = effective_epsilon(C1s, C2s, optind)
            #print('len = '+str(length)+', effective epsilon = '+str(effect_eps/3000))
            longMSEs[length] = {'MSE':good_front[optind]['MSE'],'pen':good_front[optind]['pen'],
                                'epsilon':effect_eps/3000,'FH':good_front[optind]['FH']}
            picks[length]=optind
            
        
        details = {'system':system,'generations':generations,'pop_size':pop_size,'del1':del1,'del2':del2,
                   'del3':del3,'add':add,'do_pert':do_pert,'exp_N_perts':exp_N_perts,
                   'pert_strength':pert_strength,'front_size':len(pareto_front),'espilon_unb':epsi_unb}
        
        for length in lengths:
            details['e'+str(length)] = longMSEs[length]['epsilon']
            details['m'+str(length)] = longMSEs[length]['MSE']
            details['p'+str(length)] = longMSEs[length]['pen']
            details['h'+str(length)] = longMSEs[length]['FH']
        helpo.log_data(details, 'NSGA2log.csv')
        final_model_homes = good_front[picks[5]]['homes']
    
    #return good_front[longMSEs.index(min(longMSEs))]['homes']#pareto_front[longMSEs.index(min(longMSEs))]['homes']
    return final_model_homes 
    

def make_Pareto_front_plot(gen1C1,gen1C2,first_front,finalC1,finalC2,good_front):
    fig,axs=plt.subplots(1,1,figsize=(10,6),dpi=120)
    paret1 = [gen1C1[i] for i in range(len(gen1C1)) if i in first_front]
    paret2 = [gen1C2[i] for i in range(len(gen1C1)) if i in first_front]
    x = sorted(gen1C1)
    y = [gen1C2[gen1C1.index(g)] for g in x]
    axs.plot(x,y,'x',color='C0',label='Starting Population',markersize=10,alpha=1)
    x = [i for i in x if i in paret1]; y = [i for i in y if i in paret2]
    axs.plot(x,y,'x',color='C3',label='Starting Pareto Front',markersize=10,alpha=1)
    axs.plot(x,y,color='C3',alpha=0.5,linestyle='dashed')
    
    xg = [finalC1[i] for i in range(len(finalC1)) if i in good_front]
    yg = [finalC2[i] for i in range(len(finalC2)) if i in good_front]
    #print('x: '+str(x))
    #print('y: '+str(y))
    #print('xg: '+str(xg))
    #print('yg: '+str(yg))
    #print([i for i in x if i not in xg])
    #print([i for i in y if i not in yg])
    x = sorted(finalC1)
    y = [finalC2[finalC1.index(g)] for g in x]
    axs.plot(x,y,color='C1',alpha=0.5,linestyle='dashed')

    #axs.plot([i for i in x if i not in xg],[i for i in y if i not in yg],'o',color='C1',
    #         label='Final Pareto Front',markersize=8)
    axs.plot(x,y,'o',color='C1',
             label='Final Pareto Front',markersize=12,alpha=1)
    axs.plot(xg,yg,'*',color='C2',label='MDL subset',markersize=12)
    axs.set_xlabel('Log 1-step MSE',fontsize=22)
    axs.set_ylabel('Complexity Penalty',fontsize=22)
    axs.legend(fontsize=18)
    axs.tick_params(axis='both', labelsize=16)
    plt.tight_layout()
    plt.show()
    
    
    
def Terpstien_island(TS,TS2,generations=50,sample_size=50,horizon=1,alpha=10**(-12),
                     Ff=7,Fh=60,Fp=10,Fa=0,method='DL',record_each_itt=True,setup={},lag=5,
                     Nislands=4,island_pop=8):
    verticies = np.random.uniform(-2.5,2.5,size=(8,2))
    terps = terpo.Terp_land2d(verticies,Fp=Fp,Fh=Fh,Ff=Ff,Fa=Fa)
    TSemb,Y = terps.embedd_TS(TS,lag=lag,horizon=horizon,plot=False)
    #points = [ (TSemb+np.random.normal(0,0.2,size=TSemb.shape))[i,:] for i in range(TSemb.shape[0]) ]
    LenTS = len(Y)
    TSemb2,Y2 = terps.embedd_TS(TS2,lag=lag,horizon=horizon,plot=False)
    #TSmax = np.max(TS); TSmin = np.min(TS)#; epsi = 10**(-3)
    epsi = 0.1
    
    Islands = {i:[np.random.uniform(-2.5,2.5,size=(rand.randint(6,25),2)) for j in range(island_pop)] for i in range(Nislands)}
    Epsis = {i:10**(-2-(3/(Nislands-1))*i) for i in range(Nislands)}
    epsis = {i:10**(-3-0.5*i) for i in range(Nislands)}
    pareto_front = []
    for gen in range(generations):
        print('Total examined: '+str(island_pop*Nislands*(gen+1))+', size of pareto front: '+str(len(pareto_front)))
        if gen % 10 == 0:
            time.sleep(10)
        allscores = []
        print('------------------------------------- gen '+str(gen)+'------------------')
        #print('Islands are: ')
        #print(Islands)
        for isl in range(Nislands):
            scores = []; ind = 0
            for homes in Islands[isl]:
                terps = terpo.Terp_land2d(homes,Ff=Ff,Fh=Fh,Fp=Fp,Fa=Fa)
                states,X=terps.process(TSemb)
                
                
                # Deleting any territories that recieve no driving signal
                usefull = np.var(X,axis=0)[:-1]
                n = len(usefull)//4
                usefull = usefull[:2*n].reshape(n, 2).prod(axis=1) * usefull[2*n:].reshape(n, 2).prod(axis=1)
                if len(np.where(usefull == 0)[0]) > 0:
                    delete = np.where(usefull == 0)[0][0]
                    homes = np.delete(homes, delete, axis=0)

                
                DL,MSE,Lpar,MSEunb = deslo.get_total_DL(X, Y, alpha, epsilon=Epsis[isl],method=method)
                
                if not gets_Paretod(pareto_front, Lpar, MSE, method=method):
                    coefs = terps.fit_coefs(X,Y,alpha=alpha)
                    longMSE = terps.short_recon_MSE(coefs, Y, lag=lag)
                    print(np.log(longMSE))
                    pareto_front.append({'DL':DL,'MSE':MSE,'MSEunb':MSEunb,'Lpar':Lpar,'homes':homes,
                                         'longMSE':longMSE})
                #if np.linalg.matrix_rank(X) < 4*homes.shape[0] -1:
                #    DL+= 0.5*abs(DL)
                scores.append({'DL':DL,'MSE':MSE,'MSEunb':MSEunb,'Lpar':Lpar,'homes':homes})
                ind += 1
            zscores = np.array([s['DL'] for s in scores])
            new_homes = [scores[np.argmin(zscores)]['homes']]
            zscores -= np.mean(zscores)
            zscores = softmax(-zscores/np.std(zscores))
            for mut in range(island_pop-1):
                #print(rand.choices(Islands[isl],weights=zscores,k=1)[0])
                new_homes.append(mutate( 1*rand.choices(Islands[isl],weights=zscores,k=1)[0] , TSemb))
            allscores += scores
            Islands[isl] = new_homes; epsis[isl] = min([s['MSEunb'] for s in scores])
            #epsis[isl] = min([s['MSE']*(LenTS/(LenTS-4*s['homes'].shape[0])) for s in scores])
            

        
        for isl in range(Nislands):
            if rand.random() < 0.2:
                others = list(range(Nislands))
                others.remove(isl)
                Islands[isl].append(rand.choice(Islands[rand.choice(others)]))
            
        
        #print(min([a['MSE'] for a in allscores]))
        #print(min([a['MSEunb'] for a in allscores]))
        if min(epsis.values()) < epsi:
            epsi = 1*min(epsis.values())
        #print('EPSI IS '+str(epsi))
        #print('FOR EACH ISLAND: ')
        #print(Epsis)
        Epsis = {k:np.exp(0.92*np.log(Epsis[k])+0.08*np.log((1.5-0.3*k)*epsi)) for k in Epsis.keys()}
        
        if record_each_itt or gen == generations-1:

            #if method == 'DL':
            DLs = [(LenTS/2-50)*s['MSE']/epsi + s['Lpar'] for s in allscores]
            #elif method == 'AIC':
            #    DLs = [LenTS*np.log(s['MSE']) + s['Lpar'] for s in allscores]
            best_ind = DLs.index(min(DLs))
            best_homes = allscores[best_ind]['homes']
            print('FOR BEST TERPS: Lparam = '+str(round(allscores[best_ind]['Lpar'],1))+', Lerr = '+
                  str(round((LenTS/2-50)*allscores[best_ind]['MSE']/epsi,1)))
            print('# of terps is '+str(best_homes.shape[0]))
            if record_each_itt:
                MSE = allscores[best_ind]['MSE']
                terps = terpo.Terp_land2d(best_homes,Ff=Ff,Fh=Fh,Fp=Fp,Fa=Fa)
                states,X2=terps.process(TSemb2)
                states,X=terps.process(TSemb)
                coefs = terps.fit_coefs(X,Y,alpha=alpha)
                MSE_gen = np.mean(np.square(np.matmul(X2[100:,:],np.transpose(coefs))-Y2[100:]))
                Yend = Y[-300-lag:]
                FH = terpo.get_avg_forecasting_horizon(terps, coefs, TS2, lag=lag)
                recon,recostart = terps.auto_recon(coefs,Yend[:-200].tolist(),Nsteps=200,lag=lag)
                
                #plt.title('Gen '+str(gen)+', FH = '+str(FH)+', N = '+str(len(best_homes)))
                #plt.plot(recostart,color='r')
                #plt.plot(Yend[lag:],color='C0')
                #plt.show()
                #plt.plot(recostart[100:-5],recostart[105:],color='k')
                #plt.plot(Yend[100:-5],Yend[105:],color='r')
                #plt.show()

                    
                details = {'system':setup['system'],'method':setup['method'],'N_terps':len(terps.homes),
                       'Concap':None,'FH':FH,'NMSE_unseen':MSE_gen,'itteration':gen}
                helpo.log_data(details, 'Island_stats.csv')
    longMSEs = [s['longMSE'] for s in pareto_front]
    best_ind = longMSEs.index(min(longMSEs))
    print('Final choice is entry number '+str(best_ind)+'/'+str(len(pareto_front)))
    best_homes = pareto_front[best_ind]['homes']
    return best_homes
        
            
    
def Pareto_clean(scores):
    delind = []
    for k1 in range(len(scores)-1):
        for k2 in range(k1+1,len(scores)):
            if scores[k1]['MSE']>scores[k2]['MSE'] and scores[k1]['Lpar']>=scores[k2]['Lpar']:
                delind.append(k1)
            elif scores[k2]['MSE']>scores[k1]['MSE'] and scores[k2]['Lpar']>=scores[k1]['Lpar']:
                delind.append(k2)
    scores = [scores[i] for i in range(len(scores)) if i not in delind]
    return scores

def Pareto_subsetter(crit1,crit2):
    intersects = []; subset = []
    for i in range(len(crit1)-1):
        for j in range(i+1,len(crit1)):
            if not np.exp(crit1[i])-np.exp(crit1[j]) == 0:
                intersects.append((crit2[j]-crit2[i])/(np.exp(crit1[i])-np.exp(crit1[j])))
    intersects = intersects + [0] + [2*max(intersects)]
    intersects = sorted(intersects)
    midPs = [(intersects[i]+intersects[i+1])/2 for i in range(len(intersects)-1)]
    for eps in midPs:
        vals = [eps*np.exp(crit1[i])+crit2[i] for i in range(len(crit1))]
        #print(vals.index(min(vals)))
        subset.append(vals.index(min(vals)))
    #print(subset)
    subset = list(set(subset))
    return subset

def effective_epsilon(crit1,crit2,optind):
    intersects = []; subset = []
    for i in range(len(crit1)-1):
        for j in range(i+1,len(crit1)):
            if not np.exp(crit1[i])-np.exp(crit1[j]) == 0:
                intersects.append((crit2[j]-crit2[i])/(np.exp(crit1[i])-np.exp(crit1[j])))
    intersects = intersects + [0] + [2*max(intersects)]
    intersects = sorted(intersects)
    midPs = [(intersects[i]+intersects[i+1])/2 for i in range(len(intersects)-1)]
    for eps in midPs:
        vals = [eps*np.exp(crit1[i])+crit2[i] for i in range(len(crit1))]
        #print(vals.index(min(vals)))
        subset.append(vals.index(min(vals)))
    #print('subset is : '+str(subset))
    correct_eps = [midPs[i] for i in range(len(midPs)) if subset[i] == optind]
    try:
        out = (np.min(correct_eps)+np.max(correct_eps))/2
    except:
        out = intersects[-2]
        print('EFFECTIVE EPSILON THING FAILED')
    return out


def Pareto_clean2(Pfront):
    
    delind = []; N = len(Pfront)
    for i in range(N):
        for j in range(N):
            if Pfront[j]['MSE'] < Pfront[i]['MSE'] and Pfront[j]['pen'] < Pfront[i]['pen']:
                delind.append(i)
        for j in range(i):
            if Pfront[j]['MSE'] == Pfront[i]['MSE'] and Pfront[j]['pen'] == Pfront[i]['pen']:
                delind.append(j) 
    scores = [Pfront[i] for i in range(N) if i not in delind]
    return scores

def pareto_front_sorter(crit1,crit2):
    #fig,axs=plt.subplots(1,1,figsize=(6,6))
    #plt.plot(crit1,crit2,'o',color='grey',markersize=8)
    N = len(crit1)
    range1 = max(crit1)-min(crit1); range2 = max(crit2)-min(crit2)
    paretoed_by = [[] for i in range(N)]
    for i in range(N):
        for j in range(N):
            if crit1[j] < crit1[i] and crit2[j] < crit2[i]:
                paretoed_by[i].append(j)
    fronts = []; keep_going = True; allocated = []; manhattans = [0 for i in range(N)]
    while keep_going:
        front = [i for i in range(N) if len(paretoed_by[i]) == 0 and i not in allocated]
        allocated += front
        fronts.append(front)
        lst1 = [crit1[i] for i in front]; lst2=[crit2[i] for i in front]
        MHdists = manhattener(lst1, lst2, range1, range2)
        for ind in range(len(front)):
            manhattans[front[ind]] = MHdists[ind]
            
        #plt.plot(lst1,lst2,'X',markersize=5,label=str(len(fronts)),color='C'+str(len(fronts)))
        #plt.plot(lst1,lst2,color='C'+str(len(fronts)))
        
        paretoed_by = [[j for j in p if j not in front] for p in paretoed_by]
        if len(front) == 0:
            keep_going = False
            remainders = [i for i in range(N) if i not in allocated]
            if len(remainders) > 0:
                fronts.append(remainders)
                lst1 = [crit1[i] for i in remainders]; lst2=[crit2[i] for i in remainders]
                MHdists = manhattener(lst1, lst2, range1, range2)
                for ind in range(len(remainders)):
                    manhattans[remainders[ind]] = MHdists[ind]
                
                plt.plot(lst1,lst2,'X',color='k',markersize=5,label=str(len(fronts)))
                #plt.plot(lst1,lst2,color='k')
    #plt.legend()
    #axs.set_xlim(-10,0)
    #axs.set_ylim(0,3000)
    #plt.show()
    #print('fronts: '+str(fronts))
    #print('Fronts were: ')
    #print(fronts)
    Pareto_front = fronts[0]
    #print('Pareto front is '+str(Pareto_front))
    f1c1 = [crit1[i] for i in fronts[0]]; f2c2 = [crit2[i] for i in fronts[0]]
    mdlss = Pareto_subsetter(f1c1, f2c2)
    #print('mdlss: '+str(mdlss))
    mdlss = [fronts[0][i] for i in mdlss]
    print('mdlss: '+str(mdlss))
    second = [i for i in fronts[0] if i not in mdlss]
    #print('second: '+str(second))
    fronts = [mdlss] + [second] + fronts[1:]
    #print('Fronts now: ')
    #print(fronts)
    
    front_rank = [None] * N
    for idx, sub in enumerate(fronts):
        for x in sub:
            front_rank[x] = idx
    #print('front_rank: '+str(front_rank))
    #print('')
    final_ranking_score = [10*front_rank[i]+manhattans[i] for i in range(N)]
    return final_ranking_score, Pareto_front


def manhattener(lst1,lst2,range1,range2):
    MHdists = []
    for i in range(len(lst1)):
        if lst1[i] == min(lst1): 
            MHdists.append(0)
        elif lst2[i] == min(lst2):
            MHdists.append(1)
        else:
            try:
                dist1s = [l-lst1[i] for l in lst1[:i]+lst1[i+1:]]
                dist2s = [l-lst2[i] for l in lst2[:i]+lst2[i+1:]]
                Left1 = min([abs(d) for d in dist1s if d <= 0])/range1
                Up2 = min([abs(d) for d in dist2s if d <= 0])/range2
                Right1 = min([abs(d) for d in dist1s if d >= 0])/range1
                Down2 = min([abs(d) for d in dist2s if d >= 0])/range2
                MHdists.append(3-(Left1+Up2+Right1+Down2))
            except: 
                print('BROKE MANHATTAN CONDITION')
                MHdists.append(3)
    return MHdists

def gets_Paretod(scores,Lparam,MSE,method='MDL',lenTS=1000):
    paretod = False
    if len(scores) >= 1:
        if method == 'MDL' or method == 'DL':
            for score in scores:
                if score['MSE']<=MSE and score['Lpar']<=Lparam:
                    paretod = True
        elif method == 'AIC':
            for score in scores:
                if lenTS*np.log(score['MSE'])+score['Lpar']< lenTS*np.log(MSE)+Lparam:
                    paretod = True
    return paretod

def get_best_candidate(scores,TS):
    epsilon = min([s['epsilon'] for s in scores])
    for s in scores:
        s['DL'] = len(TS)*s['MSE']/(2*epsilon)+s['Lpar']
    DLs = [s['DL'] for s in scores]
    best_score = scores[DLs.index(min(DLs))]
    return best_score

def review_scores_by_crit(scores,LenTS,method='DL'):
    #epsilon = min([s['epsilon'] for s in scores])
    #print('IN REVIEW EPSILON IS '+str(epsilon))
    #if method == 'DL':
    #    for s in scores:
    #        s['DL'] = LenTS*s['MSE']/(2*epsilon)+s['Lpar']
    #    DLs = [s['DL'] for s in scores]
    #elif method == 'AIC':
    #    for s in scores:
    #        s['DL'] = LenTS*s['MSE']/(2*epsilon)+s['Lpar']
    #    DLs = [s['DL'] for s in scores]
    DLs = [s['SHmse'] for s in scores]
    bs = scores[DLs.index(min(DLs))]
    return bs

def Compare_selection_criteria(max_N = 31,sample_size=50,horizon=1,alpha=10**(-12),
                               LenTS = 5000,system='Lorenz'):
    if system == 'Lorenz':
        TS = mapnsys.runge_Lorenz(LenTS,dt=0.02)
        TS2 = mapnsys.runge_Lorenz(5000,dt=0.02)
        
    print('USING AIC: ')
    score_track = naturally_select(TS,TS2,max_N=max_N,sample_size=sample_size,
                                   horizon=horizon,alpha=alpha,method='AIC') 
    bs = review_scores_by_crit(score_track, LenTS, method = 'AIC')
    details = {'Method':'AIC','MSE':bs['MSE'],'N':bs['N'],'MSE_unseen':bs['MSEunseen'],'Generations':max_N,
               'Pool_size':sample_size,'Horizon':[horizon],'alpha':[alpha]}
    helpo.log_data(details,'inf_crit_comp.csv')
    time.sleep(20)
    
    print('Using Stochastic Complexity')
    score_track = naturally_select(TS,TS2,max_N=max_N,sample_size=sample_size,
                                   horizon=horizon,alpha=alpha) 
    bs = review_scores_by_crit(score_track, LenTS)
    details = {'Method':'SC','MSE':bs['MSE'],'N':bs['N'],'MSE_unseen':bs['MSEunseen'],'Generations':max_N,
               'Pool_size':sample_size,'Horizon':[horizon],'alpha':[alpha]}
    helpo.log_data(details,'inf_crit_comp.csv')
    
#crit1 = [rand.random() for i in range(30)]
#print('crit1: '+str(crit1))
#crit2 = [rand.random() for i in range(30)]
#print('crit2: '+str(crit2))
#print(pareto_front_sorter(crit1, crit2))




























