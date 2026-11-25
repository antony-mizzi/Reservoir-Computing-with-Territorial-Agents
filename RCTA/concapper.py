import numpy as np 
import maps_and_systems as mapnsys
import Terps_2D as terpo 
import random as rand
import matplotlib.pyplot as plt
import helper_funcs as helpo
import time
import concap as conco
import pandas as pd
import csv
import graphing as grapo

def random_concap_data_generator(N_agents,territory_method,system='Lorenz',writefile='concaps_random2.csv'):
    
    if system == 'Lorenz':
        TS = mapnsys.runge_Lorenz(51000); Ff=12; Fh=63; Fp=40; lag = 7
    elif system == 'Rossler':
        TS = mapnsys.runge_Rossler(51000); Ff=16; Fh=80; Fp=80; lag = 5
    elif system == 'Halvorsen':
        TS = mapnsys.runge_Halvorsen(51000); Ff=28; Fh=100; Fp=80; lag = 7
    elif system == 'Mackey_G':
        TS = mapnsys.Mackey_glass(51000); Ff=32; Fh=100; Fp=80; lag = 6
    elif system == 'Lorenz96':
        TS = mapnsys.Lorenz96(51000); Ff=20; Fh=80; Fp=25; lag = 8
        
    if territory_method == 'from_TS':
        noisyTS = TS+np.random.normal(0,0.1,size=len(TS))
        noisyemb = terpo.embedd_TS(noisyTS,lag=lag)
        points = [ noisyemb[i,:] for i in range(noisyemb.shape[0]) ]
        verticies = np.array(rand.sample(points,N_agents))
    elif territory_method == 'random_uniform':
        verticies = np.random.uniform(-2.5,2.5,size=(N_agents,2))
    
    
    terps = terpo.Terp_land2d(verticies,Ff=Ff,Fh=Fh,Fp=Fp)
    terps.state = terps.homes+np.random.normal(0,0.3,size=terps.homes.shape)
    TSemb,Y = terps.embedd_TS(TS,lag=lag,horizon=1,plot=False)
    states,X1=terps.process(TSemb)
    terps.state = terps.homes+np.random.normal(0,0.3,size=terps.homes.shape)
    states,X2=terps.process(TSemb)
    concap = conco.get_concap(X1[1000:,:], X2[1000:,:])
    print('Concap = '+str(round(concap,2))+'/'+str(4*len(terps.homes)))
    details = {'system':system,'method':territory_method,'N_terps':len(terps.homes),'Concap':concap}
    helpo.log_data(details, writefile)
    
    
def concapper(fname='homes_copy.csv',writefile='concaps.csv'):
    
    try:
        rows = []
        with open('logs/'+writefile, 'r', newline='') as file:
            reader = csv.reader(file)
            for row in reader:
                rows.append(row)
        current_index = len(rows)
    except:
        print('First index')
        current_index=0
    
    rows = []
    with open('logs/'+fname, 'r', newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            rows.append(row)
    points = []; row = rows[current_index]
    for i in range(1,len(row)//2):
        points.append([float(row[2*i]),float(row[2*i+1])])
    points = np.array(points); system = row[1]; territory_method = row[0]
    
    print('Run number '+str(current_index)+'/'+str(len(rows))+', system = '+system+', method = '+territory_method)
    if system == 'Lorenz':
        TS = mapnsys.runge_Lorenz(51000); Ff=12; Fh=63; Fp=40; lag = 7
    elif system == 'Rossler':
        TS = mapnsys.runge_Rossler(51000); Ff=16; Fh=80; Fp=80; lag = 5
    elif system == 'Halvorsen':
        TS = mapnsys.runge_Halvorsen(51000); Ff=28; Fh=100; Fp=80; lag = 7
    elif system == 'Mackey_G':
        TS = mapnsys.Mackey_glass(51000); Ff=32; Fh=100; Fp=80; lag = 6
    elif system == 'Lorenz96':
        TS = mapnsys.Lorenz96(51000); Ff=20; Fh=80; Fp=25; lag = 8
        
    terps = terpo.Terp_land2d(points,Ff=Ff,Fh=Fh,Fp=Fp)
    terps.state = terps.homes+np.random.normal(0,0.3,size=terps.homes.shape)
    TSemb,Y = terps.embedd_TS(TS,lag=lag,horizon=1,plot=False)
    states,X1=terps.process(TSemb)
    terps.state = terps.homes+np.random.normal(0,0.3,size=terps.homes.shape)
    states,X2=terps.process(TSemb)
    concap = conco.get_concap(X1[1000:,:], X2[1000:,:])
    print('Concap = '+str(round(concap,2))+'/'+str(4*len(terps.homes)))
    details = {'system':system,'method':territory_method,'N_terps':len(terps.homes),'Concap':concap}
    helpo.log_data(details, writefile)
    
  

def re_calculate_FHs(fname='homes3.csv',writefile='FHs_revised.csv',lenTS=3000,alpha=10**(-8)):
    
    try:
        rows = []
        with open('logs/'+writefile, 'r', newline='') as file:
            reader = csv.reader(file)
            for row in reader:
                rows.append(row)
        current_index = len(rows)
    except:
        print('First index')
        current_index=0
    
    rows = []
    with open('logs/'+fname, 'r', newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            rows.append(row)
    points = []; row = rows[current_index]
    for i in range(1,len(row)//2):
        points.append([float(row[2*i]),float(row[2*i+1])])
    points = np.array(points); system = row[1]; territory_method = row[0]
    
    print('Run number '+str(current_index)+'/'+str(len(rows))+', system = '+system+', method = '+territory_method)
    if system == 'Lorenz':
        TS = mapnsys.runge_Lorenz(lenTS+400); Ff=12; Fh=63; Fp=40; lag = 7; benchmark_err = 0.7
        TS2 = mapnsys.runge_Lorenz(lenTS)
    elif system == 'Rossler':
        TS = mapnsys.runge_Rossler(lenTS+400); Ff=16; Fh=80; Fp=80; lag = 5; benchmark_err = 0.15
        TS2 = mapnsys.runge_Rossler(lenTS)
    elif system == 'Halvorsen':
        TS = mapnsys.runge_Halvorsen(lenTS+400); Ff=28; Fh=100; Fp=80; lag = 7; benchmark_err = 0.7
        TS2 = mapnsys.runge_Halvorsen(lenTS)
    elif system == 'Mackey_G':
        TS = mapnsys.Mackey_glass(lenTS+400); Ff=32; Fh=100; Fp=80; lag = 6; benchmark_err = 0.15
        TS2 = mapnsys.Mackey_glass(lenTS)
    elif system == 'Lorenz96':
        TS = mapnsys.Lorenz96(lenTS+400); Ff=20; Fh=80; Fp=25; lag = 8; benchmark_err = 0.15
        TS2 = mapnsys.Lorenz96(lenTS)
        
    terps = terpo.Terp_land2d(points,Ff=Ff,Fh=Fh,Fp=Fp)
    terps.state = terps.homes+np.random.normal(0,0.3,size=terps.homes.shape)
    TSemb,Y = terps.embedd_TS(TS,lag=lag,horizon=1,plot=False)
    states,X1=terps.process(TSemb)
    terpXend = X1[-300:,:]; terpX = X1[:-300,:]; Yend = Y[-300-lag:]; Y = Y[:-300]
    coefs = terps.fit_coefs(terpX,Y,alpha=alpha)
    recon,recostart = terps.auto_recon(coefs,Yend[:-200].tolist(),Nsteps=200,lag=lag)
    
    terps.state = terps.homes+np.random.normal(0,0.3,size=terps.homes.shape)
    states,X2=terps.process(TSemb)
    
    FH = terpo.get_avg_forecasting_horizon(terps, coefs, TS2, lag=lag, benchmark_err=benchmark_err)
    details = {'system':system,'method':territory_method,'N_terps':len(terps.homes),'FH':FH}
    helpo.log_data(details, writefile)
    helpo.log_recon(recon.tolist(), 'recons_revised.csv', method=territory_method, TS=system,size=len(terps.homes))
    







#for go in range(200):
#    re_calculate_FHs()
#    if go%20 == 0 and go != 0:
#        grapo.recon_composites_by_method(Method='MDL_NSGA2',fname='recons_revised.csv',f2='FHs_revised.csv')
#        time.sleep(10)



#print(1/0)  
for go in range(50):
    for N in range(6,101):
        print('RUNNING RESERVOIR SIZE '+str(N))
        random_concap_data_generator(N, 'random_uniform')
        random_concap_data_generator(N, 'from_TS')
        if N % 10 == 0:
            time.sleep(20)
    grapo.concap_plot(fname='logs/concaps_random2.csv')
    time.sleep(180)
    


#for go in range(1000):
#    concapper()
#    if go % 10 == 0:
#        time.sleep(30)

  


    
    
   
