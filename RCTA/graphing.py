import matplotlib.pyplot as plt
import scienceplots
import numpy as np 
import csv
import Voronoi2 as voro
import maps_and_systems as mapnsys
import matplotlib.gridspec as gridspec
import pandas as pd
from scipy.spatial import Voronoi, voronoi_plot_2d
import Terps_2D as terpo
import random as rand
plt.style.use(['science', 'no-latex'])

def get_error_bound(lst,log=False):
    samp_means = [np.mean(np.random.choice(lst,size=len(lst))) for i in range(len(lst))]
    stds = np.array([np.std(s) for s in samp_means])
    return np.std(lst)
    #return np.std(samp_means)

def recon_composite(method,TS='Lorenz',lag=5,fname='recons.csv'):
    Recons = []
    with open('logs/'+fname, 'r', newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            if row[0] == method and row[1] == TS:
                Recons.append([float(r) for r in row])
    fig,axs=plt.subplots(1,1,figsize=(8,8))
    for rec in Recons:
        cap = len(rec)-lag-1
        axs.plot(rec[:cap],rec[lag:cap+lag],linewidth=1,alpha=0.2)#,color='k')
    axs.set_xlim(-2.5,2.5)
    axs.set_ylim(-2.5,2.5)
    axs.set_title(fname.split('_')[1].split('.')[0],fontsize=20)
    plt.tight_layout()
    plt.show()
    
def home_map(method,TS='Lorenz',fname='homes.csv'):
    X = []; Y = []; rows = []
    with open('logs/'+fname, 'r', newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            rows.append(row)
    for row in rows:
        if row[0] == method and row[1] == TS:
            x = []; y = []
            for i in range(1,len(row)//2):
                x.append(float(row[2*i])); y.append(float(row[2*i+1]))
            X.append(x); Y.append(y)

    fig,axs=plt.subplots(1,1,figsize=(8,8))
    for i in range(len(X)):
        axs.plot(X[i],Y[i],'o')#,color='k')
    axs.set_xlim(-2.5,2.5)
    axs.set_ylim(-2.5,2.5)
    axs.set_title(fname.split('_')[1].split('.')[0],fontsize=20)
    plt.tight_layout()
    plt.show()
    
def home_density(method,fname='homes.csv',qual=241,overlay=True,TS='Lorenz'):
    Areas = np.zeros((qual,qual)); count = 0; rows = []
    with open('logs/'+fname, 'r', newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            rows.append(row)
    for row in rows:       
        if row[0] == method and row[1] == TS:
            points = []
            for i in range(1,len(row)//2):
                points.append([float(row[2*i]),float(row[2*i+1])])
            points = np.array(points)
            xs,yx,A = voro.cell_area_at_coords(points,n=qual)
            Areas += np.sqrt(A); count +=1
    
    Areas = Areas/count
    
    plt.imshow(Areas, extent=(-2.5,2.5,-2.5,2.5), origin='lower', cmap='twilight')
    if overlay:
        if TS == 'Lorenz':
            TS = mapnsys.runge_Lorenz(2006,Sr=0.02)
            plt.plot(TS[:-5],TS[5:],color='C2',alpha=0.5)
    plt.colorbar(label="Expected Voronoi Cell Area")
    plt.tight_layout()
    plt.show()
    
def all_home_densities(qual=241,overlay=True,TS='Lorenz',fname='homes.csv'):
    methods = ['from_TS','random_uniform','MDL','MDL_brute']
    Titles = ['Random Uniform', 'Sampled from Time Series', 'Selection Algorithm with MDL','Brute Force with MDL']
    if TS == 'Lorenz':
        ts = mapnsys.runge_Lorenz(2006,Sr=0.02)
    elif TS == 'Rossler':
        ts = mapnsys.runge_Rossler(2006)
    elif TS == 'Thomas':
        ts = mapnsys.runge_Thomas(2006)
    elif TS == 'Halvorsen':
        ts = mapnsys.runge_Halvorsen(2006)
    Areas = [np.zeros((qual,qual)) for i in range(4)]
    for Counter in range(4):
        count = 0; rows = []
        with open('logs/'+fname, 'r', newline='') as file:
            reader = csv.reader(file)
            for row in reader:
                print(row[:3])
                if row[0] == methods[Counter] and row[1] == TS:
                    print('appending row')
                    rows.append(row)
                else:
                    print('NO')
        for row in rows:      
            points = []
            for i in range(1,len(row)//2):
                points.append([float(row[2*i]),float(row[2*i+1])])
            points = np.array(points)
            xs,yx,A = voro.cell_area_at_coords(points,n=qual)
            Areas[Counter] += np.sqrt(A); count +=1
        Areas[Counter] = Areas[Counter]/count
            
    fig = plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(2, 3, width_ratios=[1, 1, 0.05])
    
    axs = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
    print(axs)
    cax = fig.add_subplot(gs[:, 2])
    mins = [np.min(a) for a in Areas]; maxes = [np.max(a) for a in Areas]
    mini = min(mins); maxi = max(maxes)
    for Counter in range(4):
        heat = axs[Counter].imshow(Areas[Counter],  vmin=mini, vmax=maxi,
                                                extent=(-2.5,2.5,-2.5,2.5), origin='lower', cmap='pink')
        if overlay:
            axs[Counter].plot(ts[:-5],ts[5:],color='C0',alpha=0.3)
        
        axs[Counter].set_xticks([])
        axs[Counter].set_yticks([])
        axs[Counter].set_title(Titles[Counter],fontsize=16)
    cbar = fig.colorbar(heat,cax=cax)
    cbar.ax.tick_params(labelsize=16)
    plt.tight_layout()
    plt.show()
    
def all_home_densities_all_attractors(qual=121,overlay=True,fname='homes2.csv',f2='homes_40x60.csv',sizes=[60,30]):
    TSs = ['Lorenz','Rossler','Mackey_G','Halvorsen']
    lags = {'Lorenz':7,'Lorenz96':8,'Rossler':5,'Mackey_G':6,'Halvorsen':7}
    methods = ['random_uniform','from_TS','MDL_NSGA2']
    Titles = ['Random Uniform', 'Random Invariant', 'Evolution']
    TS_series = [mapnsys.runge_Lorenz(2006), mapnsys.runge_Rossler(2006),
                 mapnsys.Mackey_glass(2006),mapnsys.runge_Halvorsen(2006)]
    fig = plt.figure(figsize=(10, 12),dpi=120)
    gs = gridspec.GridSpec(4, 4, width_ratios=[1, 1, 1, 0.05])
    
    axs = [fig.add_subplot(gs[i, j]) for i in range(4) for j in range(3)]
    print(axs)
    cax = [fig.add_subplot(gs[i, 3]) for i in range(4)]
    print(cax)
    for TS in TSs:
        print(TS)
        TSind = TSs.index(TS)
        Areas = [np.zeros((qual,qual)) for i in range(3)] 
        for Counter in range(3):
            print(methods[Counter])
            count = 0; rows = []
            if Counter == 2:
                filename = f2 
            else:
                filename = fname
            with open('logs/'+filename, 'r', newline='') as file:
                reader = csv.reader(file)
                for row in reader:
                    if row[0] == methods[Counter] and row[1] == TS:
                        if Counter < 2:
                            if len(row) == sizes[Counter]+2:
                                rows.append(row)
                        else:
                            rows.append(row)
            print(methods[Counter]+': '+str(len(rows)))
            for row in rows:      
                points = []
                for i in range(1,len(row)//2):
                    points.append([float(row[2*i]),float(row[2*i+1])])
                points = np.array(points)
                xs,yx,A = voro.cell_area_at_coords(points,n=qual)
                Areas[Counter] += np.sqrt(A); count +=1
            Areas[Counter] = Areas[Counter]/count
            print(TS+', '+methods[Counter]+': sample size is '+str(count))
            
    
        mins = [np.min(a) for a in Areas]; maxes = [np.max(a) for a in Areas]
        mini = min(mins); maxi = max(maxes)
        for Counter in range(3):
            heat = axs[Counter+3*TSind].imshow(Areas[Counter],  vmin=mini, vmax=maxi,
                                                    extent=(-2.5,2.5,-2.5,2.5), origin='lower', cmap='gist_ncar')#'pink')
            if overlay:
                ts = TS_series[TSind]
                
                axs[Counter+3*TSind].plot(ts[:-lags[TS]],ts[lags[TS]:],color='C0',alpha=0.3)
                
            axs[Counter+3*TSind].set_xlim(-2.5,2.5)
            axs[Counter+3*TSind].set_ylim(-2.5,2.5)
            axs[Counter+3*TSind].set_xticks([])
            axs[Counter+3*TSind].set_yticks([])
            if TSind == 0:
                axs[Counter+3*TSind].set_title(Titles[Counter],fontsize=20)
        cbar = fig.colorbar(heat,cax=cax[TSind])
        cbar.ax.tick_params(labelsize=16)
    plt.tight_layout()
    plt.show()
    
def all_boundary_directions(qual=11,overlay=True,TS='Lorenz'):
    fnames = ['homes_uniform','homes_random','homes_MDL','homes_AIC']
    Titles = ['Random Uniform', 'Sampled from Time Series', 'Selection Algorithm with MDL','Selection Algorithm with AIC']
    if TS == 'Lorenz':
        ts = mapnsys.runge_Lorenz(2006,Sr=0.02)
    elif TS == 'Rossler':
        ts = mapnsys.runge_Rossler(2006)
    elif TS == 'Thomas':
        ts = mapnsys.runge_Thomas(2006)
    elif TS == 'Halvorsen':
        ts = mapnsys.runge_Halvorsen(2006)
    
    vecmap = []
    for Counter in range(4):
        count = 0; rows = []; Vectors = np.zeros((qual,qual,2)); bins = np.linspace(-2.5,2.5,qual)
        bin_width = bins[1]-bins[0]; Cellcounts = np.ones((qual,qual))*10**(-12)
        with open('logs/'+fnames[Counter]+'_'+TS+'.csv', 'r', newline='') as file:
            reader = csv.reader(file)
            for row in reader:
                rows.append(row)
        for row in rows:      
            points = []
            for i in range(len(row)//2):
                points.append([float(row[2*i]),float(row[2*i+1])])
            points = np.array(points)
            vor = Voronoi(points)
            for ridge in vor.ridge_vertices:
                identifier = np.zeros((qual,qual))
                if -1 not in ridge:  # skip infinite ridges
                    v0, v1 = vor.vertices[ridge]
                    dv = v1-v0; leng = np.linalg.norm(v1-v0)
                    if np.sum(dv) < 0:
                        V = -dv
                    else:
                        V = dv
                    steps = np.linspace(0,leng,11)
                    landings = [v0+s*dv for s in steps]

                    for lan in landings:
                        if np.max(np.abs(lan)) < 2.5:
                            i = int(np.floor((lan[0]+2.5) / bin_width)); j = int(np.floor((lan[1]+2.5) / bin_width))
                            if identifier[i,j] != 1:
                                identifier[i,j] += 1
                                Cellcounts[i,j] += 1

                    vecs = identifier[...,None] * V/leng
                    Vectors += vecs
            #find end points of all edges
            vecmap.append(Vectors/Cellcounts[...,None])
            #does the boundary pass through the box 
            
    fig,axs = plt.subplots(2,2,figsize=(8, 8))

    for Counter in range(4):

        X, Y = np.meshgrid(bins, bins)
        
        # Suppose your vector field is shape (50,50,2)
        U = vecmap[Counter][..., 0]
        V = vecmap[Counter][..., 1]
        
        axs[Counter//2,Counter%2].set_title(Titles[Counter])
        axs[Counter//2,Counter%2].quiver(X, Y, U, V)
        if overlay:
            axs[Counter//2,Counter%2].plot(ts[:-5],ts[5:],color='C0',alpha=0.3)

    plt.tight_layout()
    plt.show()
    
def all_recon_composites(lag=5,TS='Lorenz',fname='recons.csv'):
    fig,axs=plt.subplots(2,2,figsize=(8,8))
    Methods = ['from_TS','random_uniform','MDL','ideal']
    Titles = ['Random Uniform', 'Sampled from Time Series', 'Evolution','Ideal Reconstructions']
    count = 0
    for method in Methods:
        Recons = []
        with open('logs/'+fname, 'r', newline='') as file:
            reader = csv.reader(file)
            for row in reader:
                if row[0] == method and row[1] == TS:
                    Recons.append([float(r) for r in row[2:]])
        print('NUMBER OF RECONS IS '+str(len(Recons)))
        for rec in Recons:
            cap = len(rec)-lag-1
            axs[count//2,count%2].plot(rec[:cap],rec[lag:cap+lag],linewidth=1,alpha=0.3)#,color='k')
        axs[count//2,count%2].set_xlim(-2.5,2.5)
        axs[count//2,count%2].set_ylim(-2.5,2.5)
        axs[count//2,count%2].set_xticks([])
        axs[count//2,count%2].set_yticks([])
        axs[count//2,count%2].set_title(Titles[count],fontsize=16)
        count += 1
    plt.tight_layout()
    plt.show()
    
def recon_composites_by_method(Method='ideal',fname='recons.csv',f2='gen_stats.csv',size=100,filter_by_size=False):

    fig,axs=plt.subplots(2,2,figsize=(8,8),dpi=120)
    TSs = ['Lorenz','Rossler','Mackey_G','Halvorsen']
    Titles = ['Lorenz','Rössler','Mackey Glass','Halvorsen']
    lags = {'Lorenz':7,'Rossler':5,'Mackey_G':6,'Halvorsen':7}
    steps_per_Ltime = {'Lorenz':45.45,'Rossler':38.61,'Mackey_G':50,'Halvorsen':42.74}
    count = 0
    df = pd.read_csv('logs/'+f2)
    for sys in TSs:
        if filter_by_size:
            FHs = df[(df['system']==sys)&(df['method']==Method)&(df['N_terps']==size)]['FH'].tolist()
        else:
            FHs = df[(df['system']==sys)&(df['method']==Method)]['FH'].tolist()
           
        Recons = []; lag = lags[sys]
        with open('logs/'+fname, 'r', newline='') as file:
            reader = csv.reader(file)
            for row in reader:
                if row[0] == Method and row[1] == sys:
                    if filter_by_size:
                        if row[2] == str(size):
                            Recons.append([float(r) for r in row[3:]])
                    else:
                        Recons.append([float(r) for r in row[3:]])
        print('NUMBER OF RECONS IS '+str(len(Recons)))
        for rec in Recons[-25:]:
            cap = len(rec)-lag-1
            axs[count//2,count%2].plot(rec[:cap],rec[lag:cap+lag],linewidth=1,alpha=0.3)#,color='k')
        axs[count//2,count%2].set_xlim(-2.5,2.5)
        axs[count//2,count%2].set_ylim(-2.5,2.5)
        axs[count//2,count%2].set_xticks([])
        axs[count//2,count%2].set_yticks([])
        #axs[count//2,count%2].set_title(Titles[count]+' ('+str(size)+')',fontsize=18)
        axs[count//2,count%2].set_title(Titles[count],fontsize=18)
        if Method != 'ideal':
            error = get_error_bound(FHs)
            print(error)
            axs[count//2,count%2].text(-2.4,2.1,'$V_{fh}='+str(round( np.mean(FHs)/steps_per_Ltime[sys] ,2))+
                                       '\pm '+str(round( 3*error/steps_per_Ltime[sys] ,2))+'$',fontsize=16)
        count += 1
    plt.tight_layout()
    plt.show()
    
    plt.tight_layout()
    plt.show()
def recon_composites_horizontal(Method='ideal',fname='recons.csv',f2='gen_stats.csv',size=100,filter_by_size=False):

    fig,axs=plt.subplots(1,4,figsize=(16,4),dpi=120)
    TSs = ['Lorenz','Rossler','Mackey_G','Halvorsen']
    Titles = ['Lorenz','Rössler','Mackey Glass','Halvorsen']
    lags = {'Lorenz':7,'Rossler':5,'Mackey_G':6,'Halvorsen':7}
    steps_per_Ltime = {'Lorenz':45.45,'Rossler':38.61,'Mackey_G':50,'Halvorsen':42.74}
    count = 0
    df = pd.read_csv('logs/'+f2)
    for sys in TSs:
        if filter_by_size:
            FHs = df[(df['system']==sys)&(df['method']==Method)&(df['N_terps']==size)]['FH'].tolist()
        else:
            FHs = df[(df['system']==sys)&(df['method']==Method)]['FH'].tolist()
           
        Recons = []; lag = lags[sys]
        with open('logs/'+fname, 'r', newline='') as file:
            reader = csv.reader(file)
            for row in reader:
                if row[0] == Method and row[1] == sys:
                    if filter_by_size:
                        if row[2] == str(size):
                            Recons.append([float(r) for r in row[3:]])
                    else:
                        Recons.append([float(r) for r in row[3:]])
        print('NUMBER OF RECONS IS '+str(len(Recons)))
        for rec in Recons[-25:]:
            cap = len(rec)-lag-1
            axs[count].plot(rec[:cap],rec[lag:cap+lag],linewidth=1,alpha=0.3)#,color='k')
        axs[count].set_xlim(-2.5,2.5)
        axs[count].set_ylim(-2.5,2.5)
        axs[count].set_xticks([])
        axs[count].set_yticks([])
        #axs[count//2,count%2].set_title(Titles[count]+' ('+str(size)+')',fontsize=18)
        axs[count].set_title(Titles[count],fontsize=18)
        if Method != 'ideal':
            error = get_error_bound(FHs)
            print(error)
            axs[count].text(-2.4,2.1,'$V_{fh}='+str(round( np.mean(FHs)/steps_per_Ltime[sys] ,2))+
                                       '\pm '+str(round( 3*error/steps_per_Ltime[sys] ,2))+'$',fontsize=16)
        count += 1
    plt.tight_layout()
    plt.show()
    
def summary_stats(fname='logs/gen_stats.csv',conditions={}):
    df = pd.read_csv(fname)
    for c in conditions.keys():
        df = df[df[c]==conditions[c]]
    sizelist = sorted(list(set(list(df['N_terps']))))
    methods = sorted(list(set(list(df['method']))))
    concomps = {m:[] for m in methods}; marks = ['o','x','s','*']; colors = ['C0','C1','C2','C3']
    NMSEs = {m:[] for m in methods}; FHs = {m:[] for m in methods}
    marks = {methods[i]:marks[i] for i in range(len(methods))}
    colors = {methods[i]:colors[i] for i in range(len(methods))}
    labels = {'AIC': 'Selection Algorithm with AIC', 'MDL': 'Selection Algorithm with MDL', 
              'from_TS': 'Sampled from Time Series', 'random_uniform': 'Random Uniform',
              'MDL_brute':'Brute Force with MDL','MDL_breathing':'Breathing MDL',
              'AIC_breathing':'Breathing AIC'}
    for size in sizelist:
        for m in methods:
            concomps[m].append(df[(df['N_terps']==size)&(df['method']==m)]['Concap'].tolist())
            FHs[m].append(np.log(df[(df['N_terps']==size)&(df['method']==m)]['FH'].tolist()))
            NMSEs[m].append(np.log(df[(df['N_terps']==size)&(df['method']==m)]['NMSE_unseen'].tolist()))

    count = 0; dics = [concomps,FHs,NMSEs]
    titles = ['Consistency Capacity vs Reservoir Size','Forecasting Horizon vs Reservoir Size',
              'Log NMSE vs Reservoir Size']
    for dic in dics:
        fig,axs = plt.subplots(1,1,figsize=(8,6))
        for m in methods:
            not_labled = True
            for i in range(len(sizelist)):
                pointset = dic[m][i]; N = sizelist[i]
                if len(pointset) >= 1:
                    if not_labled:
                        axs.plot([N for j in range(len(pointset))],pointset,marks[m],color=colors[m],label=labels[m],alpha=0.2)
                        not_labled = False
                    else:
                        axs.plot([N for j in range(len(pointset))],pointset,marks[m],color=colors[m],alpha=0.2)
        if count == 0:
            axs.plot(sizelist,[4*s+1 for s in sizelist],'k',linestyle='dashed')
        axs.legend(fontsize=15)
        axs.set_xlabel('Number of Agents',fontsize=20)
        #axs.set_ylabel('$N_c$',fontsize=20)
        axs.tick_params(axis='both', labelsize=14)
        axs.set_title(titles[count],fontsize=20)
        plt.tight_layout()
        plt.show()
        count += 1
    
def concap_plot(fname='logs/concaps_random.csv',conditions={}):
    df = pd.read_csv(fname)
    for c in conditions.keys():
        df = df[df[c]==conditions[c]]
    sizelist = sorted(list(set(list(df['N_terps']))))
    methods = sorted(list(set(list(df['method']))))
    concomps = {m:[] for m in methods}; marks = ['o','x','s','*']; colors = ['saddlebrown','magenta','C3','C0']

    marks = {methods[i]:marks[i] for i in range(len(methods))}
    colors = {methods[i]:colors[i] for i in range(len(methods))}
    labels = {'AIC': 'Selection Algorithm with AIC', 'MDL': 'Selection Algorithm with MDL', 
              'from_TS': 'Territories Sampled from Time Series', 
              'random_uniform': 'Territories Sampled Uniformly',
              'MDL_brute':'Brute Force with MDL','MDL_breathing':'Breathing MDL',
              'AIC_breathing':'Breathing AIC'}
    for size in sizelist:
        for m in methods:
            concomps[m].append(df[(df['N_terps']==size)&(df['method']==m)]['Concap'].tolist())
            

    count = 0
    titles = ['Consistency Capacity vs Reservoir Size','Forecasting Horizon vs Reservoir Size',
              'Log NMSE vs Reservoir Size']
    
    fig,axs = plt.subplots(1,1,figsize=(10,6),dpi=120)
    for m in methods:
        not_labled = True
        for i in range(len(sizelist)):
            pointset = concomps[m][i]; N = sizelist[i]
            if len(pointset) >= 1:
                if not_labled:
                    #axs.plot([N for j in range(len(pointset))],pointset,marks[m],color=colors[m],label=labels[m],alpha=0.1)
                    not_labled = False
                #else:
                    #axs.plot([N for j in range(len(pointset))],pointset,marks[m],color=colors[m],alpha=0.1)
        
        
        errs = np.array([get_error_bound(c) for c in concomps[m]])
        print(3*errs)
        means = np.array([np.mean(c) for c in concomps[m]])
        print(means)
        axs.plot(sizelist,means,color=colors[m],label=labels[m])#,linestyle='dotted')
        axs.fill_between(sizelist,means-3*errs,means+3*errs,color=colors[m],alpha=0.3)
    #print('Number of Samples is '+str(len(pointset)))
    if count == 0:
        axs.plot(sizelist,[4*s for s in sizelist],'k',linestyle='dashed')
    
    axs.legend(fontsize=15)
    axs.set_xlabel('Number of Agents',fontsize=20)
    axs.set_ylabel('Consistent Components',fontsize=20)
    axs.tick_params(axis='both', labelsize=14)
    axs.set_title(titles[count],fontsize=20)
    plt.tight_layout()
    plt.show()
    count += 1
    
def FH_by_number_of_agents(fname='logs/gen_stats.csv',conditions={}):
    df = pd.read_csv(fname)
    for c in conditions.keys():
        df = df[df[c]==conditions[c]]
    fig,axs = plt.subplots(2,2,figsize=(14,8),dpi=120)
    TSs = ['Lorenz','Rossler','Mackey_G','Halvorsen']
    Titles = ['Lorenz','Rössler','Mackey Glass','Halvorsen']
    steps_per_Ltime = [45.45,39.22,50,42.74]
    for ind in range(4):
        sizelist = sorted(list(set(list(df['N_terps']))))
        sizelist = [s for s in sizelist if s%5 == 0]
        #methods = sorted(list(set(list(df['method']))))
        methods = ['from_TS','random_uniform']
        FHs = {m:[] for m in methods}; marks = ['o','x','s','*']; colors = ['saddlebrown','magenta','C3','C0']
    
        marks = {methods[i]:marks[i] for i in range(len(methods))}
        colors = {methods[i]:colors[i] for i in range(len(methods))}
        labels = {'AIC': 'Selection Algorithm with AIC', 'MDL': 'Selection Algorithm with MDL', 
                  'from_TS': 'Terrirtories Sampled from Time Series', 
                  'random_uniform': 'Territories Sampled Uniformly',
                  'MDL_brute':'Brute Force with MDL','MDL_breathing':'Breathing MDL',
                  'AIC_breathing':'Breathing AIC','MDL_NSGA2':'NSGA-ii'}
        for size in sizelist:
            for m in methods:
                arr = df[(df['N_terps']==size)&(df['method']==m)&(df['system']==TSs[ind])]['FH']/steps_per_Ltime[ind]
                FHs[m].append(arr.tolist())
                
    
        count = 0
        
        
        for m in methods:
            
            not_labled = False
            for i in range(len(sizelist)):
                pointset = FHs[m][i]; N = sizelist[i]
            #    if len(pointset) >= 1:
            #        if not_labled:
            #            axs[ind//2,ind%2].plot([N for j in range(len(pointset))],pointset,marks[m],color=colors[m],label=labels[m],alpha=0.1)
            #            not_labled = False
            #        else:
            #            axs[ind//2,ind%2].plot([N for j in range(len(pointset))],pointset,marks[m],color=colors[m],alpha=0.1)
            
            
            errs = np.array([get_error_bound(c) for c in FHs[m]])
            means = np.array([np.mean(c) for c in FHs[m]])
            axs[ind//2,ind%2].plot(sizelist,means,color=colors[m])#,label=labels[m])
            axs[ind//2,ind%2].plot(sizelist,means,marks[m],color=colors[m],label=labels[m])
            axs[ind//2,ind%2].fill_between(sizelist,means-3*errs,means+3*errs,color=colors[m],alpha=0.3)
            axs[ind//2,ind%2].tick_params(axis='both', labelsize=16)
            axs[ind//2,ind%2].set_title(Titles[ind],fontsize=22)
        print('Number of Samples is '+str(len(pointset)))
        #if count == 0:
        #    axs.plot(sizelist,[4*s+1 for s in sizelist],'k',linestyle='dashed')
        
    axs[0,0].legend(fontsize=19)
    axs[1,0].set_xlabel('Number of Agents',fontsize=22)
    axs[1,1].set_xlabel('Number of Agents',fontsize=22)
    axs[0,0].set_xticks([])
    axs[0,1].set_xticks([])
    axs[0,0].set_ylabel('Forecasting Horizon',fontsize=22)
    axs[1,0].set_ylabel('Forecasting Horizon',fontsize=22)
    #axs[0,1].set_yticks([])
    #axs[1,1].set_yticks([])
    
    plt.tight_layout()
    plt.show()
    count += 1
    
def breathing_plot(f1='logs/Island_stats.csv',f2='logs/gen_stats.csv',conditions={}):
    df1 = pd.read_csv(f1); df2 = pd.read_csv(f2)
    for c in conditions.keys():
        df1 = df1[df1[c]==conditions[c]];  df2 = df2[df2[c]==conditions[c]]
    uniform_FH = df2[(df2['method']=='random_uniform')]['FH'].tolist()
    fromts_FH = df2[(df2['method']=='from_TS')]['FH'].tolist()
    uniform_MSE = df2[(df2['method']=='random_uniform')]['NMSE_unseen'].tolist()
    fromts_MSE = df2[(df2['method']=='from_TS')]['NMSE_unseen'].tolist()
    its = sorted(list(set(list(df1['itteration']))))
    methods = sorted(list(set(list(df1['method']))))
    horizons = {m:[] for m in methods}; MSEs = {m:[] for m in methods}
    sizes = {m:[] for m in methods}
    for it in its:
        for m in methods:
            horizons[m].append(df1[(df1['itteration']==it)&(df1['method']==m)]['FH'].tolist())
            MSEs[m].append(df1[(df1['itteration']==it)&(df1['method']==m)]['NMSE_unseen'].tolist())
            sizes[m].append(df1[(df1['itteration']==it)&(df1['method']==m)]['N_terps'].tolist())
    
    fig,axs=plt.subplots(3,1,figsize=(8,8))
    for m in methods:
        axs[0].plot(its,[np.mean(h) for h in horizons[m]],label=m)
        axs[0].set_title('Forecasting Horizon',fontsize=18)
        axs[0].set_xticks([])
        axs[1].plot(its,[np.mean(np.log(h)) for h in MSEs[m]],label=m)
        axs[1].set_title('Log Unseen NMSE',fontsize=18)
        axs[1].set_xticks([])
        axs[2].plot(its,[np.mean(h) for h in sizes[m]],label=m)
        axs[2].set_title('Network Size',fontsize=18)
        axs[2].set_xlabel('Number of Itterations',fontsize=18)
    axs[0].plot(its,[np.mean(uniform_FH) for it in its],label='Random Uniform',linestyle='dashed')
    axs[0].plot(its,[np.mean(fromts_FH) for it in its],label='Sampled from TS',linestyle='dashed')
    axs[1].plot(its,[np.mean(np.log(uniform_MSE)) for it in its],linestyle='dashed')
    axs[1].plot(its,[np.mean(np.log(fromts_MSE)) for it in its],linestyle='dashed')
    axs[0].legend(fontsize=12)
    #axs.tick_params(axis='both', labelsize=14)
    plt.tight_layout()
    plt.show()
    
    
def voronoi_example(lag=7,Ff=20,Fh=63,Fp=80,Fa=0):
    TS = mapnsys.runge_Lorenz(4000,Sr=0.02)
    noisyTS = TS+np.random.normal(0,0.3,size=len(TS))
    noisyemb = terpo.embedd_TS(noisyTS,lag=lag)
    points = [ noisyemb[i,:] for i in range(noisyemb.shape[0]) ]
    homes = np.array(rand.sample(points,20))
    terps = terpo.Terp_land2d(homes,Fp=Fp,Fh=Fh,Ff=Ff,Fa=Fa,fit_inp = False)
    TSemb,Y = terps.embedd_TS(TS,lag=lag,horizon=1,plot=True)
    states,terpX=terps.process(TSemb,save_time=-1,subtract_homes=False)
    #axs.set_title(title,fontsize=20)
    vor = Voronoi(homes)
    #voronoi_plot_2d(vor)
    mins = homes.min(axis=0)
    maxs = homes.max(axis=0)
    minx = mins[0]; miny = mins[1]; maxx = maxs[0]; maxy = maxs[1]
    fig,axs=plt.subplots(1,1,figsize=(10,9), dpi=120)
    axs.plot(TSemb[:,0],TSemb[:,1],linestyle='dotted',alpha=0.2)
    for ridge in vor.ridge_vertices:
        if -1 not in ridge:  # skip infinite ridges
            v0, v1 = vor.vertices[ridge]
            axs.plot([v0[0], v1[0]], [v0[1], v1[1]], 'k--')
        center = vor.points.mean(axis=0)
    for (p, q), rv in zip(vor.ridge_points, vor.ridge_vertices):
        if -1 in rv:
            i = rv[0] if rv[0] != -1 else rv[1]
            v = vor.vertices[i]

            # Direction perpendicular to the line between the two points
            tangent = vor.points[q] - vor.points[p]
            normal = np.array([-tangent[1], tangent[0]])
            normal /= np.linalg.norm(normal)

            # Choose direction pointing outward
            direction = np.sign(np.dot(v - center, normal)) * normal

            # Extend the infinite edge
            far_point = v + direction * 100
            axs.plot([v[0], far_point[0]], [v[1], far_point[1]], 'k--')
    axs.plot(TSemb[-15:,0],TSemb[-15:,1],color='C0',alpha=0.4,linewidth=3)
    axs.plot(TSemb[-1:,0],TSemb[-1:,1],'o',color='C0',label='Prey',markersize=10)
    Nterps = homes.shape[0]
    M = terps.M
    row0 = M[0,:]
    for i in range(Nterps):
        axs.plot(terpX[-15:,2*i],terpX[-15:,2*i+1],color='C3',alpha=0.4,linewidth=3)
        if i == 0:
            axs.plot(terpX[-1:,2*i],terpX[-1:,2*i+1],'*',color='C3',markersize=10,label='Predator')
        elif row0[i] == 1:
            axs.plot(terpX[-1:,2*i],terpX[-1:,2*i+1],'*',color='C3',markersize=10)
        else:
            axs.plot(terpX[-1:,2*i],terpX[-1:,2*i+1],'*',color='C3',markersize=10)
    axs.set_xticks([])
    axs.set_yticks([])
    #axs.legend(fontsize=18)
    axs.set_xlim([1.1*minx,1.1*maxx])
    axs.set_ylim([1.1*miny,1.1*maxy])
    
    plt.show()

def component_embedding_and_territory_vis(X,Y,homes,lag=5):
    fig = plt.figure(figsize=(15,8))
    gs = gridspec.GridSpec(2, 3, width_ratios=[2, 1, 1])
    ax_left = fig.add_subplot(gs[:, 0])
    ax_left.plot(Y[:-lag],Y[lag:],color='k',alpha=0.1)
    ax_left.set_xticks([])
    ax_left.set_yticks([])
    axes = []
    for r in range(0, 2):        
        for c in range(1,3):       
            ax = fig.add_subplot(gs[r, c])
            ax.set_xticks([])
            ax.set_yticks([])
            axes.append(ax)
    vor = Voronoi(homes)
    #voronoi_plot_2d(vor)
    mins = homes.min(axis=0)
    maxs = homes.max(axis=0)
    minx = mins[0]; miny = mins[1]; maxx = maxs[0]; maxy = maxs[1]
    for ridge in vor.ridge_vertices:
        if -1 not in ridge:  # skip infinite ridges
            v0, v1 = vor.vertices[ridge]
            ax_left.plot([v0[0], v1[0]], [v0[1], v1[1]], 'k--')
        center = vor.points.mean(axis=0)
    for (p, q), rv in zip(vor.ridge_points, vor.ridge_vertices):
        if -1 in rv:
            i = rv[0] if rv[0] != -1 else rv[1]
            v = vor.vertices[i]

            # Direction perpendicular to the line between the two points
            tangent = vor.points[q] - vor.points[p]
            normal = np.array([-tangent[1], tangent[0]])
            normal /= np.linalg.norm(normal)

            # Choose direction pointing outward
            direction = np.sign(np.dot(v - center, normal)) * normal

            # Extend the infinite edge
            far_point = v + direction * 100
            ax_left.plot([v[0], far_point[0]], [v[1], far_point[1]], 'k--')
    ax_left.plot([homes[0][0]],[homes[0][1]],'X',color='C0',markersize=13,label='$h^{(0)}$')
    ax_left.plot([homes[1][0]],[homes[1][1]],'X',color='C1',markersize=13,label='$h^{(1)}$')
    ax_left.legend(fontsize=20)
    ax_left.set_xlim([-3,3])
    ax_left.set_ylim([-3,3])
    ax_left.set_title('Home Locations and Driving Signal',fontsize=25)
    ax_left.set_ylabel('$u(t+\\tau)$',fontsize=20)
    ax_left.set_xlabel('$u(t)$',fontsize=20)
    N = homes.shape[0]
    for i in range(2,N):
        ax_left.plot([homes[i][0]],[homes[i][1]],'X',color='k',markersize=9)
    axes[0].set_title('$x^{(0)}_1$',fontsize=25)
    axes[0].set_ylabel('$x^{(0)}_1(t+\\tau)$',fontsize=20)
    axes[0].plot(X[:-lag,0],X[lag:,0],color='C0',alpha=0.2)
    axes[1].set_title('$x^{(1)}_1$',fontsize=25)
    axes[1].plot(X[:-lag,1],X[lag:,1],color='C1',alpha=0.2)
    axes[2].set_title('$\dot{x}^{(0)}_1$',fontsize=25)
    axes[2].set_ylabel('$\dot{x}^{(0)}_1(t+\\tau)$',fontsize=20)
    axes[2].set_xlabel('$\dot{x}^{(0)}_1(t)$',fontsize=20)
    axes[2].plot(X[:-lag,2*N],X[lag:,2*N],color='C0',alpha=0.2)
    axes[3].set_title('$\dot{x}^{(1)}_1$',fontsize=25)
    axes[3].set_xlabel('$\dot{x}^{(1)}_1(t)$',fontsize=20)
    axes[3].plot(X[:-lag,2*N+1],X[lag:,2*N+1],color='C1',alpha=0.2)
    plt.tight_layout()
    plt.show()
    
    

def param_scan(fname='logs/param_scan.csv', conditions={}):
    
    df = pd.read_csv(fname)
    for c in conditions.keys():
        df = df[df[c]==conditions[c]]
    
    Fflist = sorted(list(set(list(df['Ff']))))
    Fhlist = sorted(list(set(list(df['Fh']))))
    Fplist = sorted(list(set(list(df['Fp']))))
    print('Fps: '+str(Fplist))
    print('Ffs: '+str(Fflist))
    
    unseens = []; horizons = []
    fig,axs=plt.subplots(1,2,figsize=(10,5))
    for f in Fflist:
        unseens.append(np.log10(df[(df['Ff']==f)]['NMSE_unseen'].tolist()))
        horizons.append(-np.sqrt(df[(df['Ff']==f)]['FH'].tolist()))
    for f in range(len(Fflist)):
        axs[0].plot([np.log10(Fflist[f]) for j in unseens[f]],unseens[f],'o',color='k',alpha=0.1)
        
        axs[1].plot([np.log10(Fflist[f]) for j in unseens[f]],horizons[f],'o',color='k',alpha=0.1)
    axs[0].plot(np.log10(Fflist),[np.mean(unseens[f]) for f in range(len(Fflist))],color='C0')
    axs[1].plot(np.log10(Fflist),[np.mean(horizons[f]) for f in range(len(Fflist))],color='C0')
    axs[0].set_xlabel('$\\log_{10}(Ff)$',fontsize=15)
    axs[1].set_xlabel('$\\log_{10}(Ff)$',fontsize=15)
    axs[0].set_ylabel('$\\log_{10}(NMSE)$',fontsize=15)
    axs[1].set_ylabel('$-\\sqrt{FH}$',fontsize=15)
    plt.tight_layout()
    plt.show()
    
    unseens = []; horizons = []
    fig,axs=plt.subplots(1,2,figsize=(10,5))
    for f in Fplist:
        unseens.append(np.log10(df[(df['Fp']==f)]['NMSE_unseen'].tolist()))
        horizons.append(-np.sqrt(df[(df['Fp']==f)]['FH'].tolist()))
    for f in range(len(Fplist)):
        axs[0].plot([np.log10(Fplist[f]) for j in unseens[f]],unseens[f],'o',color='k',alpha=0.1)
        axs[1].plot([np.log10(Fplist[f]) for j in unseens[f]],horizons[f],'o',color='k',alpha=0.1)
    axs[0].plot(np.log10(Fplist),[np.mean(unseens[f]) for f in range(len(Fplist))],color='C0')
    axs[1].plot(np.log10(Fplist),[np.mean(horizons[f]) for f in range(len(Fplist))],color='C0')
    axs[0].set_xlabel('$\\log_{10}(Fp)$',fontsize=15)
    axs[1].set_xlabel('$\\log_{10}(Fp)$',fontsize=15)
    axs[0].set_ylabel('$\\log_{10}(NMSE)$',fontsize=15)
    axs[1].set_ylabel('$-\\sqrt{FH}$',fontsize=15)
    plt.tight_layout()
    plt.show()
    
    unseens = []; horizons = []
    fig,axs=plt.subplots(1,2,figsize=(10,5))
    for f in Fhlist:
        unseens.append(np.log10(df[(df['Fh']==f)]['NMSE_unseen'].tolist()))
        horizons.append(-np.sqrt(df[(df['Fh']==f)]['FH'].tolist()))
    for f in range(len(Fhlist)):
        axs[0].plot([np.log10(Fhlist[f]) for j in unseens[f]],unseens[f],'o',color='k',alpha=0.1)
        axs[1].plot([np.log10(Fhlist[f]) for j in unseens[f]],horizons[f],'o',color='k',alpha=0.1)
    axs[0].plot(np.log10(Fhlist),[np.mean(unseens[f]) for f in range(len(Fhlist))],color='C0')
    axs[1].plot(np.log10(Fhlist),[np.mean(horizons[f]) for f in range(len(Fhlist))],color='C0')
    axs[0].set_xlabel('$\\log_{10}(Fh)$',fontsize=15)
    axs[1].set_xlabel('$\\log_{10}(Fh)$',fontsize=15)
    axs[0].set_ylabel('$\\log_{10}(NMSE)$',fontsize=15)
    axs[1].set_ylabel('$-\\sqrt{FH}$',fontsize=15)
    plt.tight_layout()
    plt.show()
    
    
    for Fh in Fhlist:
        df = pd.read_csv(fname)
        df = df[df['Fh']==Fh]
        df['FH'] = df['FH']+1
        for c in conditions.keys():
            df = df[df[c]==conditions[c]]
        Fflist = sorted(list(set(list(df['Ff']))))
        Fplist = sorted(list(set(list(df['Fp']))))
        Fps = []; Ffs = []; Horizons = []; Unseens = []#; Concaps = []
        for h in Fflist:
            for p in Fplist:
                Fps.append(p); Ffs.append(h)
                Horizons.append(-np.mean(np.sqrt(df[(df['Ff']==h)&(df['Fp']==p)]['FH'].tolist())))
                Unseens.append(np.mean(np.log10(df[(df['Ff']==h)&(df['Fp']==p)]['NMSE_unseen'].tolist())))
                #Concaps.append(np.mean(df[(df['Fh']==h)&(df['Fp']==p)]['Concap'].tolist()))
        fig,axs=plt.subplots(1,2,figsize=(11,5))
        horizon_heat=axs[0].scatter(np.log10(Fps),np.log10(Ffs),c=Horizons,cmap='gist_rainbow',marker='s',s=420)#,
                                   # vmin=-5,vmax=-2.5)
        NMSE_heat=axs[1].scatter(np.log10(Fps),np.log10(Ffs),c=Unseens,cmap='gist_rainbow',marker='s',s=420)#,
                                # vmin=-1.6,vmax=-0.5)
        fig.colorbar(horizon_heat,ax=axs[0])
        fig.colorbar(NMSE_heat,ax=axs[1])
        axs[0].set_ylabel('$\\log_{10}(F_f)$',fontsize=20)
        axs[0].set_xlabel('$\\log_{10}(F_p)$',fontsize=20)
        axs[1].set_xlabel('$\\log_{10}(F_p)$',fontsize=20)
        axs[1].set_yticks([])
        axs[0].set_title('- Root Forecasting Horizon ')
        axs[1].set_title('NMSE Unseen (Fh = '+str(Fh)+')')
        plt.tight_layout()
        plt.show()
 

def NSGAplot(fname='logs/NSGA2log.csv',conditions={},letter='e'):
    Titles = ['Lorenz','Rössler','Mackey Glass','Halvorsen']
    df = pd.read_csv(fname)
    for c in conditions.keys():
        df = df[df[c]==conditions[c]]
    horizons = [int(c[1:]) for c in df.columns if c[0] == 'm']
    print(horizons)
    for h in horizons:
        df['e'+str(h)]=-np.log(df['e'+str(h)])
    fig,axs = plt.subplots(2,2,figsize=(12,6),dpi=120)
    count=0
    for system in ['Lorenz','Rossler','Mackey_G','Halvorsen']:
        Y = []; Y2 = []
        for h in horizons:
            Y.append(df[(df['system']==system)][letter+str(h)].tolist())
            Y2.append(df[(df['system']==system)]['m'+str(h)].tolist())
        
        errs = np.array([get_error_bound(c) for c in Y])
        means = np.array([np.mean(c) for c in Y])
        axs[count//2,count%2].plot(horizons,means,color='C0',label='effective $\log(\epsilon^2$)')
        axs[count//2,count%2].plot(horizons,means,'x',color='C0')
        axs[count//2,count%2].fill_between(horizons,means-3*errs,means+3*errs,color='C0',alpha=0.3)
        
        errs = np.array([get_error_bound(c) for c in Y2])
        means = np.array([np.mean(c) for c in Y2])
        axs[count//2,count%2].plot(horizons,means,color='C1',label='log 1-step MSE')
        axs[count//2,count%2].fill_between(horizons,means-3*errs,means+3*errs,color='C1',alpha=0.3)
        
        #axs[count//2,count%2].plot(horizons,[np.mean(y) for y in Y],label='effective $\epsilon^2$')
        #axs[count//2,count%2].plot(horizons,[np.mean(y) for y in Y2],label='log 1-step MSE')
        axs[count//2,count%2].tick_params(axis='both', labelsize=14)
        axs[count//2,count%2].set_title(Titles[count],fontsize=18)
        count += 1
    axs[0,0].legend(fontsize=15)
    axs[1,0].set_xlabel('H',fontsize=16)
    axs[1,1].set_xlabel('H',fontsize=16)
    axs[0,0].set_xticks([])
    axs[0,1].set_xticks([])
    plt.tight_layout()
    plt.show()
        
        
#for go in range(20):
#    voronoi_example()    
#NSGAplot(letter='e',conditions={'generations':50})
#FH_by_number_of_agents()
#recon_composites_by_method(Method='random_uniform',fname='recons.csv',f2='gen_stats.csv',
#                            filter_by_size=True,size=60)
#recon_composites_horizontal(Method='random_uniform',fname='recons.csv',f2='gen_stats.csv',
#                            filter_by_size=True,size=60)
#recon_composites_by_method(Method='from_TS',fname='recons.csv',f2='gen_stats.csv',
#                            filter_by_size=True,size=30)
#recon_composites_horizontal(Method='ideal',fname='recons_ideal.csv',f2='gen_stats_nsga.csv')
#recon_composites_horizontal(Method='MDL_NSGA2',fname='recons_nsga.csv',f2='gen_stats_nsga.csv')
#concap_plot(fname='logs/concaps_random.csv')
#FH_by_number_of_agents()
