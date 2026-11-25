import numpy as np 
import matplotlib.pyplot as plt
import random as rand
import maps_and_systems as mapnsys
import time
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt
import time

def fit_coefs(V,Y,alpha=0,transient=100):
    # note that alpha used to be 10**(-9)
    return np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(V[transient:,:]),V[transient:,:])+
                                             np.diag(np.ones(V[transient:,:].shape[1]))*alpha),
                                     np.transpose(V[transient:,:])),Y[transient:])

def fit_coefs_pinv(V,Y,alpha=0,transient=100):
    # note that alpha used to be 10**(-9)
    return np.matmul(np.matmul(np.linalg.pinv(np.matmul(np.transpose(V[transient:,:]),V[transient:,:])+
                                             np.diag(np.ones(V[transient:,:].shape[1]))*alpha),
                                     np.transpose(V[transient:,:])),Y[transient:])

def embedd_TS(TS,lag=1,horizon=1):
    return np.array([[TS[t],TS[t+lag]] for t in range(len(TS)-lag-horizon)])

#def create_adjacency(homes):
#    vor = Voronoi(homes)
#    n = len(homes)
#    A = np.zeros((n, n), dtype=bool)

#    for ridge in vor.ridge_points:
#        A[ridge[0],ridge[1]] = True
#        A[ridge[1],ridge[0]] = True
    
#    return np.where(A, 1, 0)

def create_adjacency(homes,mini=-2.5,maxi=2.5):
    vor = Voronoi(homes)
    n = len(homes)
    A = np.zeros((n, n), dtype=bool)
    for ridge, verts in zip(vor.ridge_points, vor.ridge_vertices):
        #print(ridge)
        p1,p2 = vor.vertices[verts]; valid_border = False
    
        if verts[0] != -1:
            if np.min(p1) > mini and np.max(p1) < maxi:
               valid_border = True
        if verts[1] != -1:
            if np.min(p2) > mini and np.max(p2) < maxi:
                valid_border = True
        #print('p1:' +str(p1))
        #print('p2:' +str(p2))
        if valid_border:
            
            A[ridge[0],ridge[1]] = True
            A[ridge[1],ridge[0]] = True
    return np.where(A, 1, 0)
    
class Terp_land2d:
    def __init__(self,verticies,Ff=7,Fh=60,Fp=1,Fa=1,fit_inp=False):
        n = len(verticies)
        self.homes = np.array([ v for v in verticies])
        self.state = np.array([ v for v in verticies])
        self.velocities = np.array([np.zeros(2) for v in verticies])
        self.friction = Ff
        self.homing = Fh
        self.preyF = Fp
        self.Fa = Fa
        try:
            self.M = create_adjacency(self.homes)
        except:
            self.M = np.diag(np.ones(n))
        #print(self.M)
        #M = np.abs(np.random.normal(0,1,size=(n,n)))
        #M = M - np.diag(M)
        #M = M - np.diag(np.sum(M,axis=1))
        #eigens = [abs(i) for i in np.linalg.eigvals(M)]
        #self.M = (0.9/max(eigens))*M
        self.fit_inp = fit_inp
    def embedd_TS(self,TS,lag=1,horizon=1,plot=False,ydim = 1,noise=0):
        if ydim == 1:
            Y = TS[lag+horizon:]
        else:
            Y = np.array([[TS[t+horizon],TS[t+lag+horizon]] for t in range(len(TS)-lag-horizon)])
        emb =  np.array([[TS[t],TS[t+lag]] for t in range(len(TS)-lag-horizon)])
        emb += np.random.normal(0,noise,size=emb.shape)
        if plot:
            plt.plot(emb[:,0],emb[:,1])
            vor = Voronoi(self.homes)
            for ridge in vor.ridge_vertices:
                if -1 not in ridge:  # skip infinite ridges
                    v0, v1 = vor.vertices[ridge]
                    plt.plot([v0[0], v1[0]], [v0[1], v1[1]], 'k--',linewidth=1)
            for ridge in vor.ridge_vertices:
                if -1 not in ridge:  # skip infinite ridges
                    v0, v1 = vor.vertices[ridge]
                    plt.plot([v0[0], v1[0]], [v0[1], v1[1]], 'k--',linewidth=1)
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
                    plt.plot([v[0], far_point[0]], [v[1], far_point[1]], 'k--',linewidth=1)
            
            for h in self.homes:
                plt.plot([h[0]],[h[1]],'x',color='k')
            plt.xlim([-3,3])
            plt.ylim([-3,3])
            plt.show()
        return emb,Y
    def mask_ts(self,TS):
        distance = np.linalg.norm(TS[:,None,:]-self.homes[None, :, :],axis=2)
        mins = np.argmin(distance, axis=1)
        M = np.zeros_like(distance) 
        M[np.arange(distance.shape[0]), mins] = 1
        return M
    def process(self,TS,fric=7,homing=60,prey=10,dt=0.02,save_time=-300,subtract_homes=True):
        states = []; X = []
        #mask = self.mask_ts(TS); 
        for t in range(len(TS)):
            #f = self.homing*(self.homes-self.state)*abs(self.homes-self.state) - self.friction*self.velocities
            f = self.homing*(self.homes-self.state) - self.friction*self.velocities
            distance = np.linalg.norm(self.homes-TS[t],axis=1)
            mins = np.argmin(distance)
            M = self.M[mins,:]
            #M = np.zeros_like(distance) 
            #M[mins] = 1
            #preyf = mask[t][:,None]*(TS[t]-self.state)
            preyf = M[:,None]*(TS[t]-self.state)
            preyf = self.preyF*preyf/(0.00001+distance[:,None])


            
            #alignf = self.Fa*np.matmul(self.M,self.velocities)
            #f += self.preyF*mask[t][:,None]*(TS[t]-self.state)/abs(TS[t]-self.state)
            #f = 400*np.tanh((f+preyf+alignf)/400)
            #f = 400*np.tanh((f+preyf)/400)
            f = f+preyf
            #f = f+preyf+alignf
            self.state += 0.5*dt*self.velocities
            self.velocities += dt*f
            self.state += 0.5*dt*self.velocities
            states.append(self.state*1)
            if subtract_homes:
                x = np.concatenate((self.velocities,self.state-self.homes))
            else:
                x = np.concatenate((self.state,self.velocities))
            X.append(x.ravel()*1)
            if t == len(TS)+save_time:
                self.last_inp = TS[t]
                save_state = self.state*1
                save_vel = self.velocities*1
        X = np.array(X)
        if self.fit_inp:
            X = np.hstack((X,TS))
        ones_column = np.ones((X.shape[0], 1), dtype=X.dtype)
        X = np.hstack((X, ones_column))
        self.velocities = save_vel
        self.state = save_state
        return states,X
    def fit_coefs(self,X,Y,alpha=10**(-12),transient=100):
        if len(Y.shape) == 1:
            coefs = fit_coefs(X, Y, alpha=alpha, transient=transient)
        else:
            coefs = fit_coefs(X, Y, alpha=alpha, transient=transient)
        return coefs
    def update(self,prey,state,velocities,dt=0.02):
        distance = np.linalg.norm(self.homes-prey,axis=1)
        mins = np.argmin(distance)
        M = self.M[mins,:]
        f = self.homing*(self.homes-state) - self.friction*velocities
        preyf = M[:,None]*(prey-state)
        preyf = self.preyF*preyf/(0.00001+distance[:,None])
        f = f+preyf
        state += 0.5*dt*velocities
        velocities += dt*f
        state += 0.5*dt*velocities
        return state, velocities
    def predict(self,state,velocities,coefs):
        readout = np.concatenate((velocities,state-self.homes))
        if self.fit_inp:
            readout = np.concatenate((readout,self.last_inp[None,:]))
        readout = readout.ravel()
        readout = np.concatenate((readout,np.ones(1)))
        return np.matmul(np.transpose(readout),coefs)
    def auto_recon(self,coefs,TSend,lag=1,dt=0.02,Nsteps=1000):
        preds = []; end_len = len(TSend)
        for t in range(end_len+Nsteps):
            
            readout = np.concatenate((self.velocities,self.state-self.homes))
            if self.fit_inp:
                readout = np.concatenate((readout,self.last_inp[None,:]))
            readout = readout.ravel()
            readout = np.concatenate((readout,np.ones(1)))

            pred = np.matmul(np.transpose(readout),coefs)
            preds.append(pred)
            if t + lag >= end_len:
                TSend.append(pred)
            #else:
            #    print('Pred is '+str(pred)+', truth is '+str(TSend[t+lag]))
            
            prey = np.array([TSend[t],TSend[t+lag]])

            distance = np.linalg.norm(self.homes-prey,axis=1)
            mins = np.argmin(distance)
            M = self.M[mins,:]
            #M = np.zeros_like(distance) 
            #M[mins] = 1
            
            if False:
                plt.plot([prey[0]],[prey[1]],'o',color='r')
                plt.plot([h[0] for h in self.state],[h[1] for h in self.state],'o',color='k')
                plt.plot([self.state[mins,0]],[self.state[mins,1]],'o',color='C0')
                plt.xlim(-2.5,2.5)
                plt.ylim(-2.5,2.5)
                plt.show()

            
            f = self.homing*(self.homes-self.state) - self.friction*self.velocities
            preyf = M[:,None]*(prey-self.state)
            preyf = self.preyF*preyf/(0.00001+distance[:,None])
            #preyf = self.preyF*preyf/np.linalg.norm(preyf)
            #alignf = self.Fa*np.matmul(self.M,self.velocities)
            
            #f += self.preyF*mask[t][:,None]*(TS[t]-self.state)/abs(TS[t]-self.state)
            #f = 400*np.tanh((f+preyf+alignf)/400)
            #f = 400*np.tanh((f+preyf)/400)
            f = f+preyf
            #f = f+preyf+alignf
            self.state += 0.5*dt*self.velocities
            self.velocities += dt*f
            self.state += 0.5*dt*self.velocities
            self.last_inp = prey
        return np.array(preds[end_len:]),np.array(preds)
    def short_recon_MSE(self,coefs,Y,lag=5,length=15):
        
        # drive for transient period 
        # repeatedly:
            #drive for a short time
            #save state
            #autonomoulsy predict ahead for T steps
            #record average error at the end of the prediction 
            #reset state to saved state
        STATE = 1*self.state; VEL = 1*self.velocities; MSEs = []
        for t in range(100):
            prey = np.array([Y[t],Y[t+lag]])
            STATE,VEL = self.update(prey, STATE, VEL)

        for t in range(100,len(Y)-2*lag-length-2,lag):
            state = 1*STATE; vel=1*VEL
            buffer = Y[t:t+lag+1].tolist()
            for tau in range(length+lag):
                #print('buffer: '+str(buffer))
                prey = np.array([buffer[tau],buffer[tau+lag]])
                #print('prey: '+str(prey))
                #print('')
                state,vel = self.update(prey, state, vel)
                pred = self.predict(state, vel, coefs)
                buffer.append(pred)
            #if t == 1000:
            #    plt.plot(buffer,color='k')
            #    plt.plot(Y[t:t+2*lag+length+1],color='r')
            #    plt.title(str(np.mean(MSEs)))
            #    plt.show()
            #print('')
            #print(buffer[lag+1:])
            #print(Y[t+lag+1:t+2*lag+length+1])
            #print('---------------------------')
            MSEs.append(np.mean(np.square(buffer[2*lag+1:]-Y[t+2*lag+1:t+2*lag+length+1])))
            for tau in range(lag):
                prey = np.array([Y[t+tau],Y[t+lag+tau]])
                STATE,VEL = self.update(prey, STATE, VEL)

        return np.mean(MSEs)

    
            
        
    def plot_homes(self,TS,lag=5,horizon=1,title='Terp Land'):
        emb =  np.array([[TS[t],TS[t+lag]] for t in range(len(TS)-lag-horizon)])
        fig,axs=plt.subplots(1,1,figsize=(10,9),dpi=120)
        axs.set_title(title,fontsize=20)
        vor = Voronoi(self.homes)
        #voronoi_plot_2d(vor)
        mins = self.homes.min(axis=0)
        maxs = self.homes.max(axis=0)
        minx = mins[0]; miny = mins[1]; maxx = maxs[0]; maxy = maxs[1]
        center = np.zeros(2)
        for ridge in vor.ridge_vertices:
            if -1 not in ridge:  # skip infinite ridges
                v0, v1 = vor.vertices[ridge]
                axs.plot([v0[0], v1[0]], [v0[1], v1[1]], 'k--')
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
        axs.set_xlim([1.05*minx,1.05*maxx])
        axs.set_ylim([1.05*miny,1.05*maxy])
        for h in self.homes:
            axs.plot([h[0]],[h[1]],'x',color='k')
        axs.plot(emb[:,0],emb[:,1],color='C0',alpha=0.3)
        plt.show()
 
def terp_agent_plot(X,Y,ind,coeffs):
    keys = ['x','y','vx','vy']
    inds = {'x':2*ind,'y':2*ind+1,'vx':2*ind+len(coeffs)//2,'vy':2*ind+1+len(coeffs)//2}
    params = {k:coeffs[inds[k]] for k in inds.keys()}
    individual_preds = {k:params[k]*X[:,inds[k]] for k in inds.keys()}
    combination_pred = sum([individual_preds[k] for k in keys])
    #print(combination_pred)
    plt.plot(Y,color='k')
    for key in inds.keys():
        plt.plot(individual_preds[key],label=key)
    plt.plot(combination_pred,label='total')
    plt.legend()
    plt.show()
    
    individual_preds['combination'] = combination_pred
    individual_preds['system'] = Y
    keys = list(individual_preds.keys())
    fig,axs=plt.subplots(2,3,figsize=(12,8))
    for i in range(6):
        axs[i//3,i%3].plot(individual_preds[keys[i]][:-5],individual_preds[keys[i]][5:],color='k')
        axs[i//3,i%3].set_title(keys[i],fontsize=20)
    plt.tight_layout()
    plt.show()
    
def get_avg_forecasting_horizon(terps,coefs,TS2,lag=5,dt=0.02,Nsteps=200,avgs=20,benchmark_err=0.5):
    cutsize = (len(TS2)-300)//avgs; FHs = []
    TSs = [TS2[i*cutsize:(i+1)*cutsize+300] for i in range(avgs)]
    for TS in TSs:
        TSemb,Y = terps.embedd_TS(TS,lag=lag,horizon=1,plot=False)
        states,terpX=terps.process(TSemb,save_time=-200)
        terpXend = terpX[-300:,:]; terpX = terpX[:-300,:]; Yend = Y[-300-lag:]; Y = Y[:-300]
        recon,recostart = terps.auto_recon(coefs,Yend[:-200].tolist(),Nsteps=200,lag=lag)
        FH = [i for i in range(len(Yend[100+lag:])-1) if abs(recostart[100+i]-Yend[100+lag+i])>benchmark_err]
        if len(FH) > 0:
            FH = min(FH)
        else:
            FH = 200
        #plt.title('FH = '+str(FH))    
        #plt.plot(recostart,color='r')
        #plt.plot(Yend[lag:],color='C0')
        #plt.vlines(100,ymin=-2,ymax=2,color='k',linestyle='dotted')
        #plt.show()
        FHs.append(FH)
    return np.mean(FHs)

def get_average_absolute_forecast_errors(terps,coefs,TS2,lag=5,dt=0.02,Nsteps=200,avgs=20):
    avgs = (len(TS2)-400)//100 - 1
    TSs = [TS2[i*100:(i+1)*100+400] for i in range(avgs)]
    errs = np.zeros(300); counter = 0
    for TS in TSs:
        TSemb,Y = terps.embedd_TS(TS,lag=lag,horizon=1,plot=False)
        states,terpX=terps.process(TSemb,save_time=-300)
        terpXend = terpX[-400:,:]; terpX = terpX[:-400,:]; Yend = Y[-400-lag:]; Y = Y[:-400]
        recon,recostart = terps.auto_recon(coefs,Yend[:-300].tolist(),Nsteps=300,lag=lag)
        
        err = np.abs(np.array(recostart[100:-lag])-np.array(Yend[100+lag:]))
        errs += err
        counter += 1
        
    return errs/counter








































