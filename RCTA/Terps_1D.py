import numpy as np 
import matplotlib.pyplot as plt
import random as rand
import maps_and_systems as mapnsys

def fit_coefs(V,Y,alpha=0):
    # note that alpha used to be 10**(-9)
    return np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(V),V)+np.diag(np.ones(V.shape[1]))*alpha),
                                     np.transpose(V)),Y)

class terp1d:
    def __init__(self,interval):
        self.lower = interval[0]
        self.upper = interval[1]
        self.state = sum(interval)/2
        self.home = sum(interval)/2
        self.velocity = 0
        self.lag = rand.randint(0,6)
    def mask_ts(self,TS):
        ts = np.concatenate([100*np.ones(self.lag),TS])
        mask = ts > self.lower 
        mask2 = ts < self.upper
        mask = mask & mask2
        return mask
    def process(self,TS,fric=0.1,homing=0.5,prey=0.01,transient=200,dt=0.01):
        mask = self.mask_ts(TS); states = []
        for t in range(len(TS)):
            f = homing*(self.home-self.state)*abs(self.home-self.state) - fric*self.velocity 
            if mask[t]:
                f += prey*(TS[t]-self.state)/abs(TS[t]-self.state)**1.5
            self.velocity += dt*f
            self.state += self.velocity
            states.append(self.state*1)
        return states[transient:]
    
class Terp_land1d:
    def __init__(self,n,interval,Ff=0.1,Fh=0.5,Fp=0.01):
        splits = np.random.uniform(low=interval[0],high=interval[1],size=(n+1))
        np.sort(splits)
        self.intevals = np.array([[splits[i],splits[i+1]] for i in range(n)])
        self.homes = np.array([(splits[i]+splits[i+1])/2 for i in range(n)])
        self.state = np.array([(splits[i]+splits[i+1])/2 for i in range(n)])
        self.velocities = np.zeros(n)
        self.friction = np.ones(n)*Ff
        self.homing = np.ones(n)*Fh
        self.preyF = np.ones(n)*Fp
    def mask_ts(self,TS):
        mask = TS > self.intervals[0,:]
        mask2 = TS < self.intervals[1,:]
        mask = mask & mask2
        return mask
    def process(self,TS,fric=0.1,homing=0.5,prey=0.01,transient=200,dt=0.01):
        mask = self.mask_ts(TS); states = []
        for t in range(len(TS)):
            f = self.homing*(self.homes-self.state)*abs(self.homes-self.state) - self.friction*self.velocity 
            f += mask[t]*self.preyF*(TS[t]-self.state)/abs(TS[t]-self.state)**1.5
            self.velocity += dt*f
            self.state += self.velocity
            states.append(self.state*1)
        return states[transient:]
    
TS = np.sin(np.linspace(-50,50,1001))
TS = mapnsys.Lorenz(1200,dt=0.02)
TS = mapnsys.Thomas(1200)
terp = terp1d([0.2,0.4])
terp.mask_ts(TS)
ts = terp.process(TS)


terps = [np.ones(len(TS[200:]))]
for i in range(200):
    #inter = [2*rand.random()-1,2*rand.random()-1]
    home = 5.6*rand.random()-2.8
    ran = abs(rand.gauss(0,0.3))
    inter = [home-ran,home+ran]
    terp = terp1d([min(inter),max(inter)])
    terp.mask_ts(TS)
    dt = rand.random()/30; fric = rand.random()/5; homing=rand.random()+0.1
    prey = rand.random()/30
    ts = terp.process(TS,fric=fric,homing=homing,prey=prey,dt=dt)
    terps.append(ts*1)
    plt.plot(ts)

X = np.transpose(np.array(terps))[:-9,:]
print(X.shape)
coefs = fit_coefs(X, TS[209:],alpha=10**(-12))

plt.plot(TS[209:],color='k')  
plt.show()
plt.plot(TS[209:],color='k')
plt.plot(np.matmul(X,coefs))
plt.show()

print(np.mean(np.square(np.matmul(X,coefs)-TS[209:])))
