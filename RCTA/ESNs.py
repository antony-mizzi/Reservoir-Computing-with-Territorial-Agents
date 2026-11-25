import numpy as np 
import time
import networkx as nx
import random as rand
import matplotlib.pyplot as plt

class ESN:
    def __init__(self,n,specrad=1,rewire=0.001,winstrength=0.3,sparsity=1):
        self.state = np.zeros(n)
        net = nx.watts_strogatz_graph(n,6,rewire)#nx.barabasi_albert_graph(n,4)
        M = nx.to_numpy_array(net)
        for i in range(n-1):
            for j in range(i,n):
                if M[i,j] == 1:
                    new_val = rand.gauss(0,1)
                    M[i,j] += new_val-1
                    M[j,i] += new_val-1
        eigens = [abs(i) for i in np.linalg.eigvals(M)]
        self.M = (specrad/max(eigens))*M
        sparse = rand.sample(list(range(n)),int(n*sparsity))
        self.Win = np.array([rand.gauss(0,winstrength) if i in sparse else 0 for i in range(n)])
    def act(self,u,leak):
        self.state = leak*np.tanh(u*self.Win+np.matmul(self.M,self.state))  + (1-leak)*self.state
    def add_noise(self,sigma):
        self.state += np.random.normal(loc=0,scale=sigma,size=len(self.state))
    def euler_increment(self,leak,u,h=4):
        self.state = (1-1/h)*self.state + (1/h)*(leak*np.tanh(u*self.Win+np.matmul(self.M,self.state)) +
                                           (1-leak)*self.state) 
    def increment(self,u,leak=1):
        self.state = leak*np.tanh(u*self.Win+np.matmul(self.M,self.state)) + (1-leak)*self.state
    def predict(self,coeffs,subset=['na']):
        if subset[0] == 'na':
            pred = np.matmul(self.state,np.transpose(coeffs))
        else:
            V = np.transpose(np.array([self.state[b] for b in subset]))
            pred = np.matmul(V,np.transpose(coeffs))
        return pred
    def autonomous_recon(self,coeffs,subset=['na'],npoints=2000,leak=1,h=4,noise=0,title='na',lag=4):
        output = []
        for fo in range(npoints):
            for inc in range(h):
                next_inp = self.predict(coeffs,subset=subset)
                if inc == 0:
                    output.append(next_inp)
                #self.act(leak,next_inp)
                self.euler_increment(leak,next_inp,h=h)
            #self.add_noise(noise)
        fig,axs = plt.subplots(1,1,figsize=(6,6))
        if title != 'na':
            axs.set_title(title)
        axs.plot(output[:-lag],output[lag:])
        plt.tight_layout()
        plt.show()
        #print(output[:3]+output[-3:])
        return output
    def generate_state_matrix(self,TS,transient=1000,leak=1):
        X = [] 
        for t in TS:
            self.act(t,leak)
            X.append(self.state*1)
        X = np.array(X[transient:])
        num_rows = X.shape[0]
        ones_column = np.ones((num_rows, 1), dtype=X.dtype)
        X = np.hstack((X, ones_column))
        return X
            