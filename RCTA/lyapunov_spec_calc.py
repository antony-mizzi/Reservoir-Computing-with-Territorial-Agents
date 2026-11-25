import numpy as np 
import maps_and_systems as mapnsys
import random as rand
import time
import math
import matplotlib.pyplot as plt

#Gram Schmidt orthogonalisation
def gram_schmidt(A): 
    n = len(A); omegas = []
    for j in range(n):
        for k in range(j):
            A[j] = A[j] - np.dot(A[k],A[j])*A[k]
        omegas.append(np.linalg.norm(A[j]))
        A[j] = A[j]/np.linalg.norm(A[j])
    return A,omegas

#Governing eqn for Lorenz96
def L96(x, R=8):
    return (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1) - x + R

#Governing eqn for Lorenz
def lorenz(x,sigma=10,rho=28,beta=8/3):
    return np.array([sigma*(x[1]-x[0]),x[0]*(rho-x[2]) - x[1], x[0]*x[1] - beta*x[2]])

#governing eqn for Rossler
def Rossler(x,dt=0.01,a=0.1,b=0.1,c=14):
    return np.array([ -x[1]-x[2], x[0] + a*x[1],b + x[2]*(x[0]-c) ])

def halvorsen(x,a=1.4):
    return np.array([-a*x[0]-4*x[1]-4*x[2]-x[1]**2,-a*x[1]-4*x[2]-4*x[0]-x[2]**2,-a*x[2]-4*x[0]-4*x[1]-x[0]**2])

def lorenz_inc_function(x,sigma=10,rho=28,beta=8/3):
    return [sigma*(x[1]-x[0]),x[0]*(rho-x[2]) - x[1], x[0]*x[1] - beta*x[2]]

def rossler_inc_function(x,a=0.1,b=0.1,c=14):
    return [-x[1]-x[2],x[0] + a*x[1], b + x[2]*(x[0]-c)]

def thomas_inc_function(x,b=0.208186):
    return [math.sin(x[1])-b*x[0], math.sin(x[2]) - b*x[1], math.sin(x[0]) - b*x[2]]

def halvorsen_inc_function(x,a=1.4):
    return [-a*x[0]-4*x[1]-4*x[2]-x[1]**2,-a*x[1]-4*x[2]-4*x[0]-x[2]**2,-a*x[2]-4*x[0]-4*x[1]-x[0]**2]

def Lorenz_runge_step(x,dt=0.001,sigma=10,rho=28,beta=8/3):
    k1 = lorenz_inc_function(x,sigma=sigma,rho=rho,beta=beta)
    k2 = lorenz_inc_function([x[0]+0.5*dt*k1[0],x[1]+0.5*dt*k1[1],x[2]+0.5*dt*k1[2]],sigma=sigma,rho=rho,beta=beta)
    k3 = lorenz_inc_function([x[0]+0.5*dt*k2[0],x[1]+0.5*dt*k2[1],x[2]+0.5*dt*k2[2]],sigma=sigma,rho=rho,beta=beta)
    k4 = lorenz_inc_function([x[0]+0.5*dt*k3[0],x[1]+0.5*dt*k3[1],x[2]+0.5*dt*k3[2]],sigma=sigma,rho=rho,beta=beta)
    return np.array([x[0]+dt*(k1[0]+2*k2[0]+2*k3[0]+k4[0])/6,x[1]+dt*(k1[1]+2*k2[1]+2*k3[1]+k4[1])/6,
            x[2]+dt*(k1[2]+2*k2[2]+2*k3[2]+k4[2])/6])

def Rossler_runge_step(x,dt=0.001,a=0.1,b=0.1,c=14):
    k1 = rossler_inc_function(x,a=a,b=b,c=c)
    k2 = rossler_inc_function([x[0]+0.5*dt*k1[0],x[1]+0.5*dt*k1[1],x[2]+0.5*dt*k1[2]],a=a,b=b,c=c)
    k3 = rossler_inc_function([x[0]+0.5*dt*k2[0],x[1]+0.5*dt*k2[1],x[2]+0.5*dt*k2[2]],a=a,b=b,c=c)
    k4 = rossler_inc_function([x[0]+0.5*dt*k3[0],x[1]+0.5*dt*k3[1],x[2]+0.5*dt*k3[2]],a=a,b=b,c=c)
    return [x[0]+dt*(k1[0]+2*k2[0]+2*k3[0]+k4[0])/6,x[1]+dt*(k1[1]+2*k2[1]+2*k3[1]+k4[1])/6,
            x[2]+dt*(k1[2]+2*k2[2]+2*k3[2]+k4[2])/6]

def Thomas_runge_step(x,dt=0.001,b=0.2):
    k1 = thomas_inc_function(x,b=b)
    k2 = thomas_inc_function([x[0]+0.5*dt*k1[0],x[1]+0.5*dt*k1[1],x[2]+0.5*dt*k1[2]],b=b)
    k3 = thomas_inc_function([x[0]+0.5*dt*k2[0],x[1]+0.5*dt*k2[1],x[2]+0.5*dt*k2[2]],b=b)
    k4 = thomas_inc_function([x[0]+0.5*dt*k3[0],x[1]+0.5*dt*k3[1],x[2]+0.5*dt*k3[2]],b=b)
    return [x[0]+dt*(k1[0]+2*k2[0]+2*k3[0]+k4[0])/6,x[1]+dt*(k1[1]+2*k2[1]+2*k3[1]+k4[1])/6,
            x[2]+dt*(k1[2]+2*k2[2]+2*k3[2]+k4[2])/6]

def Halvorsen_runge_step(x,dt=0.001,a=1.4):
    k1 = halvorsen_inc_function(x,a=a)
    k2 = halvorsen_inc_function([x[0]+0.5*dt*k1[0],x[1]+0.5*dt*k1[1],x[2]+0.5*dt*k1[2]],a=a)
    k3 = halvorsen_inc_function([x[0]+0.5*dt*k2[0],x[1]+0.5*dt*k2[1],x[2]+0.5*dt*k2[2]],a=a)
    k4 = halvorsen_inc_function([x[0]+0.5*dt*k3[0],x[1]+0.5*dt*k3[1],x[2]+0.5*dt*k3[2]],a=a)
    return [x[0]+dt*(k1[0]+2*k2[0]+2*k3[0]+k4[0])/6,x[1]+dt*(k1[1]+2*k2[1]+2*k3[1]+k4[1])/6,
            x[2]+dt*(k1[2]+2*k2[2]+2*k3[2]+k4[2])/6]

#Purturb state (x0) by a value of epsilon in the orthogonal directions V
#Integrate the system foward by tau steps after each purturbation
#returns the normalised effects of the purturbations when integrated foward through the system 
def difference_eq(V,x0,tau,epsilon=10**(-5),dt=0.001):
    U = []
    for v in V:
        x = x0+epsilon*v
        for t in range(tau):
            #change L96(x) to lorenz(x) for Lorenz
            x += dt*halvorsen(x)
        U.append(x+np.random.normal(0,10**(-12),size=len(x)))
    xfinal = x0
    for t in range(tau):
        #change L96(xfinal) to lorenz(xfinal) for Lorenz
        xfinal += dt*halvorsen(xfinal)

    return [(u-xfinal)/epsilon for u in U], xfinal

#Compute the full Lyapunov spectrum of Lorenz96 or Lorenz
#n is the dimension of the state space (3 for Lorenz, 15 for Lorenz96)
#reps is the number of purturbation re-orthogonlaisation cycles 
#epsilon defines purturbation size, tau is the number of steps integrated foward after each purturbation 
#and trained_dt is the size of the time step used for integration 
def get_spectrum(n,state,reps=10000,epsilon=10**(-5),tau=15,trained_dt=0.001):
    U = np.diag(np.ones(n)).tolist(); Omegas = {i:[] for i in range(n)}
    x = np.array(state.tolist())
    
    #Transient period to make sure we start on the attractor. Change L96() to lorenz() for Lorenz
    for transient in range(10000):
        x += trained_dt*halvorsen(x)
    
    #Loop through the purturbation and re-orthogonalisation procedure 
    #Eventually, V should point along the Lyapunov Vectors
    for r in range(reps):
        if r % 100 == 0:
            print(r)
        V,omegas = gram_schmidt(U)
        for i in range(n):
            Omegas[i].append(omegas[i])
        U,x = difference_eq(V, x, tau, epsilon=epsilon,dt=trained_dt)

    #This is the full Lyapunov spectrum (in order)
    exponents = {i:np.sum([math.log(w) for w in Omegas[i]])/(len(Omegas[i])*trained_dt*tau) for i in Omegas.keys()}
    return exponents

state = np.random.normal(0,0.8,size=3)
expons = get_spectrum(3, state)
print(expons)