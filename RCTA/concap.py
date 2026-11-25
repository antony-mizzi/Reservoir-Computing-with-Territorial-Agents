# Measuring Consistency Capacity 
import numpy as np 

def get_concap(X,X2,alpha=10**(-8)):
    X = X-X.mean(axis=0, keepdims=True)
    row_means = np.maximum(np.linalg.norm(X, axis=0, keepdims=True),10**(-6))
    X = X / row_means
    X2 = (X2-X.mean(axis=0, keepdims=True))
    row_means2 = np.maximum(np.linalg.norm(X2, axis=0, keepdims=True),10**(-6))
    X2 = X2/row_means2
    C = np.matmul(np.transpose(X),X)+alpha*np.diag(np.ones(X.shape[1]))
    vals,Q = np.linalg.eig(C)
    del C, row_means, row_means2
    #sigma2 = np.diag(vals)+np.diag(10**(-6)*np.ones(len(vals)))
    sigminv = np.diag([1/(np.sqrt(abs(v))) for v in vals])
    T = np.matmul(Q,np.matmul(sigminv,np.transpose(Q)))
    del vals, Q, sigminv
    Xtild = np.matmul(X,T)
    del X
    Xtild2 = np.matmul(X2,T)
    del X2, T
    Cc = np.matmul(np.transpose(Xtild),Xtild2)#+10**(-9)*np.diag(np.ones(Xtild.shape[1]))
    del Xtild, Xtild2
    #Cc = np.matmul(np.transpose(np.matmul(X,T)),np.matmul(X2,T))
    gammas,Qc = np.linalg.eig(Cc)
    #print('concap = '+str(sum([abs(G) for G in gammas])))
    #print('concap = '+str(sum([G.real for G in gammas])))
    return sum([G.real for G in gammas])