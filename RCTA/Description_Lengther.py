import numpy as np 


def L_model_class(X,alpha,method = 'DL'):
    if method == 'DL':
        eigs,vals = np.linalg.eigh(np.matmul(np.transpose(X),X))
        Qeigs = np.square(alpha*np.reciprocal(eigs+alpha))
        Lclass = np.real(-0.5*np.sum(np.log(Qeigs)))
    elif method == 'AIC':
        P = np.matmul(X,np.matmul(np.linalg.inv(np.matmul(np.transpose(X),X)+np.diag(np.ones(X.shape[1]))*alpha),
                                         np.transpose(X)))
        Lclass = 2*np.trace(P)
    return Lclass

def Qmatrix(X,Y,alpha):
    Ident = np.diag(np.ones(X.shape[1]))
    bigIdent = np.diag(np.ones(X.shape[0]))
    Minv = np.linalg.inv(np.matmul(np.transpose(X),X)+alpha*Ident)
    P = np.matmul(X,np.matmul(Minv,np.transpose(X)))
    Q = np.matmul(P-bigIdent,P-bigIdent)
    return Q 


    
def fit_coefs(V,Y,alpha=0,transient=100):
    return np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(V[transient:,:]),V[transient:,:])+
                                             np.diag(np.ones(V[transient:,:].shape[1]))*alpha),
                                     np.transpose(V[transient:,:])),Y[transient:])

def fit_coefsP(V,Y,alpha=0,transient=100):
    return np.matmul(np.matmul(np.linalg.pinv(np.matmul(np.transpose(V[transient:,:]),V[transient:,:])+
                                             np.diag(np.ones(V[transient:,:].shape[1]))*alpha),
                                     np.transpose(V[transient:,:])),Y[transient:])

def get_epsi(X,Y,alpha=10**(-16)):
    coefs = fit_coefsP(X, Y,alpha=alpha)
    epsilon = np.mean(np.square(np.matmul(X[100:,:],np.transpose(coefs))-Y[100:]))*(len(Y)/(len(Y)-len(coefs)))
    return epsilon

def get_total_DL(X,Y,alpha,epsilon=0,method='DL',transient=100):
    if method == 'DL':
        if epsilon == 0:
            coefs = fit_coefs(X, Y,alpha=0)
            epsilon = np.mean(np.square(np.matmul(X,np.transpose(coefs))-Y))
        coefs = fit_coefs(X, Y,alpha=alpha, transient=transient)
        MSE = MSE = np.mean(np.square(np.matmul(X[transient:,:],np.transpose(coefs))-Y[transient:]))
        Lpar = L_model_class(X[transient:,:], alpha)
        Ltot = len(Y[transient:])*MSE/(2*epsilon)+Lpar 
    elif method == 'AIC':
        if epsilon == 0:
            coefs = fit_coefs(X, Y,alpha=0)
            epsilon = np.mean(np.square(np.matmul(X,np.transpose(coefs))-Y))
        coefs = fit_coefs(X, Y,alpha=alpha, transient=transient)
        MSE = np.mean(np.square(np.matmul(X[transient:,:],np.transpose(coefs))-Y[transient:]))
        Lpar = L_model_class(X[transient:,:], alpha, method='AIC')
        Ltot = len(Y[transient:])*MSE/epsilon+Lpar 
    coefs = fit_coefsP(X, Y,alpha=10**(-16))
    MSE_unb = np.mean(np.square(np.matmul(X[transient:,:],np.transpose(coefs))-Y[transient:]))
    MSE_unb = (len(Y[transient:])/(len(Y[transient:])-len(coefs)))*MSE_unb
    return Ltot, MSE, Lpar, MSE_unb

def get_gen_err(X,Y,X2,Y2,alpha):
    coefs = fit_coefs(X, Y, alpha)
    MSE = np.mean(np.square(np.matmul(X2,np.transpose(coefs))-Y2))
    return MSE