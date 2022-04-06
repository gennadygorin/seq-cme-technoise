
import numpy as np
import numpy.matlib

import scipy

import random
from scipy.fft import irfft2


import scipy.stats.mstats
from scipy.stats import *
import numdifftools

class CMEModel:
    def __init__(self,bio_model,seq_model):
        self.bio_model = bio_model
        self.seq_model = seq_model
    def eval_model_pss(p,limits,samp=None):
        b,bet,gam = 10**p
        u = []
        mx = np.copy(limits)
        mx[-1] = mx[-1]//2 + 1
        for i in range(len(mx)):
            l = np.arange(mx[i])
            u_ = np.exp(-2j*np.pi*l/lm[i])-1
            if self.seq_model == 'Poisson':
                u_ = np.exp((10**samp[i])*u_)-1
            elif self.seq_model == 'Bernoulli':
                u_ *= samp[i]
            elif self.seq_model == 'None':
                pass
            else:
                raise ValueError('Please select a technical noise model from {Poisson}, {Bernoulli}, {None}.')
            u.append(u_)
        g = np.meshgrid(*[u_ for u_ in u], indexing='ij')
        for i in range(len(mx)):
            g[i] = g[i].flatten()[:,np.newaxis]

        gf = self.eval_model_pgf(p,g)
        gf = np.exp(gf)
        gf = gf.reshape(tuple(mx))
        Pss = irfft2(gf, s=tuple(lm)) 
        Pss = np.abs(Pss)/np.sum(np.abs(Pss))
        return Pss


    def eval_model_pgf(p,g,quad_method='fixed_quad',fixed_quad_T=10,quad_order=60,quad_vec_T=np.inf):
        p = 10**p
        if self.bio_model == 'Poisson':
            gf = g[0]*p[0] + g[1]*p[1]
        elif self.bio_model == 'Bursty':
            b,beta,gamma = p
            fun = lambda x: self.burst_intfun(x,g,b,beta,gamma)
            if quad_method=='quad_vec':
                T = quad_vec_T*(1/beta + 1/gamma + 1)
                gf = scipy.integrate.quad_vec(fun,0,T)[0]
            if quad_method=='fixed_quad':
                T = fixed_quad_T*(1/bet+1/gam)
                gf = scipy.integrate.fixed_quad(fun,0,T,n=quad_order)[0]
        elif self.bio_model == 'Extrinsic':
            raise ValueError('I still need to implement this one.')
        elif self.bio_model == 'Delay':
            raise ValueError('I still need to implement this one.')
        else:
            raise ValueError('Please select a biological noise model from {Poisson}, {Bursty}, {Extrinsic}, {Delay}.')
        return gf

    def burst_intfun(x,g,b,beta,gamma):
    """
    Computes the Singh-Bokes integrand at time x.
    """
    if np.isclose(beta,gamma): #compute prefactors for the ODE characteristics.
        c_1 = g[0] #nascent
        c_2 = x*beta*g[1]
    else:
        f = beta/(beta-gamma)
        c_2 = g[1]*f
        c_1 = g[0] - c_2

    U = b * (np.exp(-beta*x)*c_1 + np.exp(-gamma*x)*c_2)
    return U/(1-U)
