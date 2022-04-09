
import numpy as np
import numpy.matlib

import scipy

import random
from scipy.fft import irfft2


import scipy.stats.mstats
from scipy.stats import *
import numdifftools

class CMEModel:
    def __init__(self,bio_model,seq_model,fixed_quad_T=10,quad_order=60,quad_vec_T=np.inf,quad_method='fixed_quad'):
        self.bio_model = bio_model
        self.seq_model = seq_model
        self.set_integration_parameters(fixed_quad_T,quad_order,quad_vec_T,quad_method)

    def set_integration_parameters(self,fixed_quad_T,quad_order,quad_vec_T,quad_method):
        self.fixed_quad_T = fixed_quad_T
        self.quad_order = quad_order
        self.quad_vec_T = quad_vec_T
        self.quad_method = quad_method

    def eval_model_pss(self,p,limits,samp=None):
        u = []
        mx = np.copy(limits)
        mx[-1] = mx[-1]//2 + 1
        for i in range(len(mx)):
            l = np.arange(mx[i])
            u_ = np.exp(-2j*np.pi*l/limits[i])-1
            if self.seq_model == 'Poisson':
                u_ = np.exp((10**samp[i])*u_)-1
            elif self.seq_model == 'Bernoulli': #it might be better to have this one in terms of a positive optimizable value
                u_ *= samp[i]
            elif self.seq_model == 'None' or self.seq_model is None:
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
        Pss = irfft2(gf, s=tuple(limits)) 
        Pss = np.abs(Pss)/np.sum(np.abs(Pss))
        return Pss


    def eval_model_pgf(self,p,g):
        p = 10**p #these are going to have different interpretations for different models. Should we harmonize them?
        if self.bio_model == 'Poisson': #constitutive production
            gf = g[0]*p[0] + g[1]*p[1]
        elif self.bio_model == 'Bursty': #bursty production
            b,beta,gamma = p
            fun = lambda x: self.burst_intfun(x,g,b,beta,gamma)
            if self.quad_method=='quad_vec':
                T = self.quad_vec_T*(1/beta + 1/gamma + 1)
                gf = scipy.integrate.quad_vec(fun,0,T)[0]
            elif self.quad_method=='fixed_quad':
                T = self.fixed_quad_T*(1/beta + 1/gamma + 1)
                gf = scipy.integrate.fixed_quad(fun,0,T,n=self.quad_order)[0]
            else:
                raise ValueError('Please use one of the specified quadrature methods.')
        elif self.bio_model == 'Extrinsic': #constitutive production with extrinsic noise
            raise ValueError('I still need to implement this one.')
        elif self.bio_model == 'Delay': #bursty production with delayed degradation
            b,beta,tauinv = p
            tau = 1/tauinv
            U  = g[1] + (g[0]-g[1])*np.exp(-beta*tau)
            gf = -1/beta * np.log(1-b*U) + 1/beta/(1-b*g[1]) * np.log((b*U-1)/(b*g[0]-1)) + tau * b*g[1]/(1-b*g[1])
        else:
            raise ValueError('Please select a biological noise model from {Poisson}, {Bursty}, {Extrinsic}, {Delay}.')
        return gf #this is the log-generating function

    def burst_intfun(self,x,g,b,beta,gamma):
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

#rewrite this whole thing so it can use any of the models.
    # def get_MoM(moment_data,lb_log,ub_log,samp=None):
        """
        Initialize parameter search at the method of moments estimates.
        lower bound and upper bound are harmonized with optimization routine and input as log10.
        """

        # lb = 10**lb_log
        # ub = 10**ub_log
        
        # var_U, mean_U, mean_S = moment_data
        # b = var_U / mean_U - 1
        # if samp is not None:
        #     samp = 10**samp
        #     b = b / samp[0] - 1
        # else:
        #     samp = [1,1]
        # b = np.clip(b,lb[0],ub[0])
        # bet = np.clip(b * samp[0] / mean_U, lb[1], ub[1])
        # gam = np.clip(b * samp[1] / mean_S, lb[2], ub[2])
        # x0 = np.log10(np.asarray([b,bet,gam]))
        # return x0