import numpy as np
# import numpy.matlib

import scipy

from scipy.fft import irfft2
from preprocess import *

# import random
# from scipy.fft import irfft2


# import scipy.stats.mstats
# from scipy.stats import *
# import numdifftools

class CMEModel:
    def __init__(self,bio_model,seq_model,quad_method='fixed_quad',fixed_quad_T=10,quad_order=60,quad_vec_T=np.inf):
        self.bio_model = bio_model
        self.available_biomodels = ('Delay','Bursty','Extrinsic','Constitutive','CIR')
        self.available_seqmodels = ('None','Bernoulli','Poisson')
        if (seq_model == 'None') or (seq_model is None) or (seq_model == 'Null'):
            self.seq_model = 'None'
        else: 
            self.seq_model = seq_model
        self.set_integration_parameters(fixed_quad_T,quad_order,quad_vec_T,quad_method)

    def set_integration_parameters(self,fixed_quad_T,quad_order,quad_vec_T,quad_method):
        self.fixed_quad_T = fixed_quad_T
        self.quad_order = quad_order
        self.quad_vec_T = quad_vec_T
        self.quad_method = quad_method

    def get_log_name_str(self):
        if self.bio_model == 'Constitutive':
            return (r'$\log_{10} \beta$',r'$\log_{10} \gamma$')
        elif self.bio_model == 'Delay':
            return (r'$\log_{10} b$',r'$\log_{10} \beta$',r'$\log_{10} \tau^{-1}$')
        elif self.bio_model == 'Bursty':
            return (r'$\log_{10} b$',r'$\log_{10} \beta$',r'$\log_{10} \gamma$')
        elif self.bio_model == 'Extrinsic': 
            return (r'$\log_{10} \alpha$',r'$\log_{10} \beta$',r'$\log_{10} \gamma$')
        elif self.bio_model == 'CIR': 
            return (r'$\log_{10} b$',r'$\log_{10} \beta$',r'$\log_{10} \gamma$')
        else:
            raise ValueError('Please select a biological noise model from {}.'.format(self.available_biomodels))

    def get_num_params(self):
        if self.bio_model == 'Constitutive':
            return 2
        else: 
            return 3


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
            elif self.seq_model == 'None':
                pass
            else:
                raise ValueError('Please select a technical noise model from {}.'.format(self.available_seqmodels))
            u.append(u_)
        g = np.meshgrid(*[u_ for u_ in u], indexing='ij')
        for i in range(len(mx)):
            g[i] = g[i].flatten()[:,np.newaxis]

        gf = self.eval_model_pgf(p,g)
        gf = np.exp(gf)
        gf = gf.reshape(tuple(mx))
        Pss = irfft2(gf, s=tuple(limits)) 
        Pss = np.abs(Pss)/np.sum(np.abs(Pss))
        return Pss.squeeze()


    def eval_model_pgf(self,p_,g):
        p = 10**p_
        if self.bio_model == 'Constitutive': #constitutive production
            beta,gamma = p
            gf = g[0]/beta + g[1]/gamma
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
            alpha,beta,gamma = p
            gf = -alpha * np.log(1 - g[0]/beta - g[1]/gamma)
        elif self.bio_model == 'Delay': #bursty production with delayed degradation
            b,beta,tauinv = p
            tau = 1/tauinv
            U  = g[1] + (g[0]-g[1])*np.exp(-beta*tau)
            gf = -1/beta * np.log(1-b*U) + 1/beta/(1-b*g[1]) * np.log((b*U-1)/(b*g[0]-1)) + tau * b*g[1]/(1-b*g[1])
        elif self.bio_model == 'CIR': #CIR-like:
            b,beta,gamma = p
            fun = lambda x: self.cir_intfun(x,g,b,beta,gamma)
            if self.quad_method=='quad_vec':
                T = self.quad_vec_T*(1/beta + 1/gamma + 1)
                gf = scipy.integrate.quad_vec(fun,0,T)[0]
            elif self.quad_method=='fixed_quad':
                T = self.fixed_quad_T*(1/beta + 1/gamma + 1)
                gf = scipy.integrate.fixed_quad(fun,0,T,n=self.quad_order)[0]
            else:
                raise ValueError('Please use one of the specified quadrature methods.')
            gf /= 2
        else:
            raise ValueError('Please select a biological noise model from {}.'.format(self.available_biomodels))
        return gf #this is the log-generating function
    
    def cir_intfun(self,x,g,b,beta,gamma):
        """
        Computes the IG-driven integrand at time x.
        """
        if np.isclose(beta,gamma): #compute prefactors for the ODE characteristics.
            c_1 = g[0] #nascent
            c_2 = x*beta*g[1]
        else:
            f = beta/(beta-gamma)
            c_2 = g[1]*f
            c_1 = g[0] - c_2

        U = b * (np.exp(-beta*x)*c_1 + np.exp(-gamma*x)*c_2)
        return 1-np.sqrt(1-4*U)


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
    def get_MoM(self,moments,lb_log,ub_log,samp=None):
        """
        Initialize parameter search at the method of moments estimates.
        lower bound and upper bound are harmonized with optimization routine and input as log10.
        """

        lb = 10**lb_log
        ub = 10**ub_log
        if self.seq_model == 'Poisson':
            samp = 10**samp
        
        if self.bio_model == 'Bursty' or self.bio_model == 'CIR':
            b = moments['U_var'] / moments['U_mean'] - 1

            if self.seq_model == 'Bernoulli':
                b /= samp[0]
            elif self.seq_model == 'Poisson':
                b = b/samp[0] - 1

            beta = b / moments['U_mean']
            gamma = b / moments['S_mean']
            x0 = np.asarray([b,beta,gamma])

        elif self.bio_model == 'Delay':            
            b = moments['U_var'] / moments['U_mean'] - 1

            if self.seq_model == 'Bernoulli':
                b /= samp[0]
            elif self.seq_model == 'Poisson':
                b = b/samp[0] - 1

            beta = b / moments['U_mean']
            tauinv = b / moments['S_mean']
            x0 = np.asarray([b,beta,tauinv])

        elif self.bio_model == 'Constitutive':
            beta = 1 / moments['U_mean']
            gamma = 1 / moments['S_mean']
            x0 = np.asarray([beta,gamma])

        elif self.bio_model == 'Extrinsic':
            if self.seq_model == 'Poisson':
                alpha = moments['U_mean']**2/(moments['U_var'] - moments['U_mean']*(1+samp[0]))
            else:
                alpha = moments['U_mean']**2/(moments['U_var'] - moments['U_mean'])

            beta = alpha / moments['U_mean']
            gamma = alpha / moments['S_mean']
            x0 = np.asarray([alpha,beta,gamma])
        else:
            raise ValueError('Please select from existing models.')

        if self.bio_model == 'Constitutive':
            x0 *= samp
        else:
            x0[1:] = x0[1:] * samp
        for j in range(self.get_num_params()):
            x0[j] = np.clip(x0[j],lb[j],ub[j])
        x0 = np.log10(x0)
        return x0

    def eval_model_noise(self,p,samp=None):
        p=10**p
        if self.bio_model == 'Constitutive': #constitutive production
            beta,gamma = p
            mu = [1/beta,1/gamma]
        elif self.bio_model == 'Bursty': #bursty production
            b,beta,gamma = p
            mu = [b/beta,b/gamma]
        elif self.bio_model == 'Extrinsic': #constitutive production with extrinsic noise
            alpha,beta,gamma = p
            mu = [alpha/beta,alpha/gamma]
        elif self.bio_model == 'Delay': #bursty production with delayed degradation
            raise ValueError('Not yet implemented!')    
        elif self.bio_model == 'CIR': #CIR-like:
            b,beta,gamma = p
            mu = [b/beta,b/gamma]

        mu = np.asarray(mu)
        noise_int = 1/mu

        if self.bio_model == 'Constitutive': #constitutive production
            noise_ext = [0,0]
        elif self.bio_model == 'Bursty': #bursty production
            noise_ext = [beta,beta*gamma/(beta+gamma)]
        elif self.bio_model == 'Extrinsic': #constitutive production with extrinsic noise
            noise_ext = [1/alpha,1/alpha]
        elif self.bio_model == 'Delay': #bursty production with delayed degradation
            raise ValueError('Not yet implemented!')    
        elif self.bio_model == 'CIR': #CIR-like:
            noise_ext = [beta,beta*gamma/(beta+gamma)]
        noise_ext = np.asarray(noise_ext)

        if self.seq_model == 'None':
            noise = noise_int + noise_ext
            return (noise_int/noise, noise_ext/noise)
        elif self.seq_model == 'Bernoulli':
            noise = noise_int/samp + noise_ext
            return (noise_int/noise, noise_ext/noise, 1-noise_int/noise-noise_ext/noise)
        elif self.seq_model == 'Poisson':
            samp = 10**samp
            noise_tech = 1/(mu*samp)
            noise = noise_int + noise_ext + noise_tech
            return (noise_int/noise, noise_ext/noise, noise_tech/noise)