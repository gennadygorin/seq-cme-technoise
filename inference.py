import pickle
import time
import numpy as np
from scipy import optimize
from preprocess import *
from cme_toolbox import *


class InferenceParameters:
    def __init__(self,phys_lb,phys_ub,samp_lb,samp_ub,gridsize,dataset_string,model,use_lengths=True,\
    			 gradient_params = {'max_iterations':10,'init_pattern':'moments','num_restarts':2}):
        #this happens before parallelization
        self.gradient_params = gradient_params
        self.phys_lb = np.array(phys_lb)
        self.phys_ub = np.array(phys_ub)
        self.grad_bnd = scipy.optimize.Bounds(phys_lb,phys_ub)

        self.use_lengths=True

        self.samp_lb = np.array(samp_lb)
        self.samp_ub = np.array(samp_ub)
        self.gridsize = gridsize
        self.construct_grid()

        self.dataset_string = dataset_string
        inference_string = dataset_string + '/' \
            + model.bio_model + '_' + model.seq_model + '_' \
            + '{:.0f}x{:.0f}'.format(gridsize[0],gridsize[1])
        make_dir(inference_string)
        inference_parameter_string = inference_string + '/parameters.pr'
        store_inference_parameters(self,inference_parameter_string)

    def construct_grid(self):
        X,Y = np.meshgrid(\
                          np.linspace(self.samp_lb[0],self.samp_ub[0],self.gridsize[0]),\
                          np.linspace(self.samp_lb[1],self.samp_ub[1],self.gridsize[1]),indexing='ij')
        X=X.flatten()
        Y=Y.flatten()
        self.X = X
        self.Y = Y
        self.sampl_vals = list(zip(X,Y))

class GradientInference:
    def __init__(self,global_parameters,search_data,model,point_index):
        #'regressor' is a bit generic...
        regressor = np.array([global_parameters.sampl_vals[point_index]]*search_data.n_genes)
        if global_parameters.use_lengths:
            if model.seq_model == 'Bernoulli':
                raise ValueError('The Bernoulli model does not yet have a physical length-based model.')
            elif model.seq_model == 'None': 
                raise ValueError('The model without technical noise has no length effects.')
            elif model.seq_model == 'Poisson':
                regressor[:,0] += search_data.gene_log_lengths
            else:
                raise ValueError('Please select a technical noise model from {Poisson}, {Bernoulli}, {None}.')
        self.regressor = regressor
        self.grad_bnd = global_parameters.grad_bnd
        self.gradient_params = global_parameters.gradient_params
        self.phys_lb = global_parameters.phys_lb
        self.phys_ub = global_parameters.phys_ub
        if self.gradient_params['init_pattern'] == 'moments':
            self.param_MoM = np.asarray([model.get_MoM(\
                                    search_data.moments[i],\
                                    global_parameters.phys_lb,\
                                    global_parameters.phys_ub,\
                                    regressor[i]) for i in range(search_data.n_genes)])
            
    def optimize_gene(self,gene_index,model,search_data):
        n_phys_pars = len(self.phys_lb)
        x0 = np.random.rand(self.gradient_params['num_restarts'],n_phys_pars)*\
                (self.phys_ub-self.phys_lb)+self.phys_lb
        if self.gradient_params['init_pattern'] == 'moments': #this can be extended to other initialization patterns, like latin squares
            x0[0] = self.param_MoM[gene_index]
        x = x0[0]
        err = np.inf
        ERR_THRESH = 0.99

        for restart in range(self.gradient_params['num_restarts']):
            res_arr = scipy.optimize.minimize(lambda x: \
                        kl_div(
                           data=search_data.hist[gene_index],
                           proposal=model.eval_model_pss(
                               x, 
                               [search_data.M[gene_index],search_data.N[gene_index]], 
                               self.regressor[gene_index])),
                        x0=x0[restart], \
                        bounds=self.grad_bnd,\
                        options={'maxiter':self.gradient_params['max_iterations'],'disp':False})
            if res_arr.fun < err*ERR_THRESH: #do not replace old best estimate if there is little marginal benefit
                x = res_arr.x
                err = res_arr.fun
        return x, err

    def iterate_over_genes(self,model,search_data):
        t1 = time.time()
        
        param_estimates, klds = zip(*[self.optimize_gene(gene_index,model,search_data) for gene_index in range(search_data.n_genes)])
        
        klds = np.asarray(klds)
        param_estimates = np.asarray(param_estimates)
        obj_func = klds.sum()
        
        t2 = time.time()
        d_time = t2-t1

        return param_estimates, klds, obj_func, d_time

def kl_div(data, proposal,EPS=1e-15):
    """
    Kullback-Leibler divergence between experimental data histogram and proposed PMF. Proposal clipped at EPS.
    """
    proposal[proposal<EPS]=EPS
    filt = data>0
    data = data[filt]
    proposal = proposal[filt]
    d=data*np.log(data/proposal)
    return np.sum(d)

def store_inference_parameters(inference_parameters,inference_parameter_string):
    try:
        with open(inference_parameter_string,'wb') as ipfs:
            pickle.dump(inference_parameters, ipfs)
        log.info('Global inference parameters stored to {}.'.format(inference_parameter_string))
    except:
        log.error('Global inference parameters could not be stored to {}.'.format(inference_parameter_string))
