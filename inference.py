import pickle
import time
import numpy as np
from scipy import optimize
from preprocess import *
from cme_toolbox import *
import multiprocessing


class InferenceParameters:
    def __init__(self,phys_lb,phys_ub,samp_lb,samp_ub,gridsize,dataset_string,model,use_lengths=True,\
    			 gradient_params = {'max_iterations':10,'init_pattern':'moments','num_restarts':1}):
        #this happens before parallelization
        self.gradient_params = gradient_params
        self.phys_lb = np.array(phys_lb)
        self.phys_ub = np.array(phys_ub)
        self.grad_bnd = scipy.optimize.Bounds(phys_lb,phys_ub)

        self.use_lengths=use_lengths

        self.samp_lb = np.array(samp_lb)
        self.samp_ub = np.array(samp_ub)
        self.gridsize = gridsize
        self.construct_grid()

        self.dataset_string = dataset_string
        inference_string = dataset_string + '/' \
            + model.bio_model + '_' + model.seq_model + '_' \
            + '{:.0f}x{:.0f}'.format(gridsize[0],gridsize[1])
        make_dir(inference_string)
        self.inference_string = inference_string
        inference_parameter_string = inference_string + '/parameters.pr'
        self.store_inference_parameters(inference_parameter_string)

    def construct_grid(self):
        X,Y = np.meshgrid(\
                          np.linspace(self.samp_lb[0],self.samp_ub[0],self.gridsize[0]),\
                          np.linspace(self.samp_lb[1],self.samp_ub[1],self.gridsize[1]),indexing='ij')
        X=X.flatten()
        Y=Y.flatten()
        self.X = X
        self.Y = Y
        self.sampl_vals = list(zip(X,Y))
        self.n_grid_points = len(X)

    def store_inference_parameters(self,inference_parameter_string):
        try:
            with open(inference_parameter_string,'wb') as ipfs:
                pickle.dump(self, ipfs)
            log.info('Global inference parameters stored to {}.'.format(inference_parameter_string))
        except:
            log.error('Global inference parameters could not be stored to {}.'.format(inference_parameter_string))

    def fit_all_grid_points(self,num_cores,search_data,model):
        if num_cores>1:
            log.info('Starting parallelized grid scan.')
            pool=multiprocessing.Pool(processes=num_cores)
            #This might be improved.
            pool.map(self.par_fun,zip(range(self.n_grid_points),[[search_data,model]]*self.n_grid_points))
            pool.close()
            pool.join()
            log.info('Parallelized grid scan complete.')
        else:
            log.info('Starting non-parallelized grid scan.')
            for point_index in range(self.n_grid_points):
                grad_inference = GradientInference(self,model,search_data,point_index)
                grad_inference.fit_all_genes(model,search_data)
            log.info('Non-parallelized grid scan complete.')
        self.store_search_results()

    def store_search_results(self):
        results = SearchResults(self)
        results.aggregate_grid_points()
        try:
            full_result_string = self.inference_string + '/grid_scan_results.res'
            with open(full_result_string,'wb') as srfs:
                pickle.dump(results, srfs)
            log.debug('Grid scan results stored to {}.'.format(self.point_index,full_result_string))
        except:
            log.error('Grid scan results could not be stored to {}.'.format(self.point_index,full_result_string))


    def par_fun(self,inputs):
        point_index,(search_data,model) = inputs
        grad_inference = GradientInference(self,model,search_data,point_index)
        grad_inference.fit_all_genes(model,search_data)


class GradientInference:
    def __init__(self,global_parameters,model,search_data,point_index):
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
        self.grid_point = global_parameters.sampl_vals[point_index]
        self.point_index = point_index
        self.regressor = regressor
        self.grad_bnd = global_parameters.grad_bnd
        self.gradient_params = global_parameters.gradient_params
        self.phys_lb = global_parameters.phys_lb
        self.phys_ub = global_parameters.phys_ub
        self.inference_string = global_parameters.inference_string
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

    def fit_all_genes(self,model,search_data):
        search_out = self.iterate_over_genes(model,search_data)
        results = GridPointResults(*search_out,self.regressor,self.grid_point,self.point_index,self.inference_string)
        results.store_grid_point_results()

########################
## Helper functions
########################
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

########################
## Helper classes
########################
class GridPointResults:
    def __init__(self,param_estimates, klds, obj_func, d_time,regressor,grid_point,point_index,inference_string):
        self.param_estimates = param_estimates
        self.klds = klds
        self.obj_func = obj_func
        self.d_time = d_time
        self.regressor = regressor
        self.grid_point = grid_point
        self.point_index = point_index
        self.inference_string = inference_string
    def store_grid_point_results(self):
        try:
            grid_point_result_string = self.inference_string + '/grid_point_'+str(self.point_index)+'.gp'
            with open(grid_point_result_string,'wb') as gpfs:
                pickle.dump(self, gpfs)
            log.debug('Grid point {:.0f} results stored to {}.'.format(self.point_index,grid_point_result_string))
        except:
            log.error('Grid point {:.0f} results could not be stored to {}.'.format(self.point_index,grid_point_result_string))

class SearchResults:
    def __init__(self,inference_parameters):
        #load in search parameter data
        self.inference_string = inference_parameters.inference_string
        self.phys_lb = inference_parameters.phys_lb
        self.phys_ub = inference_parameters.phys_ub
        self.use_lengths=inference_parameters.use_lengths
        self.samp_lb = inference_parameters.samp_lb
        self.samp_ub = inference_parameters.samp_ub

        self.X = inference_parameters.X
        self.Y = inference_parameters.Y
        self.sampl_vals = inference_parameters.sampl_vals
        self.n_grid_points = inference_parameters.n_grid_points

        self.param_estimates = []
        self.klds = []
        self.obj_func = []
        self.d_time = []

    def aggregate_grid_points(self):
        for point_index in range(self.n_grid_points):
            self.append_grid_point(point_index)
        self.clean_up()

    def append_grid_point(self, grid_point_index):
        grid_point_result_string = self.inference_string + '/grid_point_'+str(grid_point_index)+'.gp'
        with open(grid_point_result_string,'rb') as ipfs:
            grid_point_results = pickle.load(ipfs)
            self.param_estimates += grid_point_results.param_estimates
            self.klds += grid_point_results.klds
            self.obj_func += grid_point_results.obj_func
            self.d_time += grid_point_results.d_time

    def clean_up(self):
        os.remove(self.inference_string+'/*.gp')
        self.param_estimates = np.array(param_estimates)
        self.klds = np.array(klds)
        self.obj_func = np.array(obj_func)
        self.d_time = np.array(d_time)