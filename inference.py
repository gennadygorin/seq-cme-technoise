import pickle
import time
import numpy as np
import scipy
from scipy import optimize
from preprocess import *
from cme_toolbox import *
import multiprocessing
#lbfgsb has a deprecation warning for .tostring(), probably in FORTRAN interface
import warnings
import numdifftools
# from plot_aesthetics import *

#this should be in the model

#this should be in its own file
aesthetics = {'generic_gene_color':'dimgray',\
              'accepted_gene_color':'gold',\
              'rejected_gene_color':'darkgrey',\
              'generic_gene_alpha': 0.5,\
              'accepted_gene_alpha':0.7,\
              'rejected_gene_alpha':0.7,\
              'generic_gene_ms':10,\
              'accepted_gene_ms':5,\
              'rejected_gene_ms':5,\
              'hist_face_color':'lightgray',\
              'hist_fit_color':'firebrick',\
              'hist_fit_lw':4,\
              'hist_face_alpha':1,\
              'hist_face_color_1':'lightsteelblue',\
              'hist_fit_color_1':'royalblue',\
              'hist_fit_lw_1':4,
              'hist_face_alpha_1':0.5,\
              'hist_face_color_2':'lightcoral',\
              'hist_fit_color_2':'red',\
              'hist_fit_lw_2':4,\
              'hist_face_alpha_2':0.5,\
              'errorbar_gene_color': [203/255,197/255,149/255],\
              'errorbar_gene_alpha':0.3,\
              'errorbar_lw':2,\
              'length_fit_line_color':'firebrick',\
              'length_fit_face_color':'firebrick',\
              'length_fit_face_alpha':0.5,\
              'length_fit_lw':4,\
              'linestyle':'-'}

warnings.filterwarnings("ignore", category=DeprecationWarning) 


class InferenceParameters:
    def __init__(self,phys_lb,phys_ub,samp_lb,samp_ub,gridsize,dataset_string,model,use_lengths=True,\
    			 gradient_params = {'max_iterations':10,'init_pattern':'moments','num_restarts':1}):
        #this happens before parallelization
        self.gradient_params = gradient_params
        self.phys_lb = np.array(phys_lb)
        self.phys_ub = np.array(phys_ub)
        self.grad_bnd = scipy.optimize.Bounds(phys_lb,phys_ub)

        self.use_lengths = use_lengths

        if model.seq_model == 'None':
            log.info('Sequencing model set to None. All sampling parameters set to null.')
            samp_lb = [1,1]
            samp_ub = [1,1]
            gridsize = [1,1]

        self.samp_lb = np.array(samp_lb)
        self.samp_ub = np.array(samp_ub)
        self.gridsize = gridsize


        self.construct_grid()
        self.model = model

        self.n_phys_pars = len(self.phys_lb)
        self.n_samp_pars = len(self.samp_ub)

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

    def fit_all_grid_points(self,num_cores,search_data):
        t1 = time.time()
        if num_cores>1:
            log.info('Starting parallelized grid scan.')
            #add a progress bar here.
            pool=multiprocessing.Pool(processes=num_cores)
            #This might be improved.
            pool.map(self.par_fun,zip(range(self.n_grid_points),[[search_data,self.model]]*self.n_grid_points))
            pool.close()
            pool.join()
            log.info('Parallelized grid scan complete.')
        else:
            log.info('Starting non-parallelized grid scan.')
            for point_index in range(self.n_grid_points):
                grad_inference = GradientInference(self,self.model,search_data,point_index)
                grad_inference.fit_all_genes(self.model,search_data)
            log.info('Non-parallelized grid scan complete.')
        # full_result_string = self.store_search_results()
        results = SearchResults(self,search_data)
        results.aggregate_grid_points()
        full_result_string = results.store_on_disk()

        t2 = time.time()
        log.info('Runtime: {:.1f} seconds.'.format(t2-t1))
        return full_result_string

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

        self.n_phys_pars = len(global_parameters.phys_lb)
        self.n_samp_pars = len(global_parameters.samp_ub)


        self.inference_string = global_parameters.inference_string
        if self.gradient_params['init_pattern'] == 'moments':
            self.param_MoM = np.asarray([model.get_MoM(\
                                    search_data.moments[i],\
                                    global_parameters.phys_lb,\
                                    global_parameters.phys_ub,\
                                    regressor[i]) for i in range(search_data.n_genes)])
            
    def optimize_gene(self,gene_index,model,search_data):
        x0 = np.random.rand(self.gradient_params['num_restarts'],self.n_phys_pars)*(self.phys_ub-self.phys_lb)+self.phys_lb
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
    def __init__(self,inference_parameters,search_data):
        #pull in info from search parameters.
        self.sp = inference_parameters

        self.inference_string = inference_parameters.inference_string
        self.model = inference_parameters.model

        #pull in small amount of non-cell-specific from search data
        self.n_genes = search_data.n_genes
        self.n_cells = search_data.n_cells
        self.gene_log_lengths = search_data.gene_log_lengths
        self.gene_names = search_data.gene_names

        # self.X = inference_parameters.X
        # self.Y = inference_parameters.Y
        # self.sampl_vals = inference_parameters.sampl_vals
        # self.n_grid_points = inference_parameters.n_grid_points

        self.param_estimates = []
        self.klds = []
        self.obj_func = []
        self.d_time = []
        self.regressor = []

    #functions for aggregating grid points

    def aggregate_grid_points(self):
        for point_index in range(self.sp.n_grid_points):
            self.append_grid_point(point_index)
        self.clean_up()

    def append_grid_point(self, point_index):
        grid_point_result_string = self.inference_string + '/grid_point_'+str(point_index)+'.gp'
        with open(grid_point_result_string,'rb') as ipfs:
            grid_point_results = pickle.load(ipfs)
            self.param_estimates += [grid_point_results.param_estimates]
            self.klds += [grid_point_results.klds]
            self.obj_func += [grid_point_results.obj_func]
            self.d_time += [grid_point_results.d_time]
            self.regressor += [grid_point_results.regressor]

    def clean_up(self):
        for point_index in range(self.sp.n_grid_points):
            os.remove(self.inference_string + '/grid_point_'+str(point_index)+'.gp')
        log.info('All grid point data cleaned from disk.')
        self.param_estimates = np.asarray(self.param_estimates)
        self.klds = np.asarray(self.klds)
        self.obj_func = np.asarray(self.obj_func)
        self.d_time = np.asarray(self.d_time)
        self.regressor = np.asarray(self.regressor)

        analysis_figure_string = self.inference_string + '/analysis_figures'
        self.analysis_figure_string = analysis_figure_string
        make_dir(analysis_figure_string)

    def store_on_disk(self):
        try:
            full_result_string = self.inference_string + '/grid_scan_results.res'
            with open(full_result_string,'wb') as srfs:
                pickle.dump(self, srfs)
            log.debug('Grid scan results stored to {}.'.format(full_result_string))
        except:
            log.error('Grid scan results could not be stored to {}.'.format(full_result_string))
        self.full_result_string = full_result_string
        return full_result_string


    def update_on_disk(self):
        try:
            upd_result_string = self.inference_string + '/grid_scan_results_upd.res'
            with open(upd_result_string,'wb') as srfs:
                pickle.dump(self, srfs)
            log.debug('Updated results stored to {}.'.format(upd_result_string))
        except:
            log.error('Updated results could not be stored to {}.'.format(upd_result_string))
        self.upd_result_string = upd_result_string
        return upd_result_string

        #functions for analysis

    def find_sampling_optimum(self,gene_filter=None):
        if gene_filter is None:
            total_divergence = self.obj_func
        else:
            total_divergence = self.klds[:,gene_filter].sum(1)
        samp_optimum_ind = np.argmin(total_divergence)
        self.set_sampling_optimum(samp_optimum_ind)
        return self.samp_optimum

    def set_sampling_optimum(self,samp_optimum_ind):
        self.samp_optimum_ind = samp_optimum_ind
        self.samp_optimum = self.sp.sampl_vals[samp_optimum_ind]
        self.phys_optimum = self.param_estimates[samp_optimum_ind]
        self.regressor_optimum = self.regressor[samp_optimum_ind]
        return self.samp_optimum

    def plot_landscape(self, ax, plot_optimum = True, gene_filter=None, \
        logscale=True, colorbar=False,levels=40,hideticks=False,savefig = False):
        """
        Landscape visualization function. Plots landscape into axes.
        """

        if gene_filter is None:
            total_divergence = self.obj_func
        else:
            total_divergence = self.klds[:,gene_filter].sum(1)

        if logscale:
            total_divergence = np.log10(total_divergence)

        X = np.reshape(self.sp.X,self.sp.gridsize)
        Y = np.reshape(self.sp.Y,self.sp.gridsize)
        Z = np.reshape(total_divergence,self.sp.gridsize)
        contourplot = ax.contourf(X,Y,Z,levels)
        if plot_optimum:
            ax.scatter(self.samp_optimum[0],self.samp_optimum[1],c='r',s=50)
        if colorbar:
            plt.colorbar(contourplot)
        if hideticks:
            ax.set_xticks([])
            ax.set_yticks([])
        if savefig:
            fig_string = self.analysis_figure_string+'/landscape.png'
            plt.savefig(fig_string)
            log.info('Figure stored to {}.'.format(fig_string))

    def plot_param_marg(self,gene_filter=None,nbin=15,fitlaw=scipy.stats.norminvgauss,axis_search_bounds = True,\
                        discard_rejected=True):
        fig1,ax1=plt.subplots(nrows=1,ncols=3,figsize=(12,4))
        #this should be its own function
        if gene_filter is None:
            gene_filter = np.ones(self.phys_optimum.shape[0],dtype=bool)
        else:
            if gene_filter.dtype != np.bool:
                gf_temp = np.zeros(self.phys_optimum.shape[0],dtype=bool)
                gf_temp[gene_filter] = True
                gene_filter = gf_temp

        if discard_rejected:
            if hasattr(self,'rejection_index'):
                if self.rejection_index != self.samp_optimum_ind:
                    raise ValueError('Sampling parameter value is inconsistent.')
                gene_filter = np.logical_and(~self.rejected_genes,gene_filter)
            else:
                log.info('No rejection statistics have been computed: plotting all genes.')
            # gene_filter_rej = 
        param_data = self.phys_optimum[gene_filter,:]

        for i in range(3):
            ax1[i].hist(param_data[:,i],nbin,density=True,\
                        color=aesthetics['hist_face_color'])
            if fitlaw is not None:
                fitparams = fitlaw.fit(param_data[:,i])
                
                xmin, xmax = ax1[i].get_xlim()
                x = np.linspace(xmin, xmax, 100)
                p = fitlaw.pdf(x, *fitparams)
                ax1[i].plot(x, p, '--', \
                            linewidth=aesthetics['hist_fit_lw'],\
                            color=aesthetics['hist_fit_color'])
            
            if axis_search_bounds:
                ax1[i].set_xlim([self.sp.phys_lb[i],self.sp.phys_ub[i]])
            ax1[i].set_title(self.model.get_log_name_str()[i])
            ax1[i].set_xlabel(r'$log_{10}$ value')
        fig1.tight_layout()
        fig_string = self.analysis_figure_string+'/parameter_marginals.png'
        plt.savefig(fig_string)
        log.info('Figure stored to {}.'.format(fig_string))

    def plot_KL(self,ax,nbins=15):
        #maybe with filtering

        ax.hist(self.klds[self.samp_optimum_ind],nbins,\
            color=aesthetics['hist_face_color'])
        ax.set_xlabel('KL divergence')
        ax.set_ylabel('# genes')
        fig_string = self.analysis_figure_string+'/kldiv.png'
        plt.savefig(fig_string)
        log.info('Figure stored to {}.'.format(fig_string))

    def chisquare_testing(self,search_data,viz=False,EPS=1e-12,threshold=0.05,bonferroni=True):    
        """
        Performs chi-square testing on the dataset at the best-fit sampling parameter tuple. 

        Inputs:
        result_data: ResultData structure generated by importing SearchData files.
        viz: whether to visualize the histogram of the chi-square statistic.
        nosamp: whether to perform the testing using the sampling model or no-sampling model.
            accesses the fit parameters in result_data.gene_spec_samp_params.
        EPS: probability rounding parameter -- anything below this is rounded to EPS.
        threshold: chi-square rejection criterion; everything below this critical p-value (Bonferroni-corrected) is rejected as unlikely to have been generated by the model.
            In general, we expect about 5% of the genes to be rejected.
        
        Outputs:
        Tuple with chi-squared and p values. 
        """
        # samp = [None] * search_data.n_genes if (self.model.seq_model == 'None') else self.regressor_optimum[i_]
            # Pa = np.squeeze(model.eval_model_pss(search_results.phys_optimum[i_],lm,samp))
        # cs
        # expected_freq = [cme_integrator(result_data.best_phys_params[i_],\
        #     [result_data.M[i_],result_data.N[i_]],samp[i_]).flatten() for i_ in range(result_data.n_gen)]
        t1 = time.time()

        csqarr = []
        for gene_index in range(self.n_genes):
            samp = None if (self.model.seq_model == 'None') else self.regressor_optimum[gene_index]
            lm = [search_data.M[gene_index],search_data.N[gene_index]]  
            expected_freq = self.model.eval_model_pss(self.phys_optimum[gene_index],lm,samp).flatten()
            # temp = expected_freq[i_]
            # expected_freq[expected_freq<EPS]=EPS #no need
            csqarr += [scipy.stats.mstats.chisquare(search_data.hist[gene_index].flatten(),expected_freq)]
            # expected_freq[i_] = temp
        # csqarr = [scipy.stats.mstats.chisquare(result_data.hist[i_].flatten(), 
        #                                        expected_freq[i_]) for i_ in range(result_data.n_gen)]
        # csq,pval = np.array([csqarr[i_][0] for i_ in range(len(csqarr))])
        # pval = np.array([csqarr[i_][1] for i_ in range(len(csqarr))])
        csq,pval = zip(*csqarr)
        csq = np.asarray(csq)
        pval = np.asarray(pval)

        if bonferroni:
            threshold /= self.n_genes
        self.rejected_genes = pval<threshold
        self.pval = pval
        self.csq = csq
        self.rejection_index = self.samp_optimum_ind #mostly for debug.

        # result_data.set_rej(pval,threshold=threshold,nosamp=nosamp)

        if viz:
            plt.hist(csq)
            plt.xlabel('Chi-square statistic')
            plt.ylabel('# genes')
            fig_string = self.analysis_figure_string+'/chisquare.png'
            plt.savefig(fig_string)
            log.info('Figure stored to {}.'.format(fig_string))

        t2 = time.time()
        log.info('Chi-square computation complete. Rejected {:.0f} genes out of {:.0f}. Runtime: {:.1f} seconds.'.format(\
            np.sum(self.rejected_genes),self.n_genes,t2-t1))
        return (csq,pval)

    def compute_sigma(self,search_data):
        #this is a bit slow, and should probably be parallelized.
        log.info('Computing local Hessian.')
        t1 = time.time()

        # n_phys_pars = len(self.sp.phys_lb)
        hess = np.zeros((self.n_genes,self.sp.n_phys_pars,self.sp.n_phys_pars))
        for gene_index in range(self.n_genes):
            samp = None if (self.model.seq_model == 'None') else self.regressor_optimum[gene_index]
            lm = [search_data.M[gene_index],search_data.N[gene_index]]  
            Hfun = numdifftools.Hessian(lambda x: kl_div(
                search_data.hist[gene_index], self.model.eval_model_pss(x, lm, samp)))
            hess[gene_index,:,:] = Hfun(self.phys_optimum[gene_index])
        fail = np.zeros(self.n_genes,dtype=bool)
        sigma = np.zeros((self.n_genes,self.sp.n_phys_pars))

        for gene_index in range(self.n_genes):
            try:
                hess_inv = np.linalg.inv(hess[gene_index,:,:])
                sigma[gene_index,:] = np.sqrt(np.diag(hess_inv))/np.sqrt(self.n_cells)
            except:
                fail[gene_index]=True
                log.info('Gene {:.0f} ran into singularity; replaced with mean. (Search converged to local minimum?) '.format(gene_index))
                # errorbars[i,:] = np.mean(errorbars[:i,:])
            if np.any(~np.isfinite(sigma[gene_index,:])):
                fail[gene_index] =True
                log.info('Gene {:.0f} gives negative stdev; replaced with mean. (Search converged to local minimum?)'.format(gene_index))
                # errorbars[i,:] = np.mean(errorbars[:i,:])
        sigma[fail,:] = sigma[~fail,:].mean(0)
        self.sigma = sigma
        self.sigma_index = self.samp_optimum_ind #mostly for debug

        t2 = time.time()
        log.info('Standard error of the MLE computation complete. Runtime: {:.1f} seconds.'.format(t2-t1))


    def resample_opt_viz(self,resamp_vec=(5,10,20,40,60),Ntries=4,figsize=(10,10)):
        """
        Demonstration of the sensitivity of the sampling parameter landscape and optimum to the number of genes analyzed.

        Inputs:
        resamp_vec: vector of the number of genes to select (without replacement).
        Ntries: number of times to resample.
        figsize: tuple defining the figure dimensions.
        """
        Nsamp = len(resamp_vec)

        fig1,ax1=plt.subplots(nrows=Nsamp,ncols=Ntries,figsize=figsize)
        for samp_num in range(Nsamp):
            for i_ in range(Ntries):
                axloc = (samp_num,i_) 
                gene_filter = np.random.choice(self.n_genes,resamp_vec[samp_num],replace=False)

                # plot_landscape(ax1[axloc], result_data, gene_selection=gene_selection, levels=15, hideticks=True)

                subsampled_samp_optimum = self.find_sampling_optimum(gene_filter)
                self.plot_landscape(ax1[axloc], gene_filter=gene_filter,hideticks=True)
                # ax1[axloc].scatter(result_data.X[loc_best_ind],result_data.Y[loc_best_ind],s=10,c='r')
                
                if i_==0:
                    ax1[axloc].set_ylabel('n_genes = '+str(resamp_vec[samp_num]))
        #reset sampling optimum.

        fig_string = self.analysis_figure_string+'/subsampling.png'
        plt.savefig(fig_string)
        log.info('Figure stored to {}.'.format(fig_string))
        self.find_sampling_optimum()

    def resample_opt_mc_viz(self,resamp_vec=(5,10,20,40,60),Ntries=1000,figsize=(16,4)):
        """
        Extension of resample_opt_viz: demonstrates the sensitivity of the optimum upon choosing a subset of genes.
        The optimum is visualized on the parameter landscape generated from the entire gene set.

        Inputs:
        result_data: ResultData object.
        resamp_vec: vector of the number of genes to select (without replacement).
        Ntries: number of times to resample.
        figsize: tuple defining the figure dimensions.
        """    
        Nsamp = len(resamp_vec)
        
        fig1,ax1=plt.subplots(nrows=1,ncols=Nsamp,figsize=figsize)
        # for plot_index in range(N_):
        for samp_num in range(Nsamp):
            axloc = samp_num
            subsampled_samp_optimum_array = []
            for i__ in range(Ntries):
                gene_filter = np.random.choice(self.n_genes,resamp_vec[samp_num],replace=False) 
                subsampled_samp_optimum = self.find_sampling_optimum(gene_filter)
                subsampled_samp_optimum_array.append(subsampled_samp_optimum)       
            subsampled_samp_optimum_array = np.asarray(subsampled_samp_optimum_array)
        
            self.plot_landscape(ax1[axloc], levels=30,hideticks=True)
            # plot_landscape(ax1[samp_num], result_data, levels=30,hideticks=True)
            jit = np.random.normal(scale=0.1,size=subsampled_samp_optimum_array.shape)
            subsampled_samp_optimum_array=subsampled_samp_optimum_array+jit
            ax1[axloc].scatter(subsampled_samp_optimum_array[:,0],subsampled_samp_optimum_array[:,1],c='r',s=3,alpha=0.3)
            ax1[axloc].set_title('n_genes = '+str(resamp_vec[samp_num]))
        #reset sampling optimum.
        fig_string = self.analysis_figure_string+'/subsampling_stability.png'
        plt.savefig(fig_string)
        log.info('Figure stored to {}.'.format(fig_string))

        self.find_sampling_optimum()

    def chisq_best_param_correction(self,search_data,method='nearest',Ntries=10,viz=True,szfig=(2,5),figsize=(10,4),overwrite=True):
        if viz:
            fig1,ax1=plt.subplots(nrows=szfig[0],ncols=szfig[1],figsize=figsize)
        log.info('Original optimum: {:.2f}, {:.2f}.'.format(self.samp_optimum[0],self.samp_optimum[1]))
        for i_ in range(Ntries):
            self.chisquare_testing(search_data)
            gene_filter = ~self.rejected_genes
            well_fit_samp_optimum = self.find_sampling_optimum(gene_filter)
            log.info('New optimum: {:.2f}, {:.2f}.'.format(self.samp_optimum[0],self.samp_optimum[1]))

            if viz:
                axloc = np.unravel_index(i_,szfig) if (szfig[0]>1 and szfig[1]>1) else i_
                self.plot_landscape(ax1[axloc], gene_filter = gene_filter, levels=30,hideticks=True)
        if viz:
            fig_string = self.analysis_figure_string+'/chisquare_stability.png'
            plt.savefig(fig_string)
            log.info('Figure stored to {}.'.format(fig_string))
            
        if overwrite:
            self.chisquare_testing(search_data)
            log.info('Optimum retained at {:.2f}, {:.2f}.'.format(self.samp_optimum[0],self.samp_optimum[1]))
        else:
            self.find_sampling_optimum()
            self.chisquare_testing(search_data)
            log.info('Optimum restored to {:.2f}, {:.2f}.'.format(self.samp_optimum[0],self.samp_optimum[1]))

    def plot_param_L_dep(self,gene_filter = None,\
                         plot_errorbars=False,\
                         figsize=(12,4),c=2.576,\
                         axis_search_bounds = True, plot_fit = False,\
                         distinguish_rej = True):
        fig1,ax1=plt.subplots(nrows=1,ncols=3,figsize=figsize)

        if gene_filter is None:
            gene_filter = np.ones(self.phys_optimum.shape[0],dtype=bool)
            gene_filter_rej = np.zeros(self.phys_optimum.shape[0],dtype=bool)
        else:
            if gene_filter.dtype != np.bool:
                gf_temp = np.zeros(self.phys_optimum.shape[0],dtype=bool)
                gf_temp[gene_filter] = True
                gene_filter = gf_temp
                gene_filter_rej = np.zeros(self.phys_optimum.shape[0],dtype=bool) #something like this...

        if distinguish_rej: #default
            if hasattr(self,'rejected_genes'):
                if self.rejection_index != self.samp_optimum_ind:
                    log.warning('Sampling parameter value is inconsistent.')
                    distinguish_rej = False
                else: #if everything is ready
                    gene_filter_rej =  np.logical_and(gene_filter,self.rejected_genes)
                    gene_filter = np.logical_and(gene_filter,~self.rejected_genes) #note this updates gene_filter!
                    acc_point_aesth = ('accepted_gene_color','accepted_gene_alpha','accepted_gene_ms')
                    rej_point_aesth = ('rejected_gene_color','rejected_gene_alpha','rejected_gene_ms')
            else:
                log.info('Gene rejection needs to be precomputed to distinguish rejected points.')
                distinguish_rej = False

        if not distinguish_rej: #don't distinguish
            acc_point_aesth = ('generic_gene_color','generic_gene_alpha','generic_gene_ms')
            log.info('Falling back on generic marker properties.') 

        for i in range(3):
            if plot_errorbars:
                # raise ValueError('I still need to implement this.')

                lfun = lambda x,a,b: a*x+b
                if plot_fit:
                    popt,pcov = scipy.optimize.curve_fit(lfun,self.gene_log_lengths[gene_filter],
                                                         self.phys_optimum[gene_filter,i],\
                                                         sigma=self.sigma[gene_filter,i],
                                                         absolute_sigma=True)
                    xl = np.array([min(self.gene_log_lengths),max(self.gene_log_lengths)])

                    min_param = (popt[0]-np.sqrt(pcov[0,0])*c,popt[1]-np.sqrt(pcov[1,1])*c)
                    max_param = (popt[0]+np.sqrt(pcov[0,0])*c,popt[1]+np.sqrt(pcov[1,1])*c)
                    ax1[i].fill_between(xl,\
                        lfun(xl,min_param[0],min_param[1]),\
                        lfun(xl,max_param[0],max_param[1]),\
                        facecolor=aesthetics['length_fit_face_color'],\
                        alpha=aesthetics['length_fit_face_alpha'])
                    ax1[i].plot(xl,lfun(xl,popt[0],popt[1]),\
                        c=aesthetics['length_fit_line_color'],\
                        linewidth=aesthetics['length_fit_lw'])
                ax1[i].errorbar(self.gene_log_lengths[gene_filter],
                                self.phys_optimum[gene_filter,i],
                                self.sigma[gene_filter,i]*c,c=aesthetics['errorbar_gene_color'],
                                alpha=aesthetics['errorbar_gene_alpha'],linestyle='None',
                                linewidth = aesthetics['errorbar_lw'])

            ax1[i].scatter(self.gene_log_lengths[gene_filter],
                           self.phys_optimum[gene_filter,i],\
                           c=aesthetics[acc_point_aesth[0]],\
                           alpha=aesthetics[acc_point_aesth[1]],\
                           s=aesthetics[acc_point_aesth[2]])
            if np.any(gene_filter_rej):
                ax1[i].scatter(self.gene_log_lengths[gene_filter_rej],
                               self.phys_optimum[gene_filter_rej,i],\
                               c=aesthetics[rej_point_aesth[0]],\
                               alpha=aesthetics[rej_point_aesth[1]],\
                               s=aesthetics[rej_point_aesth[2]])

            ax1[i].set_xlabel(r'$\log_{10}$ L')
            ax1[i].set_ylabel(self.model.get_log_name_str()[i])
            if axis_search_bounds:
                ax1[i].set_ylim([self.sp.phys_lb[i],self.sp.phys_ub[i]])
        fig1.tight_layout()
        fig_string = self.analysis_figure_string+'/length_dependence.png'
        plt.savefig(fig_string)
        log.info('Figure stored to {}.'.format(fig_string))

    def plot_gene_distributions(self,search_data,sz = (5,5),figsize = (10,10),\
                   marg='joint',logscale=None,title=True,nosamp=False,\
                   genes_to_plot=None):
        
        if logscale is None:
            if marg=='joint':
                logscale = True
            else:
                logscale = False

        (nrows,ncols)=sz
        fig1,ax1=plt.subplots(nrows=nrows,ncols=ncols,figsize=figsize)

        nax = np.prod(sz)
        if genes_to_plot is None:
            genes_to_plot = np.arange(nax)
        
        genes_to_plot = np.asarray(genes_to_plot)
        if genes_to_plot.dtype == bool:
            gtp_temp = np.arange(self.n_genes)
            genes_to_plot = gtp_temp[genes_to_plot]
        
        number_of_genes_to_plot = len(genes_to_plot)
        if number_of_genes_to_plot > self.n_genes:
            number_of_genes_to_plot = self.n_genes
            genes_to_plot = genes_to_plot[:self.n_genes]
        if number_of_genes_to_plot>nax: #This should no longer break...
            number_of_genes_to_plot = nax
            genes_to_plot = genes_to_plot[:nax]
            


        j_=0
        for i_ in genes_to_plot:
            lm = [search_data.M[i_],search_data.N[i_]]
            if marg == 'mature':
                lm[0]=1
            if marg == 'nascent':
                lm[1]=1
            axloc = np.unravel_index(j_,sz) if (sz[0]>1 and sz[1]>1) else j_
            # axloc = np.unravel_index(i_,sz)
            
            samp = None if (self.model.seq_model == 'None') else self.regressor_optimum[i_]
            Pa = np.squeeze(self.model.eval_model_pss(self.phys_optimum[i_],lm,samp))


            if marg=='joint':
                if logscale:
                    Pa[Pa<1e-10]=1e-10
                    Pa = np.log10(Pa)

                X_,Y_ = np.meshgrid(np.arange(search_data.M[i_])-0.5,
                                    np.arange(search_data.N[i_])-0.5)
                ax1[axloc].contourf(X_.T,Y_.T,Pa,20,cmap='summer')
                
                jitter_magn = 0.1
                jitter_x = np.random.randn(self.n_cells)*jitter_magn
                jitter_y = np.random.randn(self.n_cells)*jitter_magn
                ax1[axloc].scatter(search_data.U[i_]+jitter_x,
                                    search_data.S[i_]+jitter_y,c='k',s=1,alpha=0.1)
                
                ax1[axloc].set_xlim([-0.5,search_data.M[i_]-1.5])
                ax1[axloc].set_ylim([-0.5,search_data.N[i_]-1.5])
            else:
                plot_hist_and_fit(ax1[axloc],search_data,i_,Pa,marg)
                if logscale:
                    ax1[axloc].set_yscale('log')            
            if title: #add a "rejected" thing
                ax1[axloc].set_title(self.gene_names[i_],fontdict={'fontsize': 9})
            ax1[axloc].set_xticks([])
            ax1[axloc].set_yticks([])
            j_+=1
        fig1.tight_layout(pad=0.02)

        fig_string = self.analysis_figure_string+'/gene_distributions_{}.png'.format(marg)
        plt.savefig(fig_string)
        log.info('Figure stored to {}.'.format(fig_string))

    def get_logL(self,search_data,EPS=1e-20):    
        logL = np.zeros(self.n_genes)
        for gene_index in range(self.n_genes):
            samp = None if (self.model.seq_model == 'None') else self.regressor_optimum[gene_index]
            offs = 20
            lm = [search_data.M[gene_index]+offs,search_data.N[gene_index]+offs]  
            Pss = self.model.eval_model_pss(self.phys_optimum[gene_index],lm,samp)
            if np.any(Pss<EPS):
                Pss[Pss<EPS] = EPS
            expected_log_lik = np.log(Pss)
            logL[gene_index] = expected_log_lik[search_data.U[gene_index].astype(int),search_data.S[gene_index].astype(int)].sum()
        return logL

    def get_logL_Poiss(self,search_data,EPS=1e-20):    
        logL = np.zeros(self.n_genes)
        for gene_index in range(self.n_genes):
            samp = None if (self.model.seq_model == 'None') else self.regressor_optimum[gene_index]
            offs = 0
            lm = [search_data.M[gene_index]+offs,search_data.N[gene_index]+offs]  
            x = np.arange(lm[0])
            y = np.arange(lm[1])
            m1 = search_data.moments[gene_index]['U_mean']
            m2 = search_data.moments[gene_index]['S_mean']
            expected_log_lik = (x * np.log(m1) - m1 - scipy.special.gammaln(x+1))[:,None] + \
                               (y * np.log(m2) - m2 - scipy.special.gammaln(y+1))[None,y]
            logL[gene_index] = expected_log_lik[search_data.U[gene_index].astype(int),search_data.S[gene_index].astype(int)].sum()
        return logL

def plot_hist_and_fit(ax1,sd,i_,Pa,marg='nascent',\
                      facecolor=aesthetics['hist_face_color'],\
                      fitcolor=aesthetics['hist_fit_color'],\
                      facealpha=aesthetics['hist_face_alpha'],\
                      linestyle=aesthetics['linestyle']):
    if marg=='nascent':
        ax1.hist(sd.U[i_],
                        bins=np.arange(sd.M[i_])-0.5,\
                        density=True,\
                        color=facecolor,alpha=facealpha)
        ax1.plot(np.arange(sd.M[i_]),Pa,\
                        color=fitcolor,linestyle=linestyle)
        ax1.set_xlim([-0.5,sd.U[i_].max()+2.5])
    elif marg =='mature':
        ax1.hist(sd.S[i_],
                        bins=np.arange(sd.N[i_])-0.5,\
                        density=True,\
                        color=facecolor,alpha=facealpha)
        ax1.plot(np.arange(sd.N[i_]),Pa,\
                        color=fitcolor,linestyle=linestyle)
        ax1.set_xlim([-0.5,sd.S[i_].max()+2.5])
