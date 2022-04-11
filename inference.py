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
              'errorbar_gene_color': [203/255,197/255,149/255],\
              'errorbar_gene_alpha':0.3,\
              'errorbar_lw':2,\
              'length_fit_line_color':'firebrick',\
              'length_fit_face_color':'firebrick',\
              'length_fit_face_alpha':0.5,\
              'length_fit_lw':4}

warnings.filterwarnings("ignore", category=DeprecationWarning) 


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
        self.model = model

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
        t1 = time.time()
        if num_cores>1:
            log.info('Starting parallelized grid scan.')
            #add a progress bar here.
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
        full_result_string = self.store_search_results()
        t2 = time.time()
        log.info('Runtime: {:.1f} seconds.'.format(t2-t1))
        return full_result_string

    def store_search_results(self):
        results = SearchResults(self)
        results.aggregate_grid_points()
        try:
            full_result_string = self.inference_string + '/grid_scan_results.res'
            with open(full_result_string,'wb') as srfs:
                pickle.dump(results, srfs)
            log.debug('Grid scan results stored to {}.'.format(full_result_string))
        except:
            log.error('Grid scan results could not be stored to {}.'.format(full_result_string))
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
        #load in search parameter data. fwiw it's easier to just have an inference_parameters object as an attribute.
        self.inference_string = inference_parameters.inference_string
        self.phys_lb = inference_parameters.phys_lb
        self.phys_ub = inference_parameters.phys_ub
        self.use_lengths = inference_parameters.use_lengths
        self.samp_lb = inference_parameters.samp_lb
        self.samp_ub = inference_parameters.samp_ub
        self.gridsize = inference_parameters.gridsize
        self.model = inference_parameters.model

        self.X = inference_parameters.X
        self.Y = inference_parameters.Y
        self.sampl_vals = inference_parameters.sampl_vals
        self.n_grid_points = inference_parameters.n_grid_points

        self.param_estimates = []
        self.klds = []
        self.obj_func = []
        self.d_time = []
        self.regressor = []

    #functions for aggregating grid points

    def aggregate_grid_points(self):
        for point_index in range(self.n_grid_points):
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
        for point_index in range(self.n_grid_points):
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
        self.samp_optimum = self.sampl_vals[samp_optimum_ind]
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

        X = np.reshape(self.X,self.gridsize)
        Y = np.reshape(self.Y,self.gridsize)
        Z = np.reshape(total_divergence,self.gridsize)
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
                ax1[i].set_xlim([self.phys_lb[i],self.phys_ub[i]])
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
        for gene_index in range(search_data.n_genes):
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
            threshold /= search_data.n_genes
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
            np.sum(self.rejected_genes),search_data.n_genes,t2-t1))
        return (csq,pval)

    def compute_sigma(self,search_data):
        log.info('Computing local Hessian.')
        t1 = time.time()

        n_phys_pars = len(self.phys_lb)
        hess = np.zeros((search_data.n_genes,n_phys_pars,n_phys_pars))
        for gene_index in range(search_data.n_genes):
            samp = None if (self.model.seq_model == 'None') else self.regressor_optimum[gene_index]
            lm = [search_data.M[gene_index],search_data.N[gene_index]]  
            Hfun = numdifftools.Hessian(lambda x: kl_div(
                search_data.hist[gene_index], self.model.eval_model_pss(x, lm, samp)))
            hess[gene_index,:,:] = Hfun(self.phys_optimum[gene_index])
        fail = np.zeros(search_data.n_genes,dtype=bool)
        sigma = np.zeros((search_data.n_genes,n_phys_pars))

        for gene_index in range(search_data.n_genes):
            try:
                hess_inv = np.linalg.inv(hess[gene_index,:,:])
                sigma[gene_index,:] = np.sqrt(np.diag(hess_inv))/np.sqrt(search_data.n_cells)
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


    def resample_opt_viz(self,search_data,resamp_vec=(5,10,20,40,60),Ntries=4,figsize=(10,10)):
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
                gene_filter = np.random.choice(search_data.n_genes,resamp_vec[samp_num],replace=False)

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

    def resample_opt_mc_viz(self,search_data,resamp_vec=(5,10,20,40,60),Ntries=1000,figsize=(16,4)):
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
                gene_filter = np.random.choice(search_data.n_genes,resamp_vec[samp_num],replace=False) 
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
                axloc = np.unravel_index(i_,szfig)
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