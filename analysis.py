# from preprocess import *
from cme_toolbox import *
from inference import *
import pickle


########################
## Helper functions
########################

import pickle
def load_search_results(full_result_string):
    """
    This function attempts to load in search results.

    Input: 
    full_result_string: location of the SearchResults object.
    
    Output: 
    search_results: a SearchResults object.
    """
    try:
        with open(full_result_string,'rb') as srfs:
            search_results = pickle.load(srfs)
        log.info('Grid scan results loaded from {}.'.format(full_result_string))
        return search_results
    except:
        log.error('Grid scan results could not be loaded from {}.'.format(full_result_string))

def load_search_data(search_data_string):
    """
    This function attempts to load in search data.

    Input: 
    full_result_string: location of the SearchData object.
    
    Output: 
    sd: a SearchData object.
    """
    try:
        with open(search_data_string,'rb') as sdfs:
            sd = pickle.load(sdfs)
        log.info('Search data loaded from {}.'.format(search_data_string))
        return sd
    except:
        log.error('Search data could not be loaded from {}.'.format(search_data_string))


def make_batch_analysis_dir(sr_arr,dir_string):
    """
    This function creates a directory for batch analysis.

    Input: 
    sr_arr: list of multiple SearchResults objects.
    dir_string: batch directory 
    
    Output: 
    sd: a SearchData object.
    """
    batch_analysis_string = dir_string + '/analysis_figures'
    make_dir(batch_analysis_string)
    for sr in sr_arr:
        sr.batch_analysis_string = batch_analysis_string

def plot_params_for_pair(sr1,sr2,gene_filter_ = None,\
                     plot_errorbars=False,\
                     figsize=None,c=2.576,\
                     axis_search_bounds = True,
                     distinguish_rej = True,
                     plot_identity = True,
                     meta = '12',
                     xlabel = 'dataset 1',
                     ylabel = 'dataset 2'):
    """
    This method plots the inferred physical parameters at the sampling parameter optimum for a matched pair of datasets.

    Input:
    sr1: SearchResult instance 1.
    sr2: SearchResult instance 2.
    gene_filter_: 
        If None, plot all genes. 
        If a boolean or integer filter, plot only a subset of genes indicated by the filter.
    plot_errorbars: whether to use inferred standard error of MLEs to plot error bars.
    figsize: figure dimensions.
    c: error bar scaling factor. c=2.576 corresponds to a rough 99% CI.
    axis_search_bounds: whether to place the x-limits of the plots at the parameter search bounds.
    distinguish_rej: whether to distinguish genes in the rejected_genes attribute.
    metadata: figure name metadata.
    xlabel: name of dataset 1.
    ylabel: name of dataset 2.
    """
    num_params = sr1.sp.n_phys_pars
    if figsize is None:
        figsize = (4*num_params,4)
    fig1,ax1=plt.subplots(nrows=1,ncols=num_params,figsize=figsize)


    gene_filter = sr1.get_bool_filt(gene_filter_,discard_rejected=False)
    # gene_filter2 = sr2.get_bool_filt(gene_filter_,discard_rejected=False)
    gene_filter_rej = np.zeros(sr1.n_genes,dtype=bool)
    # gene_filter_rej2 = np.zeros(sr2.n_genes,dtype=bool)

    if distinguish_rej: #default
        filt_rej1 = sr1.get_bool_filt(gene_filter_,discard_rejected=True)
        filt_rej2 = sr2.get_bool_filt(gene_filter_,discard_rejected=True)
        filt_rej = np.logical_and(filt_rej1, filt_rej2)

        gene_filter_rej = np.logical_and(gene_filter,np.logical_not(filt_rej))
        gene_filter = np.logical_and(gene_filter,filt_rej)
        acc_point_aesth = ('accepted_gene_color','accepted_gene_alpha','accepted_gene_ms')
        rej_point_aesth = ('rejected_gene_color','rejected_gene_alpha','rejected_gene_ms')
    else: #don't distinguish
        acc_point_aesth = ('generic_gene_color','generic_gene_alpha','generic_gene_ms')
        log.info('Falling back on generic marker properties.') 


    # if gene_filter is None:
    #     gene_filter = np.ones(sr1.phys_optimum.shape[0],dtype=bool)
    #     gene_filter_rej = ~gene_filter
    # else:
    #     if gene_filter.dtype != np.bool:
    #         gf_temp = np.zeros(sr1.phys_optimum.shape[0],dtype=bool)
    #         gf_temp[gene_filter] = True
    #         gene_filter = gf_temp
    #         gene_filter_rej = np.zeros(sr1.phys_optimum.shape[0],dtype=bool) #something like this...
    #     else:
    #         gene_filter = np.copy(gene_filter)
    #         gene_filter_rej = np.zeros(gene_filter.shape,dtype=bool)

    # if distinguish_rej: #default
    #     if hasattr(sr1,'rejected_genes') and hasattr(sr2,'rejected_genes'):
    #         if sr1.rejection_index != sr1.samp_optimum_ind:
    #             log.warning('Sampling parameter value is inconsistent.')
    #             distinguish_rej = False
    #         elif sr2.rejection_index != sr2.samp_optimum_ind:
    #             log.warning('Sampling parameter value is inconsistent.')
    #             distinguish_rej = False
    #         else: #if everything is ready
    #             gene_filter_rej =  np.logical_and(gene_filter,np.logical_or(sr1.rejected_genes,sr2.rejected_genes))
    #             gene_filter = np.logical_and(gene_filter,~sr1.rejected_genes,~sr2.rejected_genes)
    #             acc_point_aesth = ('accepted_gene_color','accepted_gene_alpha','accepted_gene_ms')
    #             rej_point_aesth = ('rejected_gene_color','rejected_gene_alpha','rejected_gene_ms')
    #     else:
    #         log.info('Gene rejection needs to be precomputed to distinguish rejected points.')
    #         distinguish_rej = False

    # if not distinguish_rej: #don't distinguish
    #     acc_point_aesth = ('generic_gene_color','generic_gene_alpha','generic_gene_ms')
    #     log.info('Falling back on generic marker properties.') 

    for i in range(3):
        if plot_errorbars:
            ax1[i].errorbar(sr1.phys_optimum[gene_filter,i],\
                            sr2.phys_optimum[gene_filter,i],\
                            sr1.sigma[gene_filter,i]*c,\
                            sr2.sigma[gene_filter,i]*c,\
                            c=aesthetics['errorbar_gene_color'],
                            alpha=aesthetics['errorbar_gene_alpha'],\
                            linestyle='None',
                            linewidth = aesthetics['errorbar_lw'])
        ax1[i].scatter(sr1.phys_optimum[gene_filter,i],
                       sr2.phys_optimum[gene_filter,i],\
                       c=aesthetics[acc_point_aesth[0]],\
                       alpha=aesthetics[acc_point_aesth[1]],\
                       s=aesthetics[acc_point_aesth[2]])
        if np.any(gene_filter_rej):
            ax1[i].scatter(sr1.phys_optimum[gene_filter_rej,i],
                           sr2.phys_optimum[gene_filter_rej,i],\
               c=aesthetics[rej_point_aesth[0]],\
               alpha=aesthetics[rej_point_aesth[1]],\
               s=aesthetics[rej_point_aesth[2]])

        ax1[i].set_xlabel(xlabel)
        ax1[i].set_ylabel(ylabel)
        ax1[i].set_title(sr1.model.get_log_name_str()[i])
        if axis_search_bounds:
            ax1[i].set_xlim([sr1.sp.phys_lb[i],sr1.sp.phys_ub[i]])
            ax1[i].set_ylim([sr1.sp.phys_lb[i],sr1.sp.phys_ub[i]])
        if plot_identity:
            xl = ax1[i].get_xlim()
            ax1[i].plot(xl,xl,'r--',linewidth=2)
    fig1.tight_layout()

    
    fig_string = sr1.batch_analysis_string+'/pair_parameter_comparison_{}.png'.format(meta)
    plt.savefig(fig_string)
    log.info('Figure stored to {}.'.format(fig_string))

def find_most_concordant_samp(sr1,sr2):
    """
    This method attempts to find a search parameter optimum by comparing two matched (control) datasets
    and finding the point at which their parameter values are most concordant, according to l2 distance over
    all genes and parameters.
    This typically works poorly.

    Input:
    sr1: SearchResult instance 1.
    sr2: SearchResult instance 2.

    Output:
    sampling parameter optimum value.
    """
    discordance = ((sr1.param_estimates - sr2.param_estimates)**2).sum(2).sum(1)
    srt =  np.argsort(discordance)
    samp_concordant_ind = srt[0]
    sr1.set_sampling_optimum(samp_concordant_ind)
    sr2.set_sampling_optimum(samp_concordant_ind)

    log.info('Optimum set to at {:.2f}, {:.2f}.'.format(sr1.samp_optimum[0],sr1.samp_optimum[1]))
    return sr1.samp_optimum

def get_AIC_weights(sr_arr,sd):
    """
    This method computes the Akaike Information Criterion weights according to the optimal
    physical and sampling parameters obtained for a single dataset (sd) under several models (results
    stored in (sr_arr).

    Input:
    sr_arr: list of multiple SearchResults objects.
    sd: SearchData instance.

    Output:
    w: AIC weights corresponding to each model, a n_models x n_genes array.
    """

    n_models = len(sr_arr)
    AIC = []
    for j in range(n_models):
        AIC += [2*sr_arr[j].sp.n_phys_pars-2*sr_arr[j].get_logL(sd)] 
    AIC = np.asarray(AIC)
    min_AIC = AIC.min(0)
    normalization = np.exp(-(AIC - min_AIC)/2).sum(0)
    w = np.exp(-(AIC - min_AIC)/2) / normalization
    return w

def plot_AIC_weights(sr_arr,sd,models,ax1=None,meta=None,figsize=None,                      
                      facecolor=aesthetics['hist_face_color'],\
                      facealpha=aesthetics['hist_face_alpha'],nbin=20,savefig=False):
    """
    This method calls get_AIC_weights and plots the resulting Akaike Information Criterion weights.

    Input:
    sr_arr: list of multiple SearchResults objects.
    sd: SearchData instance.
    models: model names.
    ax1: matplotlib axes to plot into.
    meta: figure metadata.
    figsize: figure dimensions.
    facecolor: histogram face color.
    facealpha: histogram face alpha.
    nbin: number of histogram bins.

    Output:
    w: AIC weights corresponding to each model, a n_models x n_genes array.
    """
    w=get_AIC_weights(sr_arr,sd)

    if meta is None:
        meta = ''
    else:
        meta = '_'+meta    

    n_models = w.shape[0]
    if figsize is None:
        figsize = (4*n_models,4)

    if ax1 is None:
        fig1,ax1=plt.subplots(nrows=1,ncols=n_models,figsize=figsize)

    for i in range(n_models):
        ax1[i].hist(w[i],bins=nbin,\
                        density=False,\
                        color=facecolor,alpha=facealpha)
        ax1[i].set_xlabel('AIC weight at MLE')
        ax1[i].set_ylabel('# genes')
        ax1[i].set_title(models[i])

    if savefig:
        fig_string = batch_analysis_string+'/AIC_comparison{}.png'.format(meta)

        plt.savefig(fig_string)
        log.info('Figure stored to {}.'.format(fig_string))
    return w