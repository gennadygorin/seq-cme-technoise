# from preprocess import *
from cme_toolbox import *
from inference import *
import pickle


########################
## Helper functions
########################

import pickle
def load_search_results(full_result_string):
    try:
        with open(full_result_string,'rb') as srfs:
            search_results = pickle.load(srfs)
        log.debug('Grid scan results loaded from {}.'.format(full_result_string))
        return search_results
    except:
        log.error('Grid scan results could not be loaded from {}.'.format(full_result_string))

def load_search_data(search_data_string):
    try:
        with open(search_data_string,'rb') as sdfs:
            sd = pickle.load(sdfs)
        log.info('Search data loaded from {}.'.format(search_data_string))
        return sd
    except:
        log.error('Search data could not be loaded from {}.'.format(search_data_string))


def plot_params_for_pair(sr1,sr2,dir_string,gene_filter = None,\
                     plot_errorbars=False,\
                     figsize=None,c=2.576,\
                     axis_search_bounds = True,
                     distinguish_rej = True,
                     plot_identity = True,
                     meta = '12',
                     xlabel = 'dataset 1',
                     ylabel = 'dataset 2'):
    
    num_params = sr1.model.get_num_params()
    if figsize is None:
        figsize = (4*num_params,4)
    fig1,ax1=plt.subplots(nrows=1,ncols=num_params,figsize=figsize)

    if gene_filter is None:
        gene_filter = np.ones(sr1.phys_optimum.shape[0],dtype=bool)
        gene_filter_rej = ~gene_filter
    else:
        if gene_filter.dtype != np.bool:
            gf_temp = np.zeros(sr1.phys_optimum.shape[0],dtype=bool)
            gf_temp[gene_filter] = True
            gene_filter = gf_temp
            gene_filter_rej = np.zeros(sr1.phys_optimum.shape[0],dtype=bool) #something like this...
        else:
            gene_filter = np.copy(gene_filter)
            gene_filter_rej = np.zeros(gene_filter.shape,dtype=bool)

    if distinguish_rej: #default
        if hasattr(sr1,'rejected_genes') and hasattr(sr2,'rejected_genes'):
            if sr1.rejection_index != sr1.samp_optimum_ind:
                log.warning('Sampling parameter value is inconsistent.')
                distinguish_rej = False
            elif sr2.rejection_index != sr2.samp_optimum_ind:
                log.warning('Sampling parameter value is inconsistent.')
                distinguish_rej = False
            else: #if everything is ready
                gene_filter_rej =  np.logical_and(gene_filter,np.logical_or(sr1.rejected_genes,sr2.rejected_genes))
                gene_filter = np.logical_and(gene_filter,~sr1.rejected_genes,~sr2.rejected_genes)
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

    analysis_dir_string = dir_string + '/analysis_figures'
    make_dir(analysis_dir_string)
    sr1.analysis_dir_string = analysis_dir_string
    sr2.analysis_dir_string = analysis_dir_string
    
    fig_string = analysis_dir_string+'/pair_parameter_comparison_{}.png'.format(meta)
    plt.savefig(fig_string)
    log.info('Figure stored to {}.'.format(fig_string))

def find_most_concordant_samp(sr1,sr2):
    discordance = ((sr1.param_estimates - sr2.param_estimates)**2).sum(2).sum(1)
    srt =  np.argsort(discordance)
    samp_concordant_ind = srt[0]
    sr1.set_sampling_optimum(samp_concordant_ind)
    sr2.set_sampling_optimum(samp_concordant_ind)

    log.info('Optimum set to at {:.2f}, {:.2f}.'.format(sr1.samp_optimum[0],sr1.samp_optimum[1]))
    return sr1.samp_optimum

def get_AIC_weights(sr_arr,sd):
    n_models = len(sr_arr)
    AIC = []
    for j in range(n_models):
        AIC += [2*sr_arr[j].model.get_num_params()-2*sr_arr[j].get_logL(sd)] 
    AIC = np.asarray(AIC)
    min_AIC = AIC.min(0)
    normalization = np.exp(-(AIC - min_AIC)/2).sum(0)
    w = np.exp(-(AIC - min_AIC)/2) / normalization
    return w

def plot_AIC_weights(w,models,figsize=None,                      
                      facecolor=aesthetics['hist_face_color'],\
                      facealpha=aesthetics['hist_face_alpha'],):
    n_models = w.shape[0]
    if figsize is None:
        figsize = (4*n_models,4)
    fig1,ax1=plt.subplots(nrows=1,ncols=n_models,figsize=figsize)
    for i in range(n_models):
        ax1[i].hist(w[i],bins=30,\
                        density=False,\
                        color=facecolor,alpha=facealpha)
        ax1[i].set_xlabel('AIC weight at MLE')
        ax1[i].set_ylabel('# genes')
        ax1[i].set_title(models[i])
