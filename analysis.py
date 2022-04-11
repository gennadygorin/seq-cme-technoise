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

def plot_param_L_dep(search_results,search_data,gene_filter = None,\
                     plot_errorbars=False,\
                     figsize=(12,4),c=2.576,\
                     axis_search_bounds = True, plot_fit = False,\
                     distinguish_rej = True):
    fig1,ax1=plt.subplots(nrows=1,ncols=3,figsize=figsize)

    if gene_filter is None:
        gene_filter = np.ones(search_results.phys_optimum.shape[0],dtype=bool)
        gene_filter_rej = ~gene_filter
    else:
        if gene_filter.dtype != np.bool:
            gf_temp = np.zeros(search_results.phys_optimum.shape[0],dtype=bool)
            gf_temp[gene_filter] = True
            gene_filter = gf_temp
            gene_filter_rej = np.zeros(search_results.phys_optimum.shape[0],dtype=bool) #something like this...

    if distinguish_rej: #default
        if hasattr(search_results,'rejected_genes'):
            if search_results.rejection_index != search_results.samp_optimum_ind:
                log.warning('Sampling parameter value is inconsistent.')
                distinguish_rej = False
            else: #if everything is ready
                gene_filter = np.logical_and(gene_filter,~search_results.rejected_genes)
                gene_filter_rej =  np.logical_and(gene_filter,search_results.rejected_genes)
                acc_point_aesth = ('accepted_gene_color','accepted_gene_alpha','accepted_gene_ms')
                rej_point_aesth = ('rejected_gene_color','rejected_gene_alpha','rejected_gene_ms')
        else:
            log.info('Gene rejection needs to be precomputed to distinguish rejected points.')
            distinguish_rej = False

    if not distinguish_rej: #don't distinguish
        acc_point_aesth = ('generic_gene_color','generic_gene_alpha','generic_gene_ms')
        # gene_filter = np.ones(search_results.phys_optimum.shape[0],dtype=bool)
        log.info('Falling back on generic marker properties.') 
        print()   


    for i in range(3):
        if plot_errorbars:
            # raise ValueError('I still need to implement this.')

            lfun = lambda x,a,b: a*x+b
            if plot_fit:
	            popt,pcov = scipy.optimize.curve_fit(lfun,search_data.gene_log_lengths[gene_filter],
	                                                 search_results.phys_optimum[gene_filter,i],\
                                                     sigma=search_results.sigma[gene_filter,i],
	                                                 absolute_sigma=True)
	            xl = np.array([min(search_data.gene_log_lengths),max(search_data.gene_log_lengths)])

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
            ax1[i].errorbar(search_data.gene_log_lengths[gene_filter],
                            search_results.phys_optimum[gene_filter,i],
                            search_results.sigma[gene_filter,i]*c,c=aesthetics['errorbar_gene_color'],
                            alpha=aesthetics['errorbar_gene_alpha'],linestyle='None',
                            linewidth = aesthetics['errorbar_lw'])

        ax1[i].scatter(search_data.gene_log_lengths[gene_filter],
                        search_results.phys_optimum[gene_filter,i],\
                       c=aesthetics[acc_point_aesth[0]],\
                       alpha=aesthetics[acc_point_aesth[1]],\
                       s=aesthetics[acc_point_aesth[2]])
        if np.any(gene_filter_rej):
            #this isn't strictly correct: it will plot filtered-out genes too...
            ax1[i].scatter(search_data.gene_log_lengths[gene_filter_rej],
                search_results.phys_optimum[gene_filter_rej,i],\
               c=aesthetics[rej_point_aesth[0]],\
               alpha=aesthetics[rej_point_aesth[1]],\
               s=aesthetics[rej_point_aesth[2]])

        ax1[i].set_xlabel(r'$\log_{10}$ L')
        ax1[i].set_ylabel(search_results.model.get_log_name_str()[i])
        if axis_search_bounds:
            ax1[i].set_ylim([search_results.phys_lb[i],search_results.phys_ub[i]])
    fig1.tight_layout()
    fig_string = search_results.analysis_figure_string+'/length_dependence.png'
    plt.savefig(fig_string)
    log.info('Figure stored to {}.'.format(fig_string))

def plot_gene_distributions(search_results,search_data,sz = (5,5),figsize = (10,10),\
               marg='joint',logscale=None,title=True,nosamp=False,\
               number_of_genes_to_plot=None):
    
    if logscale is None:
        if marg=='joint':
            logscale = True
        else:
            logscale = False

    (nrows,ncols)=sz
    fig1,ax1=plt.subplots(nrows=nrows,ncols=ncols,figsize=figsize)
    if number_of_genes_to_plot is None:
        number_of_genes_to_plot = np.prod(sz)
    if number_of_genes_to_plot > search_data.n_genes:
        number_of_genes_to_plot = search_data.n_genes

    for i_ in range(number_of_genes_to_plot):
        lm = [search_data.M[i_],search_data.N[i_]]
        if marg == 'mature':
            lm[0]=1
        if marg == 'nascent':
            lm[1]=1
        axloc = np.unravel_index(i_,sz)
        
        samp = None if (search_results.model.seq_model == 'None') else search_results.regressor_optimum[i_]
        Pa = np.squeeze(search_results.model.eval_model_pss(search_results.phys_optimum[i_],lm,samp))

        if title: #add a "rejected" thing
            ax1[axloc].set_title(search_data.gene_names[i_],fontdict={'fontsize': 9})
        ax1[axloc].set_xticks([])
        ax1[axloc].set_yticks([])
        if marg=='joint':
            if logscale:
                Pa[Pa<1e-10]=1e-10
                Pa = np.log10(Pa)

            X_,Y_ = np.meshgrid(np.arange(search_data.M[i_])-0.5,
                                np.arange(search_data.N[i_])-0.5)
            ax1[axloc].contourf(X_.T,Y_.T,Pa,20,cmap='summer')
            
            jitter_magn = 0.1
            jitter_x = np.random.randn(search_data.n_cells)*jitter_magn
            jitter_y = np.random.randn(search_data.n_cells)*jitter_magn
            ax1[axloc].scatter(search_data.U[i_]+jitter_x,
                                search_data.S[i_]+jitter_y,c='k',s=1,alpha=0.1)
            
            ax1[axloc].set_xlim([-0.5,search_data.M[i_]-1.5])
            ax1[axloc].set_ylim([-0.5,search_data.N[i_]-1.5])
        if marg=='nascent':
            ax1[axloc].hist(search_data.U[i_],
                            bins=np.arange(search_data.M[i_])-0.5,\
                            density=True,log=log,\
                            color=aesthetics['hist_face_color'])
            ax1[axloc].plot(np.arange(search_data.M[i_]),Pa,\
                            color=aesthetics['hist_fit_color'])
            ax1[axloc].set_xlim([-0.5,search_data.M[i_]-1.5])
            if logscale:
                ax1[axloc].set_yscale('log')
        if marg=='mature':
            ax1[axloc].hist(search_data.S[i_],
                            bins=np.arange(search_data.N[i_])-0.5,\
                            density=True,log=log,\
                            color=aesthetics['hist_face_color'])
            ax1[axloc].plot(np.arange(search_data.N[i_]),Pa,\
                            color=aesthetics['hist_fit_color'])
            ax1[axloc].set_xlim([-0.5,search_data.N[i_]-1.5])
            if logscale:
                ax1[axloc].set_yscale('log')
    fig1.tight_layout(pad=0.02)

    fig_string = search_results.analysis_figure_string+'/gene_distributions_{}.png'.format(marg)
    plt.savefig(fig_string)
    log.info('Figure stored to {}.'.format(fig_string))


def plot_params_for_pair(sr1,sd1,sr2,analysis_dir_string,gene_filter = None,\
                     plot_errorbars=False,\
                     figsize=(12,4),c=2.576,\
                     axis_search_bounds = True,
                     distinguish_rej = True,
                     plot_identity = True):
    fig1,ax1=plt.subplots(nrows=1,ncols=3,figsize=figsize)

    if gene_filter is None:
        gene_filter = np.ones(sr1.phys_optimum.shape[0],dtype=bool)
        gene_filter_rej = ~gene_filter
    else:
        if gene_filter.dtype != np.bool:
            gf_temp = np.zeros(sr1.phys_optimum.shape[0],dtype=bool)
            gf_temp[gene_filter] = True
            gene_filter = gf_temp
            gene_filter_rej = np.zeros(search_results.phys_optimum.shape[0],dtype=bool) #something like this...

    if distinguish_rej: #default
        if hasattr(sr1,'rejected_genes') and hasattr(sr2,'rejected_genes'):
            if sr1.rejection_index != sr1.samp_optimum_ind:
                log.warning('Sampling parameter value is inconsistent.')
                distinguish_rej = False
            elif sr2.rejection_index != sr2.samp_optimum_ind:
                log.warning('Sampling parameter value is inconsistent.')
                distinguish_rej = False
            else: #if everything is ready
                gene_filter = np.logical_and(gene_filter,~sr1.rejected_genes,~sr2.rejected_genes)
                gene_filter_rej =  np.logical_and(gene_filter,np.logical_or(sr1.rejected_genes,sr2.rejected_genes))
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

        ax1[i].set_xlabel(r'dataset 1')
        ax1[i].set_xlabel(r'dataset 2')
        ax1[i].set_title(sr1.model.get_log_name_str()[i])
        if axis_search_bounds:
            ax1[i].set_xlim([sr1.phys_lb[i],sr1.phys_ub[i]])
            ax1[i].set_ylim([sr1.phys_lb[i],sr1.phys_ub[i]])
        if plot_identity:
            xl = ax1[i].get_xlim()
            ax1[i].plot(xl,xl,'r--',linewidth=2)
    fig1.tight_layout()

    fig_string = analysis_dir_string+'/pair_parameter_comparison.png'
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