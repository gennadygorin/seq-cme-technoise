
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt

import time
import loompy as lp
import os
from datetime import datetime
	


code_ver_global='020'

########################
## Initialization
########################

def create_dir(batch_location = '.',meta='batch',batch_id=1,\
               datestring=datetime.now().strftime("%y%m%d"),\
               creator='gg',code_ver=code_ver_global):
    dir_string = '_'.join((creator, datestring, code_ver, meta, str(batch_id)))
    dir_string = batch_location + '/' + dir_string 
    try: 
        os.mkdir(dir_string) 
        with open(dir_string+'/.gitignore', 'w') as gitignore: 
            pass
        print('Directory ' + dir_string+ ' created; gitignore written.')
    except OSError as error: 
        print('Directory ' + dir_string+ ' exists.')

def get_transcriptome(transcriptome_filepath,repeat_thr=15):
    """
    Imports transcriptome length/repeat from a previously generated file. Input:
    transcriptome_filepath: path to the file. This is a simple space-separated file.
        The convention for each line is name - length - 5mers - 6mers -.... 50mers - more
    repeat_thr: threshold for minimum repeat length to consider. 
        By default, this is 15, and so will return number of polyA stretches of 
        length 15 or more in the gene.

    Returns two dictionaries:
    len_dict: Maps gene name to gene length, in bp.
    repeat_dict: Maps gene name to number of repeats.

    repeat_dict is not used, but is supported in the current version of the code.
    """
    repeat_dict = {}
    len_dict = {}

    thr_ind = repeat_thr-3
    with open(transcriptome_filepath,'r') as file:   
        for line in file.readlines():
            d = [i for i in line.split(' ') if i]
            repeat_dict[d[0]] =  int(d[thr_ind])
            len_dict[d[0]] =  int(d[1])
    return (len_dict,repeat_dict)

def import_vlm(filename,spliced_layer,unspliced_layer,gene_attr,cell_attr):
    with lp.connect(filename) as ds:
        S = ds.layers[spliced_layer][:]
        U = ds.layers[unspliced_layer][:]
        gene_names = ds.ra[gene_attr]
        nCells = len(ds.ca[cell_attr])
    return S,U,gene_names,nCells

def select_gene_set(loom_filepaths,feat_dict,viz=False,
                          results_to_exclude=[],seed=6,n_gen=10,
                          filt_param=(0.01,0.01,350,350,4,4,1),aesthetics=((12,4),0.15,3,"Spectral"),
                            attr_names_in=['spliced','unspliced','Gene','CellID']):
    """
    Examines a set of .loom files and selects a set of genes. Inputs:
    loom_filepaths: list of strings pointing to .loom files to access. 
    feat_dict: dictionary output by get_transcriptome. Used to select features with data. 
    viz: whether to visualize the set of genes filtered by clustering.
    results_to_exclude: list of strings pointing to previous search results. 
        The genes examined in these results are exluded from analysis to avoid duplication of work.
    seed: rng seed for selecting a random set of genes.
    n_gen: number of genes to select for analysis
    filt_param: Min threshold for mean, max threshold for max, mean threshold for max. Odd: U, even: S.

    The current workflow is optimized for kallisto|bus, so the ambiguous layer is ignored.
    """
    sz,alf,ptsz,cmap = aesthetics

    n_datasets = len(loom_filepaths)

    for i_data in range(n_datasets):
        #load in the loom file,
        loom_filepath = loom_filepaths[i_data]
        print('Dataset: '+loom_filepath)
        # vlm = vcy.VelocytoLoom(loom_filepath)
        # Ncells = len(vlm.ca[list(vlm.ca.keys())[0]])
        
        # if type(spliced_layer) is str:
        #     attr_names = [spliced_layer,unspliced_layer,gene_attr,cell_attr]
        # else:
        #     attr_names = [spliced_layer[i_data],unspliced_layer[i_data],gene_attr[i_data],cell_attr[i_data]]
        # if all(isinstance(x, list) for x in attr_names_in):
        if len(attr_names_in)>1:
            attr_names = attr_names_in[i_data]
        else:
            attr_names = attr_names_in[0]

        S,U,gene_names,Ncells = import_vlm(loom_filepath,*attr_names)
        #check which genes are represented in the dataset
        # gene_names_vlm = vlm.ra['Gene']
        ann_filt = identify_annotated_genes(gene_names,feat_dict)
        #get only the genes with length annotations
        S = S[ann_filt,:]
        U = U[ann_filt,:]
        gene_names = gene_names[ann_filt]
        
        #compute number of cells remaining after upstream processing
        print(str(Ncells)+ ' cells detected.')

        # gene_names = np.asarray(vlm.ra['Gene'])
        S_max = np.amax(S,1)
        U_max = np.amax(U,1)
        S_mean = np.mean(S,1)
        U_mean = np.mean(U,1)        

        #compute the lengths of each gene in filtered matrix
        len_arr = np.asarray([feat_dict[k] for k in gene_names])

        #compute clusters for easy identification of low-expression genes
        gene_cluster_labels = compute_cluster_labels(len_arr,S_mean)
        
        #plot all genes, color by cluster: blue for high expression, red for low.
        if viz:
            var_name = ('S','U')
            var_arr = ('S_mean','U_mean')

            fig1, ax1 = plt.subplots(nrows=1,ncols=2,figsize=sz)
            for i in range(2):
                ax1[i].scatter(np.log10(len_arr), np.log10(eval(var_arr[i]) + 0.001),s=ptsz,
                            c=gene_cluster_labels,alpha=alf,cmap=cmap)
                ax1[i].set_xlabel('log10 gene length')
                ax1[i].set_ylabel('log10 (mean '+var_name[i]+' + 0.001)')
        
        #plot genes in high-expression cluster
        if int(filt_param[-1]) is not -1:
            gene_filter = np.array(gene_cluster_labels,dtype=bool)
        else:
            gene_filter = np.ones(shape=gene_cluster_labels.shape,dtype=bool)
        if viz:
            fig2, ax2 = plt.subplots(nrows=1,ncols=2,figsize=sz)
            for i in range(2):
                ax2[i].scatter(np.log10(len_arr)[gene_filter], 
                            np.log10(eval(var_arr[i]) + 0.001)[gene_filter],s=ptsz,c='k',alpha=alf)
                ax2[i].set_xlabel('log10 gene length')
                ax2[i].set_ylabel('log10 (mean '+var_name[i]+' + 0.001)')
                
        print(str(sum(gene_filter))+' genes retained as high-expression.')
        gene_filter2  = gene_filter \
            & (U_mean > filt_param[0]) \
            & (S_mean > filt_param[1]) \
            & (S_max < filt_param[2]) \
            & (U_max < filt_param[3]) \
            & (S_max > filt_param[4]) \
            & (U_max > filt_param[5])
        
        #filer genes based on expression and sparsity
        gene_names_filt = gene_names[gene_filter2]
        expression_filt =  np.asarray([True if x in gene_names_filt else False for x in gene_names],dtype=bool)
        # vlm.filter_genes(by_custom_array=vlm_gene_filter)
        S = S[expression_filt,:]
        U = U[expression_filt,:]
        gene_names = gene_names[expression_filt]
        print(str(len(gene_names))+' genes retained in loom structure based on filter.')
        

        not_previously_run_filt = np.arange(len(gene_names))
        
        #Certain genes might have to be excluded based on previous runs, if we don't want to duplicate work.
        if len(results_to_exclude)>0:
            GN=[]
            for i_ in range(len(results_to_exclude)):
                with open(results_to_exclude[i_],'rb') as hf:
                    SO = pickle.load(hf)
                    GN.extend(SO.gene_names)
            print(str(len(GN))+' genes previously run...')
            GN = set(GN)
            print(str(len(GN))+' genes were unique.')
            not_previously_run_filt = [i_ for i_ in not_previously_run_filt if gene_names[i_] not in GN]
            print(str(len(not_previously_run_filt))+' genes retained in loom structure based on previous results.')

        #Finally, we would like to construct a set of genes to sample from.
        #If we are interested in examining multiple datasets, we simply take the intersection of genes
        #that meet the filtering criteria in all.
        gene_sampling_domain = gene_names[not_previously_run_filt]
        if i_data == 0:
            set_intersection = set(gene_sampling_domain)
        else:
            set_intersection = set_intersection.intersection(gene_sampling_domain)
        print('Gene set size: '+str(len(set_intersection)))
        print('-----------')
        

    #Finally, we select a subset of genes by sampling without replacement from the set of genes 
    #that meet our desired criteria in all datasets.
    random.seed(a=seed)
    sampling_gene_set = np.array(list(set_intersection))
    if n_gen < len(sampling_gene_set):
        gene_select = np.random.choice(sampling_gene_set,n_gen,replace=False)
        print(str(n_gen)+' genes selected.')
    else:
        gene_select = sampling_gene_set
        print(str(len(sampling_gene_set))+' genes selected: cannot satisfy query of '+str(n_gen)+' genes.')
    
    gene_select=list(gene_select)
    sampling_gene_set = list(sampling_gene_set)
    return gene_select, sampling_gene_set

def identify_annotated_genes(gene_names_vlm,feat_dict):
    n_gen_tot = len(gene_names_vlm)
    #check which genes I have length data for
    sel_ind_annot = [k for k in range(len(gene_names_vlm)) if gene_names_vlm[k] in feat_dict]
    
    NAMES = [gene_names_vlm[k] for k in range(len(sel_ind_annot))]
    COUNTS = collections.Counter(NAMES)
    sel_ind = [x for x in sel_ind_annot if COUNTS[gene_names_vlm[x]]==1]

    print(str(len(gene_names_vlm))+' features observed, '+str(len(sel_ind_annot))+' match genome annotations. '
        +str(len(sel_ind))+' are unique. ')

    ann_filt = np.zeros(n_gen_tot,dtype=bool)
    ann_filt[sel_ind] = True
    return ann_filt 

def compute_cluster_labels(len_arr,S_mean,init=np.asarray([[4,-2.5],[4.5,-0.5]])):
    warnings.filterwarnings("ignore")
    clusters = KMeans(n_clusters=2,init=init,algorithm="full").fit(
        np.vstack((np.log10(len_arr),np.log10(S_mean + 0.001))).T)
    warnings.resetwarnings()
    gene_cluster_labels = clusters.labels_
    return gene_cluster_labels

########################
## Class definitions
########################

class SearchParameters:
    def __init__(self):
        pass
    def define_search_parameters(self,num_restarts,lb_log,ub_log,maxiter,init_pattern ='moments',use_lengths=True):
        self.num_restarts = num_restarts
        self.lb_log = lb_log
        self.ub_log = ub_log
        self.maxiter = maxiter
        self.init_pattern = init_pattern
        self.use_lengths = use_lengths

class SearchData:
    def __init__(self):
        pass
    def set_gene_data(self,M,N,hist,moment_data,gene_log_lengths,n_gen,gene_names,Ncells,raw_U,raw_S):
        self.M = M
        self.N = N
        self.hist = hist
        self.moment_data = moment_data
        self.gene_log_lengths = gene_log_lengths
        self.n_gen = n_gen
        self.gene_names = gene_names
        self.Ncells = Ncells
        self.raw_U = raw_U
        self.raw_S = raw_S
    def set_search_params(self,search_params):
        self.search_params = search_params
    def set_scan_grid(self,n_pt1,n_pt2,samp_lb,samp_ub):
        self.n_pt1 = n_pt1
        self.n_pt2 = n_pt2
        self.N_pts = n_pt1*n_pt2
        (X,Y,sampl_vals) = build_grid((n_pt1,n_pt2),samp_lb,samp_ub)
        self.X = X
        self.Y = Y
        self.sampl_vals = sampl_vals
    def set_file_string(self,file_string):
        self.file_string = file_string
    def get_pts(self):
        point_list = [i for i in range(self.N_pts) if not os.path.isfile(self.file_string+'/grid_point_'+str(i)+'.pickle')]
        print(str(len(point_list)) + ' of '+str(self.N_pts)+' points to be evaluated.')
        self.point_list = point_list
    def set_results(self,divg,T_,gene_params,gene_spec_err):
        self.divg = divg
        self.T_ = T_
        self.gene_params = gene_params
        self.gene_spec_err = gene_spec_err
    def set_nosamp_results(self,nosamp_gene_params,nosamp_gene_spec_err,nosamp_search_params):
        self.nosamp_gene_params = nosamp_gene_params
        self.nosamp_gene_spec_err = nosamp_gene_spec_err
        self.nosamp_search_params = nosamp_search_params

class ResultData:
    def __init__(self):
        self.gene_names = []
        self.hist = []
        self.M = np.zeros(0,dtype=int)
        self.N = np.zeros(0,dtype=int)
        self.gene_log_lengths = np.zeros(0)
        self.moment_data = np.zeros((0,3))
        self.n_gen = 0
        
    def set_parameters(self,search_results):
        attrs = ('n_pt1','n_pt2','N_pts','X','Y','sampl_vals',
                 'search_params','Ncells')
        for attr in attrs:
            setattr(self,attr,getattr(search_results,attr))
        # if hasattr(search_results,'init_pattern'):
        #     setattr(self,'init_pattern',getattr(search_results,'init_pattern'))
        # else:
        #     setattr(self,'init_pattern','moments')
        # if hasattr(search_results,'phys_ub_nosamp'):
        #     setattr(self,'phys_ub_nosamp',getattr(search_results,'phys_ub_nosamp'))
        #     setattr(self,'phys_lb_nosamp',getattr(search_results,'phys_lb_nosamp'))
        self.divg = np.zeros(self.N_pts)
        self.gene_params = np.zeros((self.N_pts,0,3))
        self.gene_spec_err = np.zeros((self.N_pts,0))
        self.raw_U = np.zeros((0,self.Ncells),dtype=int)
        self.raw_S = np.zeros((0,self.Ncells),dtype=int)
        self.T_ = np.zeros((self.N_pts,0))

        if hasattr(search_results,'nosamp_gene_params'):
            self.nosamp_gene_params = np.zeros((0,3))
            self.nosamp_gene_spec_err = np.zeros(0)
            self.nosamp_search_params = search_results.nosamp_search_params

    def set_variables(self,search_results):
        self.divg += search_results.divg
        self.gene_params = np.concatenate((self.gene_params,search_results.gene_params),axis=1)
        self.gene_spec_err = np.concatenate((self.gene_spec_err,search_results.gene_spec_err),axis=1)
        self.N = np.concatenate((self.N,search_results.N),axis=0)
        self.M = np.concatenate((self.M,search_results.M),axis=0)
        self.gene_log_lengths = np.concatenate((self.gene_log_lengths,search_results.gene_log_lengths),axis=0)
        self.moment_data = np.concatenate((self.moment_data,search_results.moment_data),axis=0)
        self.raw_U = np.concatenate((self.raw_U,search_results.raw_U),axis=0)
        self.raw_S = np.concatenate((self.raw_S,search_results.raw_S),axis=0)
        self.T_ = np.concatenate((self.T_,np.reshape(search_results.T_,(self.N_pts,1))),axis=1)
        self.gene_names.extend(search_results.gene_names)
        self.hist.extend(search_results.hist)
        self.n_gen += search_results.n_gen

        if hasattr(search_results,'nosamp_gene_params'):
        	self.nosamp_gene_params = np.concatenate((self.nosamp_gene_params,search_results.nosamp_gene_params),axis=0)
        	self.nosamp_gene_spec_err = np.concatenate((self.nosamp_gene_spec_err,search_results.nosamp_gene_spec_err),axis=0)

    def find_best_params(self):
        self.best_ind = np.argmin(self.divg)
        self.best_samp_params = self.sampl_vals[self.best_ind]
        self.best_phys_params = self.gene_params[self.best_ind]
        if self.search_params.use_lengths:
            self.gene_spec_samp_params = np.array([(self.gene_log_lengths[i_] + self.best_samp_params[0], 
              self.best_samp_params[1]) for i_ in range(self.n_gen)])
        else:
            self.gene_spec_samp_params = np.array([(self.best_samp_params[0], 
              self.best_samp_params[1]) for i_ in range(self.n_gen)])
    
    def set_rej(self,pval,threshold=0.05,bonferroni=True,nosamp=False):
        if bonferroni:
            threshold=threshold/self.n_gen
        if not nosamp:
            self.gene_rej = pval<threshold
        else:
        	self.gene_rej_nosamp = pval<threshold

    def set_sigma(self,sigma,nosamp):
        if nosamp:
            self.sigma_nosamp=sigma
        else: 
            self.sigma = sigma
