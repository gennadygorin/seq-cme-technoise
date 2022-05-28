
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt

import time
import loompy as lp
import anndata as ad
import os
from datetime import datetime
import pytz
import collections
import csv
import warnings



code_ver_global='021'

########################
## Debug and error logging
########################
import logging, sys

logging.basicConfig(stream=sys.stdout)
log = logging.getLogger()
log.setLevel(logging.INFO)

import logging.config
logging.config.dictConfig({
    'version': 1,
    'disable_existing_loggers': True,
})

########################
## Main code
########################

def construct_batch(loom_filepaths, transcriptome_filepath, dataset_names, batch_id=1,\
                    n_genes=100, seed=6, viz=True,\
                    filt_param={'min_U_mean':0.01,'min_S_mean':0.01,'max_U_max':400,\
                                'max_S_max':400,'min_U_max':3,'min_S_max':3},\
                    attribute_names=['spliced','unspliced','gene_name','barcode'],\
                    meta='batch',\
                    datestring=datetime.now(pytz.timezone('US/Pacific')).strftime("%y%m%d"),\
                    creator='gg', code_ver=code_ver_global,\
                    batch_location='.'):
    """
    This function runs basic pre-processing on the batch, creates directories, and writes a 
    list of genes to analyze.

    Input: 
    loom_filepaths: 
    transcriptome_filepath:
    n_genes: 
    seed:
    filt_param:    
    batch_location: the parent directory (no trailing /).
    meta: any string metadata. I recommend putting number of genes here.
    batch_id: experiment number.
    datestring: current date, in ISO format. Califonia time by default.
    creator: creator initials.
    code_ver: version of the code used to perform the experiment.
    """
    log.info('Beginning data preprocessing and filtering.')
    dir_string = batch_location + '/' + ('_'.join((creator, datestring, code_ver, meta, str(batch_id))))
    make_dir(dir_string)
    dataset_strings = []

    if type(loom_filepaths) is str:
        loom_filepaths = [loom_filepaths] #if we get in a single string, we are processing one file /
    n_datasets = len(loom_filepaths)

    if type(attribute_names[0]) is str:
        attribute_names = [attribute_names]*n_datasets

    transcriptome_dict = get_transcriptome(transcriptome_filepath)

    for dataset_index in range(n_datasets):
        loom_filepath = loom_filepaths[dataset_index]
        log.info('Dataset: '+dataset_names[dataset_index])
        dataset_attr_names = attribute_names[dataset_index] #pull out the correct attribute names
        S,U,gene_names,n_cells = import_raw(loom_filepath,*dataset_attr_names)
        log.info(str(n_cells)+ ' cells detected.')

        #identify genes that are in the length annotations. discard all the rest.
        #for models without length-based sequencing, this may not be necessary 
        #though I do not see the purpose of analyzing genes that not in the
        #genome.
        #if this is ever necessary, just make a different reference list.
        annotation_filter = identify_annotated_genes(gene_names,transcriptome_dict)
        S,U,gene_names = filter_by_gene(annotation_filter,S,U,gene_names)

        #initialize the gene length array.
        len_arr = np.array([transcriptome_dict[k] for k in gene_names])

        #compute summary statistics
        gene_exp_filter = threshold_by_expression(S,U,filt_param)
        if viz:
            var_name = ('S','U')
            var_arr = (S.mean(1),U.mean(1))

            fig1, ax1 = plt.subplots(nrows=1,ncols=2,figsize=(12,4))
            for i in range(2):
                ax1[i].scatter(np.log10(len_arr)[~gene_exp_filter], np.log10(var_arr[i][~gene_exp_filter] + 0.001),s=3,
                            c='silver',alpha=0.15)
                ax1[i].scatter(np.log10(len_arr[gene_exp_filter]), np.log10(var_arr[i][gene_exp_filter] + 0.001),s=3,
                            c='indigo',alpha=0.3)
                ax1[i].set_xlabel('log10 gene length')
                ax1[i].set_ylabel('log10 (mean '+var_name[i]+' + 0.001)')

        S,U,gene_names = filter_by_gene(gene_exp_filter,S,U,gene_names)

        if dataset_index == 0:
            set_intersection = set(gene_names)
        else:
            set_intersection = set_intersection.intersection(gene_names)
        
        dataset_dir_string = dir_string + '/' + dataset_names[dataset_index]
        make_dir(dataset_dir_string)
        dataset_strings.append(dataset_dir_string)

    log.info('Gene set size: '+str(len(set_intersection)))
    
    np.random.seed(seed)
    sampling_gene_set = np.sort(np.array(list(set_intersection)))
    if n_genes < len(sampling_gene_set):
        gene_select = np.random.choice(sampling_gene_set,n_genes,replace=False)
        log.info(str(n_genes)+' genes selected.')
    else:
        gene_select = sampling_gene_set
        log.warning(str(len(sampling_gene_set))+' genes selected: cannot satisfy query of '+str(n_genes)+' genes.')
    
    gene_select=list(gene_select)
    sampling_gene_set = list(sampling_gene_set)

    save_gene_list(dir_string,gene_select,'genes')
    save_gene_list(dir_string,sampling_gene_set,'gene_set')

    if viz:
        diagnostics_dir_string = dir_string + '/diagnostic_figures'
        make_dir(diagnostics_dir_string)
        for figure_ind in plt.get_fignums():
            plt.figure(figure_ind)
            plt.savefig(diagnostics_dir_string+'/{}.png'.format(dataset_names[figure_ind-1]))
    return dir_string,dataset_strings

########################
## Helper functions
########################

# def filter_by_gene(filter,S,U,gene_names):
#     #take in a Boolean or integer filter over genes, select the entries of inputs that match the filter.
#     S = S[filter,:]
#     U = U[filter,:]
#     gene_names = gene_names[filter]
#     return S,U,gene_names
def filter_by_gene(filter,*args):
    #take in a Boolean or integer filter over genes, select the entries of inputs that match the filter.
    out = []
    for arg in args:
        out += [arg[filter].squeeze()]
    return tuple(out)

def threshold_by_expression(S,U,
    filt_param={'min_U_mean':0.01,\
                'min_S_mean':0.01,\
                'max_U_max':350,\
                'max_S_max':350,\
                'min_U_max':4,\
                'min_S_max':4}):
    #take in S, U, and a dict of thresholds for mean and max, output a Boolean filter of genes that meet these thresholds.
    S_max = S.max(1)
    U_max = U.max(1)
    S_mean = S.mean(1)
    U_mean = U.mean(1)  

    gene_exp_filter = \
        (U_mean > filt_param['min_U_mean']) \
        & (S_mean > filt_param['min_S_mean']) \
        & (U_max < filt_param['max_U_max']) \
        & (S_max < filt_param['max_S_max']) \
        & (U_max > filt_param['min_U_max']) \
        & (S_max > filt_param['min_S_max'])     
    log.info(str(np.sum(gene_exp_filter))+ ' genes retained after expression filter.')
    return gene_exp_filter


def save_gene_list(dir_string,gene_list,filename):
    #saves a list of genes to csv format.
    with open(dir_string+'/'+filename+'.csv','w') as f:
        writer = csv.writer(f)
        writer.writerow(gene_list)

def make_dir(dir_string):
    #attempts to create a directory.
    try: 
        os.mkdir(dir_string) 
        log.info('Directory ' + dir_string+ ' created.')
    except OSError as error: 
        log.warning('Directory ' + dir_string + ' already exists.')

def import_raw(filename,spliced_layer,unspliced_layer,gene_attr,cell_attr):
    fn_extension = filename.split('.')[-1]
    if fn_extension == 'loom': #loom file
        return import_vlm(filename,spliced_layer,unspliced_layer,gene_attr,cell_attr)
    elif fn_extension == 'h5ad':
        return import_h5ad(filename,spliced_layer,unspliced_layer,gene_attr,cell_attr)
    else:
        return import_mtx(filename)

def import_h5ad(filename,spliced_layer,unspliced_layer,gene_attr,cell_attr):
    #Imports anndata file with spliced and unspliced RNA counts.
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    ds = ad.read_h5ad(filename)
    # with lp.connect(filename) as ds:
    S = ds.layers[spliced_layer][:]
    U = ds.layers[unspliced_layer][:]
    gene_names = ds.obs[gene_attr]
    nCells = len(ds.var[cell_attr])
    warnings.resetwarnings()
    return S,U,gene_names,nCells


def import_vlm(filename,spliced_layer,unspliced_layer,gene_attr,cell_attr):
    #Imports a velocyto loom file with spliced and unspliced RNA counts.
    #Note that there is a new deprecation warning in the h5py package 
    #underlying loompy.
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    with lp.connect(filename) as ds:
        S = ds.layers[spliced_layer][:]
        U = ds.layers[unspliced_layer][:]
        gene_names = ds.ra[gene_attr]
        nCells = len(ds.ca[cell_attr])
    warnings.resetwarnings()
    return S,U,gene_names,nCells

def get_transcriptome(transcriptome_filepath,repeat_thr=15):
    """
    Imports transcriptome length/repeat from a previously generated file. Input:
    transcriptome_filepath: path to the file. This is a simple space-separated file.
        The convention for each line is name - length - 5mers - 6mers -.... 50mers - more
    repeat_thr: threshold for minimum repeat length to consider. 
        By default, this is 15, and so will return number of polyA stretches of 
        length 15 or more in the gene.
    the repeat dictionary is not used in this version of the code.
    """
    repeat_dict = {}
    len_dict = {}
    thr_ind = repeat_thr-3
    with open(transcriptome_filepath,'r') as file:   
        for line in file.readlines():
            d = [i for i in line.split(' ') if i]
            repeat_dict[d[0]] =  int(d[thr_ind])
            len_dict[d[0]] =  int(d[1])
    return len_dict

def identify_annotated_genes(gene_names_vlm,feat_dict):
    #check which genes have length data.
    n_gen_tot = len(gene_names_vlm)
    sel_ind_annot = [k for k in range(len(gene_names_vlm)) if gene_names_vlm[k] in feat_dict]
    
    NAMES = [gene_names_vlm[k] for k in range(len(sel_ind_annot))]
    COUNTS = collections.Counter(NAMES)
    sel_ind = [x for x in sel_ind_annot if COUNTS[gene_names_vlm[x]]==1]

    log.info(str(len(gene_names_vlm))+' features observed, '+str(len(sel_ind_annot))+' match genome annotations. '
        +str(len(sel_ind))+' were unique.')

    ann_filt = np.zeros(n_gen_tot,dtype=bool)
    ann_filt[sel_ind] = True
    return ann_filt 