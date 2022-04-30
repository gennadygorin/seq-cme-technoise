import numpy as np
import csv
import matplotlib.pyplot as plt
import pickle

from preprocess import identify_annotated_genes, get_transcriptome, import_vlm, filter_by_gene, make_dir


code_ver_global='020'

########################
## Debug and error logging
########################
import logging, sys

logging.basicConfig(stream=sys.stdout)
log = logging.getLogger()
log.setLevel(logging.DEBUG)

import logging.config
logging.config.dictConfig({
    'version': 1,
    'disable_existing_loggers': True,
})

########################
## Main code
########################

def extract_data(loom_filepath, transcriptome_filepath, dataset_name,
                    dataset_string, dir_string,\
                    viz=True,\
                    dataset_attr_names=['spliced','unspliced','gene_name','barcode'],\
                    padding = [10,10],
                    filter_cells_S = 0, filter_cells_U = 0):
    log.info('Beginning data extraction.')
    log.info('Dataset: '+dataset_name)

    transcriptome_dict = get_transcriptome(transcriptome_filepath)
    S,U,gene_names,n_cells = import_vlm(loom_filepath,*dataset_attr_names)

    #identify genes that are in the length annotations. discard all the rest.
    #for models without length-based sequencing, this may not be necessary 
    #though I do not see the purpose of analyzing genes that not in the
    #genome.
    #if this is ever necessary, just make a different reference list.
    annotation_filter = identify_annotated_genes(gene_names,transcriptome_dict)
    S,U,gene_names = filter_by_gene(annotation_filter,S,U,gene_names)

    #initialize the gene length array.
    len_arr = np.array([transcriptome_dict[k] for k in gene_names])

    gene_result_list_file = dir_string+'/genes.csv'
    try:
        with open(gene_result_list_file, newline='') as f:
            reader = csv.reader(f)
            analysis_gene_list = list(reader)[0]
        log.info('Gene list extracted from {}.'.format(gene_result_list_file))
    except:
        log.error('Gene list could not be extracted from {}.'.format(gene_result_list_file))
        #raise an error here in the next version.

    if viz:
        var_name = ('S','U')
        var_arr = (S.mean(1),U.mean(1))
        fig1, ax1 = plt.subplots(nrows=1,ncols=2,figsize=(12,4))
        for i in range(2):
            ax1[i].scatter(np.log10(len_arr), np.log10(var_arr[i] + 0.001),s=3,
                        c='silver',alpha=0.15)
            ax1[i].set_xlabel('log10 gene length')
            ax1[i].set_ylabel('log10 (mean '+var_name[i]+' + 0.001)')

    gene_names = list(gene_names)
    gene_filter = [gene_names.index(gene) for gene in analysis_gene_list]
    gene_names = np.asarray(gene_names)
    S,U,gene_names,len_arr = filter_by_gene(gene_filter,S,U,gene_names,len_arr)
    if filter_cells_S>0:
        log.info('Throwing out the {:.0f} highest spliced expression cells.'.format(filter_cells_S))
        n_cells -= filter_cells_S
        filt = np.argsort(-S.sum(0))[filter_cells:]
        S = S[:,filt]
        U = U[:,filt]
    if filter_cells_U>0:
        log.info('Throwing out the {:.0f} highest unspliced expression cells.'.format(filter_cells_U))
        n_cells -= filter_cells_U
        filt = np.argsort(-U.sum(0))[filter_cells:]
        S = S[:,filt]
        U = U[:,filt]

    if viz:
        for i in range(2):
            var_arr = (S.mean(1),U.mean(1))
            ax1[i].scatter(np.log10(len_arr), np.log10(var_arr[i] + 0.001),s=5,
                        c='firebrick',alpha=0.9)
        dataset_diagnostics_dir_string = dataset_string + '/diagnostic_figures'
        make_dir(dataset_diagnostics_dir_string)
        plt.savefig(dataset_diagnostics_dir_string+'/{}.png'.format(dataset_name))

    n_genes = len(gene_names)
    M = np.asarray([np.amax(U[gene_index]) for gene_index in range(n_genes)],dtype=int)+padding[0]
    N = np.asarray([np.amax(S[gene_index]) for gene_index in range(n_genes)],dtype=int)+padding[1]

    gene_log_lengths = np.log10(len_arr)

    hist = []
    moments = [] 
    raw_U = []
    raw_S = []
    for gene_index in range(n_genes):
        H, xedges, yedges = np.histogram2d(U[gene_index],S[gene_index], 
                                          bins=[np.arange(M[gene_index]+1)-0.5,
                                          np.arange(N[gene_index]+1)-0.5],
                                          density=True)
        hist.append(H)

        moments.append({'S_mean':S[gene_index].mean(), \
                        'U_mean':U[gene_index].mean(), \
                        'S_var':S[gene_index].var(), \
                        'U_var':U[gene_index].var()})
    
    attr_names = ('M','N','hist','moments','gene_log_lengths','n_genes','gene_names','n_cells','S','U')
    search_data = SearchData(attr_names,\
                             M,N,hist,moments,gene_log_lengths,n_genes,gene_names,n_cells,S,U)
    search_data_string = dataset_string+'/raw.sd'
    store_search_data(search_data,search_data_string)
    return search_data

########################
## Helper functions
########################
def store_search_data(search_data,search_data_string):
    try:
        with open(search_data_string,'wb') as sdfs:
            pickle.dump(search_data, sdfs)
        log.info('Search data stored to {}.'.format(search_data_string))
    except:
        log.error('Search data could not be stored to {}.'.format(search_data_string))

########################
## Helper classes
########################
class SearchData:
    def __init__(self,attr_names,*input_data):
        for j in range(len(input_data)):
            setattr(self,attr_names[j],input_data[j])