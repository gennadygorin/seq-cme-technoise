import numpy as np
import multiprocessing
import sys
import csv
from datetime import date

from seq_cme_inference import *
def param_parser(inputfile):
    with open(inputfile,'r') as f:
        L = f.readline()
        dataset_directory = read_split(f)
        result_directory = read_split(f)
        loom_filenames = read_split(f).split(',')
        transcriptome_filename = read_split(f)
        polyA_threshold = int(read_split(f))
        transcriptome_ind = int(read_split(f))
        filter_param = [float(k) for k in read_split(f).split(',')]
        all_prev_results = read_split(f).split(',')
        if len(all_prev_results[0])==0:
            all_prev_results=[]
        attribute_names = eval(read_split(f))
        gene_sel_seed = int(read_split(f))
        n_gen = int(read_split(f))
        loom_index = int(read_split(f))
        gene_result_list = read_split(f).split(',')
        if len(gene_result_list[0])==0:
            gene_result_list=[]
        phys_lb = np.array([float(k) for k in read_split(f).split(',')])
        phys_ub = np.array([float(k) for k in read_split(f).split(',')])
        search_restarts = int(read_split(f))
        init_pattern = read_split(f)
        use_lengths = read_split(f)=="True"
        maxiter = int(read_split(f))
        n_pt1 = int(read_split(f))
        n_pt2 = int(read_split(f))
        samp_lb = np.array([float(k) for k in read_split(f).split(',')])
        samp_ub = np.array([float(k) for k in read_split(f).split(',')])
        ID_suffix = int(read_split(f))
        creator = read_split(f)
        NCOR = int(read_split(f))
        date_override = read_split(f)


    return dataset_directory, result_directory, loom_filenames,transcriptome_filename,\
    polyA_threshold, transcriptome_ind, filter_param, all_prev_results, attribute_names,\
    gene_sel_seed, n_gen, loom_index, gene_result_list, phys_lb, phys_ub, search_restarts,\
    init_pattern, use_lengths, maxiter, n_pt1,n_pt2,samp_lb, samp_ub, ID_suffix, creator, NCOR,\
    date_override

def read_split(f):
	L = f.readline()
	if L:
		return L.split(':')[1].strip()
	else:
		return None

def parout(PARINPUT):
	search_data,i=PARINPUT
	grid_search_driver(search_data,i)

def inference_workflow(input_param_file):
	dataset_directory, result_directory, datasets, transcriptome_filename,\
    polyA_threshold, transcriptome_ind, filter_param, all_prev_results, attribute_names,\
    gene_sel_seed, n_gen, IND, gene_result_list, phys_lb, phys_ub, search_restarts,\
    init_pattern, use_lengths, maxiter,n_pt1,n_pt2, samp_lb, samp_ub, ID_suffix, creator, NCOR, \
    date_override \
    = param_parser(input_param_file)
	
	len_dict = get_transcriptome(transcriptome_filename)[transcriptome_ind]
	datestr = (date.today().strftime("%y%m%d") if (not date_override) else date_override)
	loom_filenames = [dataset_directory+k+'.loom' for k in datasets] #loom file
	print(loom_filenames)

	if IND==-1 or len(gene_result_list)==0:
		#This performs gene selection, either as a "dry run" (IND=-1) or as the preliminary to a fit (IND>=0).
		print('Beginning preprocessing routine:')
		
		gene_set,trunc_gene_set = select_gene_set(loom_filenames,
		                      len_dict,viz=True,
		                          results_to_exclude=all_prev_results,seed=gene_sel_seed,n_gen=n_gen,
		                          filt_param=filter_param,attr_names_in=attribute_names)
		print('Gene set selected!')
		namestr=creator+'_'+datestr+'_'
		with open(namestr+'filtered_genes.csv','w') as f:
			writer = csv.writer(f)
			writer.writerow(trunc_gene_set)
		with open(namestr+'selected_genes.csv','w') as f:
			writer = csv.writer(f)
			writer.writerow(gene_set)


	if len(gene_result_list)>0:
		if gene_result_list[0][-3:]=='csv':
			with open(gene_result_list[0], newline='') as f:
				reader = csv.reader(f)
				gene_set = list(reader)[0]
			with open(gene_result_list[1], newline='') as f:
				reader = csv.reader(f)
				trunc_gene_set = list(reader)[0]
		else:
			result_data = import_datasets(gene_result_list)
			gene_set = result_data.gene_names

	if IND>=0:
		print('Beginning search routine.')
		attrn = (attribute_names[0] if len(attribute_names)==1 else attribute_names[IND])
		# attrn = attribute_names
		# print(attrn)
		search_data = get_gene_data(loom_filenames[IND],len_dict,gene_set,trunc_gene_set,viz=True,attr_names=attrn)
		search_params = SearchParameters()
		search_params.define_search_parameters(search_restarts,phys_lb,phys_ub,maxiter,init_pattern,use_lengths)
		search_data.set_search_params(search_params)

		search_data.set_scan_grid(n_pt1,n_pt2,samp_lb,samp_ub)
		file_string = create_dir(search_data,result_directory,ID_suffix,meta=datasets[IND],creator=creator,DATESTRING=datestr)

		search_data.get_pts()

		t1 = time.time()
		if NCOR>1:
			print('Starting search...')
			pool=multiprocessing.Pool(processes=NCOR)
			pool.map(parout,zip([search_data]*len(search_data.point_list),search_data.point_list))
			pool.close()
			pool.join()
			print('Parallelization done!')
		else:
			print('Starting search...')
			for i in search_data.point_list:
				grid_search_driver(search_data,i)
			print('Loop done!')

		grid_search_driver(search_data)
		t2 = time.time()
		print('Runtime: {:.1f} min.'.format((t2-t1)/60))
		dump_results(file_string,include_nosamp=True)
	return

if __name__=="__main__":
	input_param_file=sys.argv[1]
	inference_workflow(input_param_file)

