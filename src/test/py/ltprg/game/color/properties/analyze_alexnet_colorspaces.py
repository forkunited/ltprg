from __future__ import division
import numpy as np
from scipy.spatial import distance
from representational_similarity_analysis import make_dissimilarity_matrix, rsa_wrapper
from get_stim_embeddings import load_stim_hsls, get_stim_alexnet_embeddings
from colorspace_conversions import hsls_to_rgbs, rgbs_to_labs, rgbs_to_luvs, color_paper_space
from dimensionality_reduction import apply_tsne
import random

def wrapper(num_samples, data_path, save_path):
	stim_hsls = load_stim_hsls(data_path) # array of hsls

	# select random subset
	random.shuffle(stim_hsls)
	stim_hsls = stim_hsls[0 : num_samples]

	# get coordinates in various color spaces
	stim_rgbs = np.array(hsls_to_rgbs(stim_hsls))
	stim_labs = np.array(rgbs_to_labs(stim_rgbs))
	stim_luvs = np.array(rgbs_to_luvs(stim_rgbs))
	stim_cps = np.array(color_paper_space(stim_hsls))
	stim_hsls = np.array(stim_hsls)

	# get Alexnet fc-6 and fc-7 embeddings of stim RGB images
	fc6_embeddings = get_stim_alexnet_embeddings(stim_rgbs, 'fc6')
	fc7_embeddings = get_stim_alexnet_embeddings(stim_rgbs, 'fc7')

	# perform tSNE to visualize
	apply_tsne(fc6_embeddings, 'AlexNet FC-6', hsls_to_rgbs(stim_hsls), save_path)
	apply_tsne(fc7_embeddings, 'AlexNet FC-7', hsls_to_rgbs(stim_hsls), save_path)

	# specify distance function for RDMs
	dist_f = distance.euclidean
	print '\nDistance function:'
	print dist_f

	# create RDMs
	hsl_rdm = make_dissimilarity_matrix(stim_hsls, dist_f)
	lab_rdm = make_dissimilarity_matrix(stim_labs, dist_f)
	luv_rdm = make_dissimilarity_matrix(stim_luvs, dist_f)
	cps_rdm = make_dissimilarity_matrix(stim_cps, dist_f)
	fc6_rdm = make_dissimilarity_matrix(fc6_embeddings, dist_f)
	fc7_rdm = make_dissimilarity_matrix(fc7_embeddings, dist_f)

	# representational similarity analysis
	print '\nComparing FC-6 Representations to Colorspace Representations\n'
	rsa_wrapper(fc6_rdm, hsl_rdm, 'AlexNet FC-6', 'HSL')
	rsa_wrapper(fc6_rdm, cps_rdm, 'AlexNet FC-6', 'Fourier Transform')
	rsa_wrapper(fc6_rdm, lab_rdm, 'AlexNet FC-6', 'CIELAB')
	rsa_wrapper(fc6_rdm, luv_rdm, 'AlexNet FC-6', 'CIELUV')
	
	print '\nComparing FC-7 Representations to Colorspace Representations\n'
	rsa_wrapper(fc7_rdm, hsl_rdm, 'AlexNet FC-7', 'HSL')
	rsa_wrapper(fc7_rdm, cps_rdm, 'AlexNet FC-7', 'Fourier Transform')
	rsa_wrapper(fc7_rdm, lab_rdm, 'AlexNet FC-7', 'CIELAB')
	rsa_wrapper(fc7_rdm, luv_rdm, 'AlexNet FC-7', 'CIELUV')
	
	print '\nComparing FC-6 and FC-7 Representations\n'
	rsa_wrapper(fc6_rdm, fc7_rdm, 'AlexNet FC-6', 'AlexNet FC-7')

	print '\nComparing Colorspace Representations\n'
	rsa_wrapper(hsl_rdm, cps_rdm, 'HSL', 'Fourier Transform')
	rsa_wrapper(hsl_rdm, lab_rdm, 'HSL', 'CIELAB')
	rsa_wrapper(hsl_rdm, lab_rdm, 'HSL', 'CIELUV')
	rsa_wrapper(lab_rdm, cps_rdm, 'CIELAB', 'Fourier Transform')
	rsa_wrapper(luv_rdm, cps_rdm, 'CIELUV', 'Fourier Transform')
	rsa_wrapper(lab_rdm, luv_rdm, 'CIELAB', 'CIELUV')

if __name__=='__main__':
	wrapper(2000, '../../../../../../../../Datasets/color_sua_speaker/',
		    'figures/')