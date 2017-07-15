import sys
import ast
import numpy as np
from fixed_alternatives_set_models import train_model, load_json

# args should be in order:
# 	model_name
#	hidden_szs
#	hiddens_nonlinearity
#	train_data_fname
#	validation_data_fname
#	weight_decay
#	learning_rate
#	alpha
#	cost_weight
#	save_path

def parse_args_and_run():
	print sys.argv
	assert len(sys.argv) == 11

	model_name = sys.argv[1]
	hidden_szs = ast.literal_eval(sys.argv[2])
	hiddens_nonlinearity = sys.argv[3]
	train_data_fname = sys.argv[4]
	validation_data_fname = sys.argv[5]
	weight_decay = ast.literal_eval(sys.argv[6])
	learning_rate = ast.literal_eval(sys.argv[7])
	alpha = ast.literal_eval(sys.argv[8])
	cost_weight = ast.literal_eval(sys.argv[9])
	save_path = sys.argv[10]

	# load stuff
	train_data = load_json(train_data_fname)
	validation_data = load_json(validation_data_fname)

	# fixed params
	data_rt = 'synthetic_data/'
	d = load_json(data_rt + 'true_lexicon.JSON')
	utt_set_sz = len(d)
	obj_set_sz = len(d['0'])
	utt_dict = load_json(data_rt + 'utt_inds_to_names.JSON')
	obj_dict = load_json(data_rt + 'obj_inds_to_names.JSON')
	cost_dict = load_json(data_rt + 'costs_by_utterance.JSON')
	true_lexicon  = load_json(data_rt + 'true_lexicon.JSON')
	true_lexicon = np.array([true_lexicon[str(k)] for k in range(utt_set_sz)]) + 10e-06
	obj_embedding_type = 'onehot'
	visualize_opt = False

	train_model(model_name, hidden_szs, hiddens_nonlinearity,
				 train_data, validation_data, utt_set_sz,
				 obj_set_sz, obj_embedding_type, utt_dict, obj_dict,
				 weight_decay, learning_rate,
				 visualize_opt,  
				 alpha, cost_dict, cost_weight, true_lexicon,
				 save_path)

if __name__=='__main__':
	parse_args_and_run()