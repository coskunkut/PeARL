# batch experiment runner

import collections
import json
import multiprocessing as mp
import os
import sys

import huffman
import numpy as np
from natsort import natsorted
from pbnm.estimator import SLWE, SLWE_P
from tqdm import tqdm


# each process calls this function to run a given experiment
def run_experiment(experiment_file):

	# read experiment file as JSON
	experiment_file_path = "{}/{}".format(experiment_files_folder_path, experiment_file)
	with open(experiment_file_path) as current_experiment_file:
		current_experiment = json.load(current_experiment_file)

	# skip if already processed
	# if "results" in current_experiment:
		# return

	# current_experiment["results"] = {}

	# read parameters
	sequence = current_experiment["sequence"]["discrete_clean"]
	alp_car = current_experiment["sequence"]["alphabet_cardinality"]
	algorithms_to_run = current_experiment["algorithms_to_run"]

	# huffman_static
	if "huffman_static" in algorithms_to_run:
		huffman_static_encoded_data = ""
		freqs = collections.Counter(sequence).items()
		cb = huffman.codebook(freqs)
		for token in sequence:
			huffman_static_encoded_data += cb[token]
		static_huffman_compressed_size = len(huffman_static_encoded_data)
		current_experiment["results"]["huffman_static"] = static_huffman_compressed_size

	# huffman_slwe_e
	if "slwe_e" in algorithms_to_run:
		slwe_e_lambda = current_experiment["parameters"]["slwe_e"]["lambda"]
		slwe_e_ins = SLWE(alp_car, slwe_e_lambda)
		slwe_e_ins.process_sequence(sequence, progress=False)
		slwe_e_estimates = []
		for estimate in slwe_e_ins.estimates:
			current_estimate = []
			for token, index in slwe_e_ins.token_index_map.items():
				current_estimate.append((token, estimate[index]))
			slwe_e_estimates.append(current_estimate)
		huffman_slwe_e_encoded_data = ""
		for token_count, token in enumerate(sequence):
			freqs = list(slwe_e_estimates[token_count])
			cb_huffman = huffman.codebook(freqs)
			huffman_slwe_e_encoded_data += cb_huffman[token]
		huffman_slwe_e_compressed_size = len(huffman_slwe_e_encoded_data)
		current_experiment["results"]["huffman_slwe_e"] = huffman_slwe_e_compressed_size

	# huffman_slwe_t
	if "slwe_t" in algorithms_to_run:
		slwe_t_lambda = current_experiment["parameters"]["slwe_t"]["lambda"]
		slwe_t_ins = SLWE(alp_car, slwe_t_lambda)
		slwe_t_ins.process_sequence(sequence, progress=False)
		slwe_t_estimates = []
		for estimate in slwe_t_ins.estimates:
			current_estimate = []
			for token, index in slwe_t_ins.token_index_map.items():
				current_estimate.append((token, estimate[index]))
			slwe_t_estimates.append(current_estimate)
		huffman_slwe_t_encoded_data = ""
		for token_count, token in enumerate(sequence):
			freqs = list(slwe_t_estimates[token_count])
			cb_huffman = huffman.codebook(freqs)
			huffman_slwe_t_encoded_data += cb_huffman[token]
		huffman_slwe_t_compressed_size = len(huffman_slwe_t_encoded_data)
		current_experiment["results"]["huffman_slwe_t"] = huffman_slwe_t_compressed_size

	# huffman_slwe_p
	if "slwe_p" in algorithms_to_run:
		slwe_p_ins = SLWE_P(alp_car)
		slwe_p_ins.eta_min = current_experiment["parameters"]["slwe_p"]["eta_min"]
		slwe_p_ins.eta_max = current_experiment["parameters"]["slwe_p"]["eta_max"]
		slwe_p_ins.k = current_experiment["parameters"]["slwe_p"]["k"]
		slwe_p_ins.L = current_experiment["parameters"]["slwe_p"]["L"]
		slwe_p_ins.process_sequence(sequence, progress=False)
		slwe_p_estimates = []
		for estimate in slwe_p_ins.estimates:
			current_estimate = []
			for token, index in slwe_p_ins.token_index_map.items():
				current_estimate.append((token, estimate[index]))
			slwe_p_estimates.append(current_estimate)
		huffman_slwe_p_encoded_data = ""
		for token_count, token in enumerate(sequence):
			freqs = list(slwe_p_estimates[token_count])
			cb_huffman = huffman.codebook(freqs)
			huffman_slwe_p_encoded_data += cb_huffman[token]
		huffman_slwe_p_compressed_size = len(huffman_slwe_p_encoded_data)
		current_experiment["results"]["huffman_slwe_p"] = huffman_slwe_p_compressed_size

	# append results to experiment file
	with open(experiment_file_path, "w") as current_experiment_file:
		json.dump(current_experiment, current_experiment_file, indent=4)

# get experiment id
if len(sys.argv) == 2:
	batch_id = sys.argv[1]
else:
	sys.exit()

# get a list of files in experiment folder
root_folder_path = os.path.abspath(os.curdir)
test_folder_path = os.path.join(root_folder_path, "test", "systematic_tests", "adaptive_coding")
batch_output_folder_path = os.path.join(test_folder_path, "batch_output", batch_id)
if not os.path.exists(batch_output_folder_path):
	print("Batch output folder not found.")
	sys.exit()
experiment_files_folder_path = os.path.join(batch_output_folder_path, "experiment_files")
experiment_files = natsorted(os.listdir(experiment_files_folder_path), reverse=False)
figures_folder_path = os.path.join(batch_output_folder_path, "figures")

# map each experimet file to a process
pool = mp.Pool()
# pool.map(run_experiment, experiment_files)
r = list(tqdm(
	pool.imap(run_experiment, experiment_files),
	total = len(experiment_files),
	desc = "run".ljust(10),
	bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}',
	ascii = True
))
pool.close()
