# batch experiment generator

import collections
import glob
import json
import os
import sys
from datetime import datetime

import matplotlib.image as mpimg
import numpy as np
import pandas as pd
from natsort import natsorted
from scipy import signal
from tqdm import tqdm


def dump_json(current_experiment, experiment_file_path):
	with open(experiment_file_path, "w") as json_file:
		json.dump(current_experiment, json_file, indent=4)

# get batch id
if len(sys.argv) == 2:
	batch_id = sys.argv[1]
else:
	sys.exit()

# build paths
root_folder_path = os.path.abspath(os.curdir)
test_folder_path = os.path.join(root_folder_path, "test", "systematic_tests", "adaptive_coding")
data_folder_path = os.path.join(test_folder_path, "batch_input", batch_id)
input_file_names = natsorted(os.listdir(data_folder_path), reverse=False)

# read given batch config file
batch_config_folder_path = os.path.join(test_folder_path, "batch_config")
batch_config_path = os.path.join(batch_config_folder_path, "{}.json".format(batch_id))
if not os.path.exists(batch_config_path):
	print("Batch config file not found.")
	sys.exit()
with open(batch_config_path) as batch_config_file:
	batch_config = json.load(batch_config_file)

# prepare folders
batch_output_parent_folder_path = os.path.join(test_folder_path, "batch_output")
if not os.path.exists(batch_output_parent_folder_path):
	os.mkdir(batch_output_parent_folder_path)
batch_output_folder_path = os.path.join(batch_output_parent_folder_path, batch_id)
if not os.path.exists(batch_output_folder_path):
	os.mkdir(batch_output_folder_path)
experiment_files_folder_path = os.path.join(batch_output_folder_path, "experiment_files")
if not os.path.exists(experiment_files_folder_path):
	os.mkdir(experiment_files_folder_path)
else:
	experiment_files = glob.glob("{}/*.json".format(experiment_files_folder_path))
	for experiment_file in experiment_files:
	    os.remove(experiment_file)

# initialize rng
default_seed = 1123
rng = np.random.default_rng(default_seed)

# experiment parameters
# number_of_runs = batch_config["parameters"]["experiment"]["number_of_runs"]
# seed_list = rng.integers(low=0, high=999999, size=number_of_runs)
algorithms_to_run = batch_config["algorithms_to_run"]
experiment_type = batch_config["experiment_info"]["type"]
experiment_file_count = 1

# slwe parameters
slwe_e_lambda_list = batch_config["parameters"]["slwe_e"]["lambda"]
slwe_t_lambda_list = batch_config["parameters"]["slwe_t"]["lambda"]

# slwe_p parameters
slwe_p_eta_min_list = batch_config["parameters"]["slwe_p"]["eta_min"]
slwe_p_eta_max_list = batch_config["parameters"]["slwe_p"]["eta_max"]
slwe_p_k_list = batch_config["parameters"]["slwe_p"]["k"]
slwe_p_L_list = batch_config["parameters"]["slwe_p"]["L"]


total_number_of_runs = np.prod([
		len(input_file_names),
		len(slwe_e_lambda_list),
		len(slwe_t_lambda_list),
		len(slwe_p_eta_min_list),
		len(slwe_p_eta_max_list),
		len(slwe_p_k_list),
		len(slwe_p_L_list)
	]
)

# generate experiment files
with tqdm(
	total = total_number_of_runs,
	position = 0,
	leave = True,
	desc = "generate".ljust(10),
	bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}',
	ascii = True
	) as pbar:

	for input_file_name in input_file_names:

		# prepare current experiment file
		current_experiment = {}
		current_experiment["experiment_info"] = {}
		current_experiment["experiment_info"]["experiment_id"] = ""
		current_experiment["experiment_info"]["type"] = experiment_type
		current_experiment["algorithms_to_run"] = algorithms_to_run
		current_experiment["parameters"] = {}
		current_experiment["sequence"] = {}
		current_experiment["results"] = {}

		# build absolute path for the current input image
		input_file_path = "{}/{}".format(data_folder_path, input_file_name)

		# read current input file based on experiment type
		if experiment_type == "image":
			input_sequence = [str(pixel) for pixel in mpimg.imread(input_file_path).flatten()]
		elif experiment_type == "text":
			with open(input_file_path) as input_file:
				input_sequence = list(input_file.read())
		current_experiment["sequence"]["discrete_clean"] = input_sequence

		# calculate total length and uncompressed size
		total_sequence_length = len(input_sequence)
		current_experiment["sequence"]["length"] = total_sequence_length
		uncompressed_size = total_sequence_length * 8
		current_experiment["results"]["uncompressed"] = uncompressed_size

		# obtain alphabet
		alphabet = list(collections.Counter(input_sequence).keys())
		current_experiment["sequence"]["alphabet"] = alphabet

		# obtain alphabet cardinality
		alphabet_cardinality = len(alphabet)
		current_experiment["sequence"]["alphabet_cardinality"] = alphabet_cardinality

		# nr = 0.4
		# noise_indices = np.random.choice(total_sequence_length, int(total_sequence_length*nr))
		# for i in noise_indices:
		# 	input_sequence[i] = np.random.choice(alphabet)
			
		# slwe_e parameters
		for slwe_e_lambda in slwe_e_lambda_list:
			current_experiment["parameters"]["slwe_e"] = {}
			current_experiment["parameters"]["slwe_e"]["lambda"] = slwe_e_lambda

			# slwe_t parameters
			for slwe_t_lambda in slwe_t_lambda_list:
				current_experiment["parameters"]["slwe_t"] = {}
				current_experiment["parameters"]["slwe_t"]["lambda"] = slwe_t_lambda

				# slwe_p parameters
				for slwe_p_eta_min in slwe_p_eta_min_list:
					for slwe_p_eta_max in slwe_p_eta_max_list:
						for slwe_p_k in slwe_p_k_list:
							for slwe_p_L in slwe_p_L_list:
					
								current_experiment["parameters"]["slwe_p"] = {}
								current_experiment["parameters"]["slwe_p"]["eta_min"] = slwe_p_eta_min
								current_experiment["parameters"]["slwe_p"]["eta_max"] = slwe_p_eta_max
								current_experiment["parameters"]["slwe_p"]["k"] = slwe_p_k
								current_experiment["parameters"]["slwe_p"]["L"] = slwe_p_L

								experiment_name = "exp_{}".format(experiment_file_count)
								current_experiment["experiment_info"]["name"] = experiment_name
								experiment_file_path = "{}/{}.json".format(experiment_files_folder_path, experiment_name)
								dump_json(current_experiment, experiment_file_path)
								experiment_file_count += 1
								pbar.update()
