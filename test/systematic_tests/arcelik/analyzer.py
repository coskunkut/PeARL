# batch experiment analyser

import csv
import json
import os
import sys

import numpy as np
from natsort import natsorted
from pbnm.tools import align_matrix
from sklearn import metrics
from tqdm import tqdm

# get experiment id
if len(sys.argv) == 2:
	batch_id = sys.argv[1]
else:
	sys.exit()

# get a list of files in experiment folder
root_folder_path = os.path.abspath(os.curdir)
test_folder_path = os.path.join(root_folder_path, "test", "systematic_tests", "arcelik")
batch_output_folder_path = os.path.join(test_folder_path, "batch_output", batch_id)
if not os.path.exists(batch_output_folder_path):
	print("Batch output folder not found.")
	sys.exit()
experiment_files_folder_path = os.path.join(batch_output_folder_path, "experiment_files")
experiment_files = natsorted(os.listdir(experiment_files_folder_path), reverse=False)
run_logs_path = os.path.join(batch_output_folder_path, "run_logs.csv")

# get a sample experiment file to obtain the list of algorithms run
sample_experiment_file_path = "{}/{}".format(experiment_files_folder_path, experiment_files[0])
with open(sample_experiment_file_path) as sample_experiment_file:
	sample_experiment = json.load(sample_experiment_file)
	algorithms_to_run = sample_experiment["algorithms_to_run"]

# prepare column names of the log file
fieldnames = [
		# "experiment_type",
		"sequence_id",
		"seed",
		"experiment_name",
		"alp_car",
		"noise_rate",
]
if "slwe_e" in algorithms_to_run:
	fieldnames.extend([
		"slwe_e_lambda",
		"slwe_e_mae"
	])
if "slwe_t" in algorithms_to_run:
	fieldnames.extend([
		"slwe_t_lambda",
		"slwe_t_mae"
	])
if "slwe_p_a" in algorithms_to_run:
	fieldnames.extend([
		"slwe_p_a_eta_min",
		"slwe_p_a_eta_max",
		"slwe_p_a_rho_eta_mapping",
		"slwe_p_a_k",
		"slwe_p_a_L",
		"slwe_p_a_mae"
	])
if "slwe_p_e" in algorithms_to_run:
	fieldnames.extend([
		"slwe_p_e_eta_min",
		"slwe_p_e_eta_max",
		"slwe_p_e_rho_eta_mapping",
		"slwe_p_e_k",
		"slwe_p_e_L",
		"slwe_p_e_mae"
	])

# prepare csv file, write header row
with open(run_logs_path, mode="w") as run_logs_file:
	writer = csv.DictWriter(run_logs_file, fieldnames=fieldnames)
	writer.writeheader()

# iterate experiment file list
experiment_file_count = 1
for experiment_file in tqdm(
	experiment_files,
	desc = "analyze".ljust(10),
	bar_format = '{l_bar}{bar:10}{r_bar}{bar:-10b}',
	ascii = True
	):

	# read experiment file as JSON
	experiment_file_path = "{}/{}".format(experiment_files_folder_path, experiment_file)
	with open(experiment_file_path) as current_experiment_file:
		current_experiment = json.load(current_experiment_file)
	experiment_file_count += 1

	# get sequence info
	alp_car = current_experiment["preprocessing_info"]["clustering_result"]["number_of_clusters"]
	noise_rate = current_experiment["preprocessing_info"]["noise_injection"]["noise_rate"]
	
	# build result dictionary for current run
	run_result_dict = {}
	# run_result_dict["experiment_type"] = current_experiment["experiment_info"]["type"]
	run_result_dict["seed"] = current_experiment["experiment_info"]["seed"]
	run_result_dict["sequence_id"] = current_experiment["experiment_info"]["sequence_id"]
	run_result_dict["experiment_name"] = current_experiment["experiment_info"]["name"]
	run_result_dict["alp_car"] = alp_car
	run_result_dict["noise_rate"] = noise_rate

	sequence_length = len(current_experiment["sequence"]["discrete_clean"])
	regime_count = len(current_experiment["sequence"]["parameters"])
	regime_length = int(sequence_length / regime_count)
	sampling_rate = 50000
	p_sequence = []
	for key, val in current_experiment["sequence"]["parameters"].items():
		p = list(val.values())
		p_sequence.extend(np.tile(p, (regime_length, 1)).tolist())
	# current_experiment["sequence"]["p_sequence"] = p_sequence

	# slwe_e
	if "slwe_e" in algorithms_to_run:
		run_result_dict["slwe_e_lambda"] = current_experiment["parameters"]["slwe_e"]["lambda"]
		slwe_e_estimates = current_experiment["results"]["slwe_e"]["estimates"]
		slwe_e_token_index_map = current_experiment["results"]["slwe_e"]["token_index_map"]
		slwe_e_aligned_p_sequence = align_matrix(p_sequence, slwe_e_token_index_map)
		slwe_e_mae = metrics.mean_absolute_error(slwe_e_aligned_p_sequence, slwe_e_estimates)
		run_result_dict["slwe_e_mae"] = slwe_e_mae
		current_experiment["results"]["slwe_e"]["mae"] = slwe_e_mae

	# slwe_t
	if "slwe_t" in algorithms_to_run:
		run_result_dict["slwe_t_lambda"] = current_experiment["parameters"]["slwe_t"]["lambda"]
		slwe_t_estimates = current_experiment["results"]["slwe_t"]["estimates"]
		slwe_t_token_index_map = current_experiment["results"]["slwe_t"]["token_index_map"]
		slwe_t_aligned_p_sequence = align_matrix(p_sequence, slwe_t_token_index_map)
		slwe_t_mae = metrics.mean_absolute_error(slwe_t_aligned_p_sequence, slwe_t_estimates)
		run_result_dict["slwe_t_mae"] = slwe_t_mae
		current_experiment["results"]["slwe_t"]["mae"] = slwe_t_mae

	# slwe_p_a
	if "slwe_p_a" in algorithms_to_run:
		run_result_dict["slwe_p_a_eta_min"] = current_experiment["parameters"]["slwe_p_a"]["eta_min"]
		run_result_dict["slwe_p_a_eta_max"] = current_experiment["parameters"]["slwe_p_a"]["eta_max"]
		run_result_dict["slwe_p_a_rho_eta_mapping"] = current_experiment["parameters"]["slwe_p_a"]["rho_eta_mapping"]
		run_result_dict["slwe_p_a_k"] = current_experiment["parameters"]["slwe_p_a"]["k"]
		run_result_dict["slwe_p_a_L"] = current_experiment["parameters"]["slwe_p_a"]["L"]
		slwe_p_a_estimates = current_experiment["results"]["slwe_p_a"]["estimates"]
		slwe_p_a_token_index_map = current_experiment["results"]["slwe_p_a"]["token_index_map"]
		slwe_p_a_aligned_p_sequence = align_matrix(p_sequence, slwe_p_a_token_index_map)
		slwe_p_a_mae = metrics.mean_absolute_error(slwe_p_a_aligned_p_sequence, slwe_p_a_estimates)
		run_result_dict["slwe_p_a_mae"] = slwe_p_a_mae
		current_experiment["results"]["slwe_p_a"]["mae"] = slwe_p_a_mae
		
	# slwe_p_e
	if "slwe_p_e" in algorithms_to_run:
		run_result_dict["slwe_p_e_eta_min"] = current_experiment["parameters"]["slwe_p_e"]["eta_min"]
		run_result_dict["slwe_p_e_eta_max"] = current_experiment["parameters"]["slwe_p_e"]["eta_max"]
		run_result_dict["slwe_p_e_rho_eta_mapping"] = current_experiment["parameters"]["slwe_p_e"]["rho_eta_mapping"]
		run_result_dict["slwe_p_e_k"] = current_experiment["parameters"]["slwe_p_e"]["k"]
		run_result_dict["slwe_p_e_L"] = current_experiment["parameters"]["slwe_p_e"]["L"]
		slwe_p_e_estimates = current_experiment["results"]["slwe_p_e"]["estimates"]
		slwe_p_e_token_index_map = current_experiment["results"]["slwe_p_e"]["token_index_map"]
		slwe_p_e_aligned_p_sequence = align_matrix(p_sequence, slwe_p_e_token_index_map)
		slwe_p_e_mae = metrics.mean_absolute_error(slwe_p_e_aligned_p_sequence, slwe_p_e_estimates)
		run_result_dict["slwe_p_e_mae"] = slwe_p_e_mae
		current_experiment["results"]["slwe_p_e"]["mae"] = slwe_p_e_mae
	
	# append results of current run to csv file
	with open(run_logs_path, mode="a") as run_logs_file:
		writer = csv.DictWriter(run_logs_file, fieldnames=fieldnames)
		writer.writerow(run_result_dict)

	# append results to experiment file
	with open(experiment_file_path, "w") as current_experiment_file:
		json.dump(current_experiment, current_experiment_file, indent=4)
