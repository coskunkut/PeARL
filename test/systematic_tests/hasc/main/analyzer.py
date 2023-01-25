# batch experiment analyser

import csv
import json
import os
import sys

import numpy as np
from natsort import natsorted
from sklearn import metrics
from tqdm import tqdm

from pearl.tools.signal import align_matrix

# get experiment id
if len(sys.argv) == 2:
	batch_id = sys.argv[1]
else:
	sys.exit()

# get a list of files in experiment folder
root_folder_path = os.path.abspath(os.curdir)
test_folder_path = os.path.join(root_folder_path, "test", "systematic_tests", "hasc", "main")
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
	exp_type = sample_experiment["experiment_info"]["type"]

# prepare column names of the log file
fieldnames = [
	"exp_type",
	"exp_seed",
	"exp_name",
	"sequence_id",
	"alp_car"
]

# exp type dependent field names
if exp_type == "fixed_omega":
	fieldnames.extend([
		"noise_rate"
	])
elif exp_type == "changing_omega":
	fieldnames.extend([
		"noise_drift_params"
	])

# algorithm dependent field names
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
if "pearl_a" in algorithms_to_run:
	fieldnames.extend([
		"pearl_a_eta_min",
		"pearl_a_eta_max",
		"pearl_a_k",
		"pearl_a_L",
		"pearl_a_mae"
	])
if "pearl_e" in algorithms_to_run:
	fieldnames.extend([
		"pearl_e_eta_min",
		"pearl_e_eta_max",
		"pearl_e_k",
		"pearl_e_L",
		"pearl_e_mae"
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
	alp_car = current_experiment["preprocessing"]["alphabet_cardinality"]

	if exp_type == "changing_omega":
		n_drift_type = current_experiment["preprocessing"]["noise"]["drift_type"]
		if n_drift_type == "sudden":
			n_drift_params = "{}/{}/{}".format(
				current_experiment["preprocessing"]["noise"]["drift_params"]["frequency"]["d"],
				current_experiment["preprocessing"]["noise"]["drift_params"]["severity"]["min"],
				current_experiment["preprocessing"]["noise"]["drift_params"]["severity"]["max"]
			)
	
	# build result dictionary for current run
	run_result_dict = {
		"exp_type": current_experiment["experiment_info"]["type"],
		"exp_seed": current_experiment["experiment_info"]["seed"],
		"exp_name": current_experiment["experiment_info"]["name"],
		"sequence_id": current_experiment["experiment_info"]["sequence_id"],
		"alp_car": current_experiment["preprocessing"]["alphabet_cardinality"]
	}

	if exp_type == "fixed_omega":
		run_result_dict["noise_rate"] = current_experiment["preprocessing"]["noise"]["noise_rate"]
	elif exp_type == "changing_omega":
		run_result_dict["noise_drift_params"] = n_drift_params

	sequence_length = len(current_experiment["sequence"]["discrete_clean"])
	p_sequence = np.zeros([sequence_length, alp_car])
	for key, val in current_experiment["sequence"]["activities"].items():
		activity_start, activity_end, activity_name = key.split("-")
		p = list(val["distribution_parameters"].values())
		p_sequence[int(activity_start):int(activity_end), :] = p

	# slwe_e
	if "slwe_e" in algorithms_to_run:
		run_result_dict["slwe_e_lambda"] = current_experiment["parameters"]["slwe_e"]["lambda"]
		slwe_e_estimates = current_experiment["results"]["slwe_e"]["estimates"]
		slwe_e_symbol_index_map = current_experiment["results"]["slwe_e"]["symbol_index_map"]
		slwe_e_aligned_p_sequence = align_matrix(p_sequence, slwe_e_symbol_index_map)
		slwe_e_abs_diff = np.abs(np.subtract(slwe_e_aligned_p_sequence, slwe_e_estimates[1:]))
		slwe_e_abs_error = np.mean(slwe_e_abs_diff, axis=1)
		slwe_e_mae = metrics.mean_absolute_error(slwe_e_aligned_p_sequence, slwe_e_estimates[1:])
		run_result_dict["slwe_e_mae"] = slwe_e_mae
		current_experiment["results"]["slwe_e"]["ae"] = slwe_e_abs_error.tolist()
		current_experiment["results"]["slwe_e"]["mae"] = slwe_e_mae

	# slwe_t
	if "slwe_t" in algorithms_to_run:
		run_result_dict["slwe_t_lambda"] = current_experiment["parameters"]["slwe_t"]["lambda"]
		slwe_t_estimates = current_experiment["results"]["slwe_t"]["estimates"]
		slwe_t_symbol_index_map = current_experiment["results"]["slwe_t"]["symbol_index_map"]
		slwe_t_aligned_p_sequence = align_matrix(p_sequence, slwe_t_symbol_index_map)
		slwe_t_abs_diff = np.abs(np.subtract(slwe_t_aligned_p_sequence, slwe_t_estimates[1:]))
		slwe_t_abs_error = np.mean(slwe_t_abs_diff, axis=1)
		slwe_t_mae = metrics.mean_absolute_error(slwe_t_aligned_p_sequence, slwe_t_estimates[1:])
		run_result_dict["slwe_t_mae"] = slwe_t_mae
		current_experiment["results"]["slwe_t"]["ae"] = slwe_t_abs_error.tolist()
		current_experiment["results"]["slwe_t"]["mae"] = slwe_t_mae

	# pearl_a
	if "pearl_a" in algorithms_to_run:
		run_result_dict["pearl_a_eta_min"] = current_experiment["parameters"]["pearl_a"]["eta_min"]
		run_result_dict["pearl_a_eta_max"] = current_experiment["parameters"]["pearl_a"]["eta_max"]
		run_result_dict["pearl_a_k"] = current_experiment["parameters"]["pearl_a"]["k"]
		run_result_dict["pearl_a_L"] = current_experiment["parameters"]["pearl_a"]["L"]
		pearl_a_estimates = current_experiment["results"]["pearl_a"]["estimates"]
		pearl_a_symbol_index_map = current_experiment["results"]["pearl_a"]["symbol_index_map"]
		pearl_a_aligned_p_sequence = align_matrix(p_sequence, pearl_a_symbol_index_map)
		pearl_a_abs_diff = np.abs(np.subtract(pearl_a_aligned_p_sequence, pearl_a_estimates[1:]))
		pearl_a_abs_error = np.mean(pearl_a_abs_diff, axis=1)
		pearl_a_mae = metrics.mean_absolute_error(pearl_a_aligned_p_sequence, pearl_a_estimates[1:])
		run_result_dict["pearl_a_mae"] = pearl_a_mae
		current_experiment["results"]["pearl_a"]["ae"] = pearl_a_abs_error.tolist()
		current_experiment["results"]["pearl_a"]["mae"] = pearl_a_mae

	# pearl_e
	if "pearl_e" in algorithms_to_run:
		run_result_dict["pearl_e_eta_min"] = current_experiment["parameters"]["pearl_e"]["eta_min"]
		run_result_dict["pearl_e_eta_max"] = current_experiment["parameters"]["pearl_e"]["eta_max"]
		run_result_dict["pearl_e_k"] = current_experiment["parameters"]["pearl_e"]["k"]
		run_result_dict["pearl_e_L"] = current_experiment["parameters"]["pearl_e"]["L"]
		pearl_e_estimates = current_experiment["results"]["pearl_e"]["estimates"]
		pearl_e_symbol_index_map = current_experiment["results"]["pearl_e"]["symbol_index_map"]
		pearl_e_aligned_p_sequence = align_matrix(p_sequence, pearl_e_symbol_index_map)
		pearl_e_abs_diff = np.abs(np.subtract(pearl_e_aligned_p_sequence, pearl_e_estimates[1:]))
		pearl_e_abs_error = np.mean(pearl_e_abs_diff, axis=1)
		pearl_e_mae = metrics.mean_absolute_error(pearl_e_aligned_p_sequence, pearl_e_estimates[1:])
		run_result_dict["pearl_e_mae"] = pearl_e_mae
		current_experiment["results"]["pearl_e"]["ae"] = pearl_e_abs_error.tolist()
		current_experiment["results"]["pearl_e"]["mae"] = pearl_e_mae
	
	# append results of current run to csv file
	with open(run_logs_path, mode="a") as run_logs_file:
		writer = csv.DictWriter(run_logs_file, fieldnames=fieldnames)
		writer.writerow(run_result_dict)

	# append results to experiment file
	with open(experiment_file_path, "w") as current_experiment_file:
		json.dump(current_experiment, current_experiment_file, indent=4)
