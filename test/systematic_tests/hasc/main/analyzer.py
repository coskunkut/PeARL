# batch experiment analyser

import json
import os
import csv
import sys
import numpy as np
from natsort import natsorted
from tqdm import tqdm
from sklearn import metrics

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

# prepare column names of the log file
fieldnames = [
		"experiment_type",
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
if "slwe_p" in algorithms_to_run:
	fieldnames.extend([
		"slwe_p_eta_min",
		"slwe_p_eta_max",
		"slwe_p_k",
		"slwe_p_L",
		"slwe_p_mae"
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
	# run_result_dict["seed"] = current_experiment["experiment_info"]["seed"]
	run_result_dict["experiment_name"] = current_experiment["experiment_info"]["name"]
	run_result_dict["alp_car"] = alp_car
	run_result_dict["noise_rate"] = noise_rate

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
		slwe_e_mae = metrics.mean_absolute_error(p_sequence, slwe_e_estimates)
		run_result_dict["slwe_e_mae"] = slwe_e_mae
		current_experiment["results"]["slwe_e"]["mae"] = slwe_e_mae

	# slwe_t
	if "slwe_t" in algorithms_to_run:
		run_result_dict["slwe_t_lambda"] = current_experiment["parameters"]["slwe_t"]["lambda"]
		slwe_t_estimates = current_experiment["results"]["slwe_t"]["estimates"]
		slwe_t_mae = metrics.mean_absolute_error(p_sequence, slwe_t_estimates)
		run_result_dict["slwe_t_mae"] = slwe_t_mae
		current_experiment["results"]["slwe_t"]["mae"] = slwe_t_mae

	# slwe_p
	if "slwe_p" in algorithms_to_run:
		run_result_dict["slwe_p_eta_min"] = current_experiment["parameters"]["slwe_p"]["eta_min"]
		run_result_dict["slwe_p_eta_max"] = current_experiment["parameters"]["slwe_p"]["eta_max"]
		run_result_dict["slwe_p_k"] = current_experiment["parameters"]["slwe_p"]["k"]
		run_result_dict["slwe_p_L"] = current_experiment["parameters"]["slwe_p"]["L"]
		slwe_p_estimates = current_experiment["results"]["slwe_p"]["estimates"]
		slwe_p_mae = metrics.mean_absolute_error(p_sequence, slwe_p_estimates)
		run_result_dict["slwe_p_mae"] = slwe_p_mae
		current_experiment["results"]["slwe_p"]["mae"] = slwe_p_mae
	
	# append results of current run to csv file
	with open(run_logs_path, mode="a") as run_logs_file:
		writer = csv.DictWriter(run_logs_file, fieldnames=fieldnames)
		writer.writerow(run_result_dict)

	# append results to experiment file
	with open(experiment_file_path, "w") as current_experiment_file:
		json.dump(current_experiment, current_experiment_file, indent=4)
