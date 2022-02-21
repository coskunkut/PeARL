# batch experiment generator

import glob
import json
import os
import sys
from datetime import datetime

import numpy as np
from pbnm.generator import Multinomial
from tqdm import tqdm


def dump_json(current_experiment, experiment_file_path):
	with open(experiment_file_path, "w") as json_file:
		json.dump(current_experiment, json_file, indent=4)

# get batch id
if len(sys.argv) == 2:
	batch_id = sys.argv[1]
else:
	sys.exit()

# read given batch config file
root_folder_path = os.path.abspath(os.curdir)
test_folder_path = os.path.join(root_folder_path, "test", "systematic_tests", "multinomial")
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
exp_type = batch_config["exp_type"]
number_of_runs = batch_config["parameters"]["experiment"]["number_of_runs"]
algorithms_to_run = batch_config["algorithms_to_run"]
seed_list = rng.integers(low=0, high=999999, size=number_of_runs)
experiment_file_count = 1

# sequence parameters
alp_car = batch_config["sequence_info"]["alphabet_cardinality"]

# drift in p
p_drift_type = list(batch_config["sequence_info"]["p_drift"].keys())[0]
if p_drift_type == "sudden":
	p_drift_frequency = batch_config["sequence_info"]["p_drift"]["sudden"]["frequency"]
	p_drift_severity = batch_config["sequence_info"]["p_drift"]["sudden"]["severity"]
	p_drift_n, p_drift_d = p_drift_frequency["n"], p_drift_frequency["d"]
	p_drift_min_diff, p_drift_max_diff = p_drift_severity["min"], p_drift_severity["max"]

# noise
injection_type = batch_config["sequence_info"]["noise"]["injection_type"]
if exp_type == "fixed_omega":
	start = batch_config["sequence_info"]["noise"]["noise_rates"]["start"]
	stop = batch_config["sequence_info"]["noise"]["noise_rates"]["stop"]
	step = batch_config["sequence_info"]["noise"]["noise_rates"]["step"]
	num = int(((stop - start) / step) + 1)
	noise_rate_list = np.linspace(start, stop, num)
elif exp_type == "changing_omega":
	n_drift_type = list(batch_config["sequence_info"]["noise"]["noise_drift"].keys())[0]
	if n_drift_type == "sudden":
		n_drift_frequency = batch_config["sequence_info"]["noise"]["noise_drift"]["sudden"]["frequency"]
		n_drift_severity = batch_config["sequence_info"]["noise"]["noise_drift"]["sudden"]["severity"]
		n_drift_n, n_drift_d = n_drift_frequency["n"], n_drift_frequency["d"]
		n_drift_min_diff, n_drift_max_diff = n_drift_severity["min"], n_drift_severity["max"]

total_number_of_runs = np.prod([
		number_of_runs,
		len(noise_rate_list) if exp_type == "fixed_omega" else 1,
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

	for run_count in range(number_of_runs):

		# get the current seed for rng
		current_seed = int(seed_list[run_count])

		# prepare current experiment file
		current_experiment = {
			"experiment_info": {
				"type": exp_type,
				"seed": current_seed,
				"name": ""
			},
			"algorithms_to_run": algorithms_to_run,
			"parameters": {
				"slwe_e": {
					"lambda": batch_config["parameters"]["slwe_e"]["lambda"]
				},
				"slwe_t": {
					"lambda": batch_config["parameters"]["slwe_t"]["lambda"]
				},
				"slwe_p_a": {
					"eta_min": batch_config["parameters"]["slwe_p_a"]["eta"]["min"],
					"eta_max": batch_config["parameters"]["slwe_p_a"]["eta"]["max"],
					"rho_eta_mapping": batch_config["parameters"]["slwe_p_a"]["rho_eta_mapping"],
					"k": batch_config["parameters"]["slwe_p_a"]["k"],
					"L": batch_config["parameters"]["slwe_p_a"]["L"]
				},
				"slwe_p_e": {
					"eta_min": batch_config["parameters"]["slwe_p_e"]["eta"]["min"],
					"eta_max": batch_config["parameters"]["slwe_p_e"]["eta"]["max"],
					"rho_eta_mapping": batch_config["parameters"]["slwe_p_e"]["rho_eta_mapping"],
					"k": batch_config["parameters"]["slwe_p_e"]["k"],
					"L": batch_config["parameters"]["slwe_p_e"]["L"]
				}
			},
			"sequence_info": {
				"alphabet_cardinality": alp_car,
				"p_drift": {},
				"noise": {}
			},
			"visualization": batch_config["visualization"]
		}

		if exp_type == "changing_omega":
			if n_drift_type == "sudden":
				multinomial_ins = Multinomial(alp_car, current_seed)
				multinomial_ins.sudden_drift_diff(p_drift_n, p_drift_d, p_drift_min_diff, p_drift_max_diff)
				multinomial_ins.noise_sudden_drift(n_drift_n, n_drift_d, n_drift_min_diff, n_drift_max_diff)

				current_experiment["sequence_info"]["p_drift"]["type"] = p_drift_type
				current_experiment["sequence_info"]["p_drift"]["params"] = {
					"frequency": p_drift_frequency,
					"severity": p_drift_severity
				}

				current_experiment["sequence_info"]["noise"]["injection_type"] = injection_type
				current_experiment["sequence_info"]["noise"]["drift_type"] = n_drift_type
				current_experiment["sequence_info"]["noise"]["drift_params"] = {
					"frequency": n_drift_frequency,
					"severity": n_drift_severity
				}
				current_experiment["sequence_info"]["noise"]["noise_regimes"] = multinomial_ins.noise_regimes

				current_experiment["sequence_info"]["regime_parameters"] = multinomial_ins.p
				current_experiment["sequence_info"]["sequence"] = list(map(str, multinomial_ins.o_sequence))
				current_experiment["sequence_info"]["p_sequence"] = multinomial_ins.p_sequence

				experiment_name = "exp_{}".format(experiment_file_count)
				current_experiment["experiment_info"]["name"] = experiment_name
				experiment_file_path = "{}/{}.json".format(experiment_files_folder_path, experiment_name)
				dump_json(current_experiment, experiment_file_path)
				experiment_file_count += 1
				pbar.update()

		if exp_type == "fixed_omega":
			for noise_rate in noise_rate_list:
				
				# sequence parameters
				if p_drift_type == "sudden":
					multinomial_ins = Multinomial(alp_car, current_seed)
					multinomial_ins.sudden_drift_diff(p_drift_n, p_drift_d, p_drift_min_diff, p_drift_max_diff)
					multinomial_ins.inject_white_noise(noise_rate)

					current_experiment["sequence_info"]["p_drift"]["type"] = p_drift_type
					current_experiment["sequence_info"]["p_drift"]["params"] = {
						"frequency": p_drift_frequency,
						"severity": p_drift_severity
					}

					current_experiment["sequence_info"]["noise"]["injection_type"] = injection_type
					current_experiment["sequence_info"]["noise"]["noise_rate"] = noise_rate

					current_experiment["sequence_info"]["regime_parameters"] = multinomial_ins.p
					current_experiment["sequence_info"]["sequence"] = list(map(str, multinomial_ins.o_sequence))
					current_experiment["sequence_info"]["p_sequence"] = multinomial_ins.p_sequence
					
					experiment_name = "exp_{}".format(experiment_file_count)
					current_experiment["experiment_info"]["name"] = experiment_name
					experiment_file_path = "{}/{}.json".format(experiment_files_folder_path, experiment_name)
					dump_json(current_experiment, experiment_file_path)
					experiment_file_count += 1
					pbar.update()
