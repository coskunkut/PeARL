# batch experiment generator

import json
import os
import glob
import sys
import numpy as np
from tqdm import tqdm
from datetime import datetime

from pbnm.generator import Multinomial

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
number_of_runs = batch_config["parameters"]["experiment"]["number_of_runs"]
algorithms_to_run = batch_config["algorithms_to_run"]
seed_list = rng.integers(low=0, high=999999, size=number_of_runs)
experiment_file_count = 1

# sequence parameters
drift_types = batch_config["sequence_info"]["drift_type"].keys()
run_multiplier = 1
if "sudden" in drift_types:
	run_multiplier *= len(batch_config["sequence_info"]["drift_type"]["sudden"]["frequency"])
	run_multiplier *= len(batch_config["sequence_info"]["drift_type"]["sudden"]["severity"])
elif "incremental_sin" in drift_types:
	run_multiplier *= len(batch_config["sequence_info"]["drift_type"]["incremental_sin"]["pi_range"])
	run_multiplier *= len(batch_config["sequence_info"]["drift_type"]["incremental_sin"]["frequency"])
elif "incremental_lin" in drift_types:
	run_multiplier *= len(batch_config["sequence_info"]["drift_type"]["incremental_lin"]["frequency"])
	run_multiplier *= len(batch_config["sequence_info"]["drift_type"]["incremental_lin"]["severity"])

alp_car_list = batch_config["sequence_info"]["alphabet_cardinality"]
noise_rate_list = batch_config["sequence_info"]["noise_rate"]

# slwe parameters
slwe_e_lambda_list = batch_config["parameters"]["slwe_e"]["lambda"]
slwe_t_lambda_list = batch_config["parameters"]["slwe_t"]["lambda"]

# slwe_p parameters
slwe_p_eta_list = batch_config["parameters"]["slwe_p"]["eta"]
slwe_p_k_list = batch_config["parameters"]["slwe_p"]["k"]
slwe_p_L_list = batch_config["parameters"]["slwe_p"]["L"]


total_number_of_runs = np.prod([
		number_of_runs,
		len(drift_types),
		run_multiplier,
		len(alp_car_list),
		len(noise_rate_list),
		len(slwe_e_lambda_list),
		len(slwe_t_lambda_list),
		len(slwe_p_eta_list),
		len(slwe_p_k_list),
		len(slwe_p_L_list)
	]
)

# generate experiment files
with tqdm(
	total = total_number_of_runs,
	position = 0,
	leave = True,
	desc = "generator".ljust(10),
	bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}',
	ascii = True
	) as pbar:

	for run_count in range(number_of_runs):

		# get the current seed for rng
		current_seed = int(seed_list[run_count])

		# prepare current experiment file
		current_experiment = {}
		current_experiment["experiment_info"] = {}
		current_experiment["experiment_info"]["type"] = "multinomial"
		current_experiment["experiment_info"]["seed"] = current_seed
		current_experiment["algorithms_to_run"] = algorithms_to_run
		current_experiment["parameters"] = {}
		current_experiment["sequence_info"] = {}
		current_experiment["sequence_info"]["drift_type"] = {}

		# slwe_e parameters
		for slwe_e_lambda in slwe_e_lambda_list:
			current_experiment["parameters"]["slwe_e"] = {}
			current_experiment["parameters"]["slwe_e"]["lambda"] = slwe_e_lambda

			# slwe_t parameters
			for slwe_t_lambda in slwe_t_lambda_list:
				current_experiment["parameters"]["slwe_t"] = {}
				current_experiment["parameters"]["slwe_t"]["lambda"] = slwe_t_lambda

				# slwe_p parameters
				for slwe_p_eta in slwe_p_eta_list:
					for slwe_p_k in slwe_p_k_list:
						for slwe_p_L in slwe_p_L_list:

									current_experiment["parameters"]["slwe_p"] = {}
									current_experiment["parameters"]["slwe_p"]["eta_min"] = slwe_p_eta["min"]
									current_experiment["parameters"]["slwe_p"]["eta_max"] = slwe_p_eta["max"]
									current_experiment["parameters"]["slwe_p"]["k"] = slwe_p_k
									current_experiment["parameters"]["slwe_p"]["L"] = slwe_p_L

									for alp_car in alp_car_list:

										current_experiment["sequence_info"]["alphabet_cardinality"] = alp_car

										for noise_rate in noise_rate_list:

											current_experiment["sequence_info"]["noise_rate"] = noise_rate
											
											# sequence parameters
											for drift_type in drift_types:

												if drift_type == "sudden":
													frequencies = batch_config["sequence_info"]["drift_type"]["sudden"]["frequency"]
													severities = batch_config["sequence_info"]["drift_type"]["sudden"]["severity"]
													for frequency in frequencies:
														n, d = frequency["n"], frequency["d"]
														for severity in severities:
															min_diff, max_diff = severity["min"], severity["max"]
															multinomial_ins = Multinomial(current_seed)
															multinomial_ins.sudden_drift_diff(alp_car, n, d, min_diff, max_diff)
															multinomial_ins.inject_white_noise(alp_car, noise_rate)
															current_experiment["sequence_info"]["drift_type"] = "sudden"
															current_experiment["sequence_info"]["drift_params"] = {}
															current_experiment["sequence_info"]["drift_params"]["n"] = n
															current_experiment["sequence_info"]["drift_params"]["d"] = d
															current_experiment["sequence_info"]["drift_params"]["p_diff_min"] = min_diff
															current_experiment["sequence_info"]["drift_params"]["p_diff_max"] = max_diff
															current_experiment["sequence_info"]["regime_parameters"] = multinomial_ins.p
															# current_experiment["sequence_info"]["sequence"] = multinomial_ins.sequence
															current_experiment["sequence_info"]["sequence"] = list(map(int, multinomial_ins.sequence))
															current_experiment["sequence_info"]["p_sequence"] = multinomial_ins.p_sequence
															experiment_name = "exp_{}".format(experiment_file_count)
															current_experiment["experiment_info"]["name"] = experiment_name
															experiment_file_path = "{}/{}.json".format(experiment_files_folder_path, experiment_name)
															dump_json(current_experiment, experiment_file_path)
															experiment_file_count += 1
															pbar.update()

												# elif drift_type == "mixed":

												# 	small_sudden = batch_config["sequence_info"]["drift_type"]["mixed"]["small_sudden"]
												# 	large_sudden = batch_config["sequence_info"]["drift_type"]["mixed"]["large_sudden"]
												# 	multinomial_ins = Multinomial(current_seed)

												# 	# small sudden
												# 	multinomial_ins.sudden_drift_diff(
												# 		alp_car, 
												# 		small_sudden["n"], 
												# 		small_sudden["d"], 
												# 		small_sudden["min"], 
												# 		small_sudden["max"],
												# 		append = False
												# 	)

												# 	# large sudden
												# 	multinomial_ins.sudden_drift_diff(
												# 		alp_car, 
												# 		large_sudden["n"], 
												# 		large_sudden["d"], 
												# 		large_sudden["min"], 
												# 		large_sudden["max"],
												# 		append = True
												# 	)

												# 	current_experiment["sequence_info"]["drift_type"] = "mixed"
												# 	current_experiment["sequence_info"]["drift_params"] = {}
												# 	current_experiment["sequence_info"]["drift_params"]["small_sudden"] = {}
												# 	current_experiment["sequence_info"]["drift_params"]["small_sudden"]["n"] = small_sudden["n"]
												# 	current_experiment["sequence_info"]["drift_params"]["small_sudden"]["d"] = small_sudden["d"]
												# 	current_experiment["sequence_info"]["drift_params"]["small_sudden"]["p_diff_min"] = small_sudden["min"]
												# 	current_experiment["sequence_info"]["drift_params"]["small_sudden"]["p_diff_max"] = small_sudden["max"]
												# 	current_experiment["sequence_info"]["drift_params"]["large_sudden"] = {}
												# 	current_experiment["sequence_info"]["drift_params"]["large_sudden"]["n"] = large_sudden["n"]
												# 	current_experiment["sequence_info"]["drift_params"]["large_sudden"]["d"] = large_sudden["d"]
												# 	current_experiment["sequence_info"]["drift_params"]["large_sudden"]["p_diff_min"] = large_sudden["min"]
												# 	current_experiment["sequence_info"]["drift_params"]["large_sudden"]["p_diff_max"] = large_sudden["max"]
												# 	current_experiment["sequence_info"]["regime_parameters"] = multinomial_ins.p
												# 	current_experiment["sequence_info"]["sequence"] = multinomial_ins.sequence
												# 	current_experiment["sequence_info"]["p_sequence"] = multinomial_ins.p_sequence
												# 	experiment_name = "exp_{}".format(experiment_file_count)
												# 	current_experiment["experiment_info"]["name"] = experiment_name
												# 	experiment_file_path = "{}/{}.json".format(experiment_files_folder_path, experiment_name)
												# 	dump_json(current_experiment, experiment_file_path)
												# 	experiment_file_count += 1
												# 	pbar.update()
