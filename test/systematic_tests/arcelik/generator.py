# batch experiment generator

import collections
import glob
import json
import math
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from bkmeans import BKMeans
from natsort import natsorted
from scipy import signal
from sklearn import cluster, metrics, preprocessing
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
test_folder_path = os.path.join(root_folder_path, "test", "systematic_tests", "arcelik")
data_folder_path = os.path.join(root_folder_path, "data", "arcelik")

# clean_sequence_parent_folder_path = os.path.join(data_folder_path, "0_sequence")

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
seed_list = rng.integers(low=0, high=999999, size=number_of_runs)
algorithms_to_run = batch_config["algorithms_to_run"]
sampling_rate = 50000
experiment_file_count = 1

# ----

# read the info file
info_table_path = os.path.join(data_folder_path, "info.csv")
info_table = pd.read_csv(info_table_path, sep=",", header=0)

# filter data
filtered_info_table = info_table.copy()
filtered_info_table = filtered_info_table.loc[filtered_info_table["Anonim_ID"].isin(batch_config["data"]["motor_ids"])]

# get raw signals
raw_signal_file_names = filtered_info_table[batch_config["data"]["feature"]].tolist()
raw_signal_file_paths = [os.path.join(data_folder_path, "samples", fn) for fn in raw_signal_file_names]

# clustering parameters
clustering_config = batch_config["preprocessing"]["clustering"]

# slwe parameters
slwe_e_lambda_list = batch_config["parameters"]["slwe_e"]["lambda"]
slwe_t_lambda_list = batch_config["parameters"]["slwe_t"]["lambda"]

# slwe_p parameters
slwe_p_eta_min_list = batch_config["parameters"]["slwe_p"]["eta_min"]
slwe_p_eta_max_list = batch_config["parameters"]["slwe_p"]["eta_max"]
slwe_p_k_list = batch_config["parameters"]["slwe_p"]["k"]
slwe_p_L_list = batch_config["parameters"]["slwe_p"]["L"]

noise_rate_list = batch_config["preprocessing"]["noise_injection"]["noise_rate"]

total_number_of_runs = np.prod([
		number_of_runs,
		len(batch_config["data"]["motor_ids"]),
		len(noise_rate_list),
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

	for run_count in range(number_of_runs):

		# get the current seed for rng
		current_seed = int(seed_list[run_count])
		current_rng = np.random.default_rng(current_seed)

		for raw_signal_file_path in raw_signal_file_paths:

			for noise_rate in noise_rate_list:

				# get person and experiment id
				sequence_data_id = raw_signal_file_path.rsplit("/", 1)[1].split(".")[0]


				# prepare current experiment file
				current_experiment = {}
				current_experiment["algorithms_to_run"] = algorithms_to_run
				current_experiment["experiment_info"] = {}
				current_experiment["experiment_info"]["sequence_id"] = sequence_data_id
				current_experiment["experiment_info"]["experiment_id"] = ""
				current_experiment["experiment_info"]["seed"] = current_seed
				current_experiment["preprocessing_info"] = {}
				current_experiment["parameters"] = {}
				current_experiment["sequence"] = {}

				# read raw data
				X_clean = np.fromfile(raw_signal_file_path, dtype=np.float32, count=-1, sep="").reshape(-1, 1)
				steady_state_index_start, steady_state_index_end = [math.ceil(t * sampling_rate) for t in batch_config["data"]["steady_state_time"]]
				X_clean = X_clean[steady_state_index_start:steady_state_index_end]
				observation_count = X_clean.shape[0]
				bounds = [np.min(X_clean, axis=0), np.max(X_clean, axis=0)]
				noisy_instance_count = int(np.ceil(observation_count * noise_rate))
				noisy_instance_indices = current_rng.choice(observation_count, noisy_instance_count, replace=False)
				white_noise = np.zeros(noisy_instance_count)
				white_noise += current_rng.uniform(bounds[0], bounds[1], noisy_instance_count)
				X_noisy = np.array(X_clean, copy=True)  
				X_noisy[noisy_instance_indices, 0] += white_noise
				noise_flags = np.zeros([observation_count], dtype=int)
				noise_flags[noisy_instance_indices] += 1

				current_experiment["preprocessing_info"]["noise_injection"] = {
					"noise_rate": noise_rate
				}

				if clustering_config["algorithm"] == "bkmeans":
					number_of_clusters = clustering_config["parameters"]["n_clusters"]
					current_config = {
						"algorithm": "bkmeans",
						"parameters": {
							"n_clusters": number_of_clusters,
							"m": 4
						}
					}
					clusterer = BKMeans(**current_config["parameters"], random_state=0)
					clusterer.fit(X_clean)

					discrete_clean_sequence = clusterer.predict(X_clean)
					discrete_noisy_sequence = clusterer.predict(X_noisy)

					current_experiment["sequence"]["discrete_clean"] = [str(i) for i in discrete_clean_sequence]
					current_experiment["sequence"]["discrete_noisy"] = [str(i) for i in discrete_noisy_sequence]

					current_experiment["preprocessing_info"]["clustering_config"] = clustering_config
					current_experiment["preprocessing_info"]["clustering_result"] = {}
					current_experiment["preprocessing_info"]["clustering_result"]["number_of_clusters"] = number_of_clusters

					# distribution parameters
					discrete_clean_sequence_pd = pd.Series(discrete_clean_sequence)
					distribution_parameters = discrete_clean_sequence_pd.value_counts(normalize=True, sort=False).to_dict()
					for token in range(number_of_clusters):
						if token not in distribution_parameters:
							distribution_parameters[token] = 0
					distribution_parameters = collections.OrderedDict(sorted(distribution_parameters.items(), key=lambda t: t[0]))
					current_experiment["sequence"]["distribution_parameters"] = distribution_parameters

					# # calculate distribution parameters for each action in the sequence
					# for sequence_label_range_index in sequence_label_ranges_index:
					# 	activity_start = int(sequence_label_range_index[0])
					# 	activity_end = int(sequence_label_range_index[1])
					# 	activity_id = "{}-{}-{}".format(activity_start, activity_end, sequence_label_range_index[2])
					# 	discrete_clean_sequence_for_action = discrete_clean_sequence[activity_start:activity_end]
					# 	discrete_clean_sequence_for_action_pd = pd.Series(discrete_clean_sequence_for_action)
					# 	distribution_parameters = discrete_clean_sequence_for_action_pd.value_counts(normalize=True, sort=False).to_dict()
					# 	for token in range(number_of_clusters):
					# 		if token not in distribution_parameters:
					# 			distribution_parameters[token] = 0
					# 	distribution_parameters = collections.OrderedDict(sorted(distribution_parameters.items(), key=lambda t: t[0]))
					# 	current_experiment["sequence"]["activities"][activity_id] = {}
					# 	current_experiment["sequence"]["activities"][activity_id]["distribution_parameters"] = distribution_parameters

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
