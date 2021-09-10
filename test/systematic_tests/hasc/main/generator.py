# batch experiment generator

import json
import os
import glob
import sys
import numpy as np
import pandas as pd
import collections
from scipy import signal
from natsort import natsorted
from tqdm import tqdm
from datetime import datetime
from sklearn import preprocessing, cluster, metrics
from bkmeans import BKMeans

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
test_folder_path = os.path.join(root_folder_path, "test", "systematic_tests", "hasc", "main")
data_folder_path = os.path.join(root_folder_path, "data", "hasc")

clean_sequence_parent_folder_path = os.path.join(data_folder_path, "0_sequence")

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
experiment_file_count = 1

# sequence paths
clean_sequence_data_paths = natsorted(glob.glob("{}/*/HASC*.csv".format(clean_sequence_parent_folder_path)), reverse=False)

# sequences
sequence_list = batch_config["sequences"]

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
		len(sequence_list) if sequence_list[0] != "all" else 18,
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

		for clean_sequence_data_path in clean_sequence_data_paths:

			for noise_rate in noise_rate_list:

				# get person and experiment id
				person_id = clean_sequence_data_path.rsplit("/", 2)[1]
				sequence_data_id = clean_sequence_data_path.rsplit("/", 1)[1].split(".")[0]

				if sequence_list[0] != "all" and sequence_data_id not in sequence_list:
					continue

				# prepare current experiment file
				current_experiment = {}
				current_experiment["algorithms_to_run"] = algorithms_to_run
				current_experiment["experiment_info"] = {}
				current_experiment["experiment_info"]["person_id"] = person_id
				current_experiment["experiment_info"]["sequence_id"] = sequence_data_id
				current_experiment["experiment_info"]["experiment_id"] = ""
				current_experiment["experiment_info"]["seed"] = current_seed
				current_experiment["preprocessing_info"] = {}
				current_experiment["parameters"] = {}
				current_experiment["sequence"] = {}
				current_experiment["sequence"]["activities"] = {}

				# read raw data (including time stamps)
				raw_data = np.genfromtxt(clean_sequence_data_path, usecols=(0, 1, 2, 3), delimiter=",")

				# downsample
				# downsample_rate = 3
				# raw_data = raw_data[::downsample_rate,:]

				# downsampling alternative, applies a low-pass filter first
				# raw_data = signal.decimate(raw_data, q=3, n=60, ftype="fir", axis=0)

				time_stamps = raw_data[:, 0]
				X_clean = raw_data[:, 1:]

				# round time stamps to make it easier to compare with label ranges
				sequence_times = np.round(time_stamps,3)

				# apply normalization
				# X = preprocessing.normalize(X)

				# label range file path
				sequence_label_ranges_time_path = clean_sequence_data_path.replace("csv", "label")

				observation_count = X_clean.shape[0]
				mins = np.min(X_clean, axis=0)
				maxs = np.max(X_clean, axis=0)
				bounds = list(zip(mins, maxs))
				noisy_instance_count = int(np.ceil(observation_count * noise_rate))
				noisy_instance_indices = current_rng.choice(observation_count, noisy_instance_count, replace=False)
				white_noise = np.zeros([noisy_instance_count, 3])
				for dim in range(3):
					white_noise[:, dim] += current_rng.uniform(bounds[dim][0], bounds[dim][1], noisy_instance_count)
				X_noisy = np.array(X_clean, copy=True)  
				X_noisy[noisy_instance_indices, :] += white_noise
				noise_flags = np.zeros([observation_count], dtype=int)
				noise_flags[noisy_instance_indices] += 1

				current_experiment["preprocessing_info"]["noise_injection"] = {
					"noise_rate": noise_rate
				}

				if clustering_config["algorithm"] == "bkmeans":
					number_of_clusters = clustering_config["parameters"]["n_clusters"][str(sequence_data_id)]
					current_config = {
						"algorithm": "bkmeans",
						"parameters": {
							"n_clusters": number_of_clusters,
							"m": 4,
							"theta": 1.1
						}
					}
					clusterer = BKMeans(**current_config["parameters"], random_state=0)
					clusterer.fit(X_clean)

					discrete_clean_sequence = clusterer.predict(X_clean)
					discrete_noisy_sequence = clusterer.predict(X_noisy)

					current_experiment["sequence"]["discrete_clean"] = discrete_clean_sequence.tolist()
					current_experiment["sequence"]["discrete_noisy"] = discrete_noisy_sequence.tolist()

					current_experiment["preprocessing_info"]["clustering_config"] = clustering_config
					current_experiment["preprocessing_info"]["clustering_result"] = {}
					current_experiment["preprocessing_info"]["clustering_result"]["number_of_clusters"] = number_of_clusters
					# current_experiment["preprocessing_info"]["clustering_result"]["number_of_noise_points"] = number_of_noise_points
					# current_experiment["preprocessing_info"]["clustering_result"]["dbs"] = dbs
					# current_experiment["preprocessing_info"]["clustering_result"]["chi"] = chi

					# obtain sequence labels as both time and index and remove noise points
					sequence_label_ranges_time_lines = open(sequence_label_ranges_time_path, "r").read().splitlines()
					sequence_label_ranges_index = np.array([]).reshape((0, 3))
					change_points = np.array([], dtype=int)
					# sequence_clustered_no_noise = np.array([], dtype=int)
					for line in sequence_label_ranges_time_lines[1:]:
						line_splitted = line.split(",")
						activity_start = np.round(float(line_splitted[0]),3)
						activity_end = np.round(float(line_splitted[1]),3)
						activity_name = line_splitted[2]
						activity_start_index = np.abs(sequence_times-activity_start).argmin()
						activity_end_index = np.abs(sequence_times-activity_end).argmin()
					# 	subsequence = sequence_clustered[activity_start_index:activity_end_index]
					# 	noise_indices = np.where(subsequence == -1)[0]
					# 	number_of_noise_points_in_subsequence = len(noise_indices)
					# 	subsequence_clustered_no_noise = np.delete(subsequence, noise_indices)
					# 	rng.shuffle(subsequence_clustered_no_noise)
					# 	activity_start_index = len(sequence_clustered_no_noise)
					# 	sequence_clustered_no_noise = np.append(sequence_clustered_no_noise, subsequence_clustered_no_noise)
					# 	activity_end_index = len(sequence_clustered_no_noise)
						change_points = np.append(change_points, activity_end_index)
						sequence_label_ranges_index = np.vstack((sequence_label_ranges_index, np.array([activity_start_index, activity_end_index, activity_name])))

					current_experiment["sequence"]["change_points"] = change_points.tolist()
					# current_experiment["sequence"]["clustered_sequence"] = sequence_clustered_no_noise.tolist()

					# calculate distribution parameters for each action in the sequence
					for sequence_label_range_index in sequence_label_ranges_index:
						activity_start = int(sequence_label_range_index[0])
						activity_end = int(sequence_label_range_index[1])
						activity_id = "{}-{}-{}".format(activity_start, activity_end, sequence_label_range_index[2])
						discrete_clean_sequence_for_action = discrete_clean_sequence[activity_start:activity_end]
						discrete_clean_sequence_for_action_pd = pd.Series(discrete_clean_sequence_for_action)
						distribution_parameters = discrete_clean_sequence_for_action_pd.value_counts(normalize=True, sort=False).to_dict()
						for token in range(number_of_clusters):
							if token not in distribution_parameters:
								distribution_parameters[token] = 0
						distribution_parameters = collections.OrderedDict(sorted(distribution_parameters.items(), key=lambda t: t[0]))
						current_experiment["sequence"]["activities"][activity_id] = {}
						current_experiment["sequence"]["activities"][activity_id]["distribution_parameters"] = distribution_parameters

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
