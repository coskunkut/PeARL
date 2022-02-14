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
from more_itertools.recipes import grouper
from natsort import natsorted
from scipy import signal
from sklearn import cluster, metrics, preprocessing
from tqdm import tqdm
from pbnm.tools import decompose_signal


def dump_json(current_experiment, experiment_file_path):
	with open(experiment_file_path, "w") as json_file:
		json.dump(current_experiment, json_file, indent=4)

def id_to_path(info_table, motor_id, feature):
	filtered_info_table = info_table.loc[info_table["Anonim_ID"] == motor_id]
	return filtered_info_table[feature].values[0]
	print(filtered_info_table)

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

raw_signal_file_paths_dict = {}
for motor_id_list in batch_config["data"]["motor_ids"]:
	raw_signal_file_names = [id_to_path(info_table, id, batch_config["data"]["feature"]) for id in motor_id_list]
	raw_signal_file_paths = [os.path.join(data_folder_path, "samples", fn) for fn in raw_signal_file_names]
	raw_signal_file_paths_dict["_".join([str(e) for e in motor_id_list])] = raw_signal_file_paths

# filter data
# filtered_info_table = info_table.copy()
# filtered_info_table = filtered_info_table.loc[filtered_info_table["Anonim_ID"].isin(batch_config["data"]["motor_ids"])]

# get raw signals
# raw_signal_file_names = filtered_info_table[batch_config["data"]["feature"]].tolist()
# raw_signal_file_paths = [os.path.join(data_folder_path, "samples", fn) for fn in raw_signal_file_names]

# clustering parameters
clustering_config = batch_config["preprocessing"]["clustering"]

noise_rate_list = batch_config["preprocessing"]["noise_injection"]["noise_rate"]

total_number_of_runs = np.prod([
		number_of_runs,
		len(batch_config["data"]["motor_ids"]),
		len(noise_rate_list)
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

		for raw_signal_ids, raw_signal_file_paths in raw_signal_file_paths_dict.items():

			for noise_rate in noise_rate_list:

				# prepare current experiment file
				current_experiment = {}
				current_experiment["algorithms_to_run"] = algorithms_to_run
				current_experiment["experiment_info"] = {}
				current_experiment["experiment_info"]["sequence_id"] = raw_signal_ids
				current_experiment["experiment_info"]["experiment_id"] = ""
				current_experiment["experiment_info"]["seed"] = current_seed
				current_experiment["preprocessing_info"] = {}
				current_experiment["parameters"] = {}
				current_experiment["sequence"] = {
					"parameters": {},
					"raw_clean": [],
					"discrete_clean": [],
					"discrete_noisy": [],
					"hamming_distance": -1,
					"effective_length": -1,
					"effective_noise_rate": -1
				}
				# current_experiment["sequence"]["parameters"] = {}

				# read raw data
				X_clean = []
				for index, raw_signal_file_path in enumerate(raw_signal_file_paths):
					X_clean_partial = np.fromfile(raw_signal_file_path, dtype=np.float32, count=-1, sep="").reshape(-1, 1).ravel().tolist()
					ssi_start, ssi_end = [math.ceil(t * sampling_rate) for t in batch_config["data"]["steady_state_time"]]
					X_clean_partial = X_clean_partial[ssi_start:ssi_end]
					# if index == 1:
						# X_clean_partial = np.multiply(X_clean_partial, 2)
					# current_rng.shuffle(X_clean_partial)
					X_clean.extend(X_clean_partial)
				regime_length = ssi_end - ssi_start
				current_experiment["sequence"]["raw_clean"] = X_clean
				# X_clean = np.array(X_clean).reshape(-1, 1)
				X_clean = np.array(X_clean)
				# print(X_clean.shape)

				# inject white noise
				observation_count = X_clean.shape[0]
				bounds = [np.min(X_clean), np.max(X_clean)]
				noisy_instance_count = int(np.ceil(observation_count * noise_rate))
				noisy_instance_indices = current_rng.choice(observation_count, noisy_instance_count, replace=False)
				white_noise = np.zeros(noisy_instance_count)
				white_noise += current_rng.uniform(bounds[0], bounds[1], noisy_instance_count)
				X_noisy = np.array(X_clean, copy=True)
				if batch_config["preprocessing"]["noise_injection"]["injection_policy"] == "add":
					X_noisy[noisy_instance_indices] += white_noise
				elif batch_config["preprocessing"]["noise_injection"]["injection_policy"] == "replace":
					X_noisy[noisy_instance_indices] = white_noise
				noise_flags = np.zeros([observation_count], dtype=int)
				noise_flags[noisy_instance_indices] += 1

				piece_length = 10
				X_clean_decomposed = decompose_signal(X_clean, piece_length, True)
				
				# print(X_clean_decomposed[0:10])

				X_clean_grouped = decompose_signal(X_clean, piece_length, False)
				X_noisy_grouped = decompose_signal(X_noisy, piece_length, False)

				current_experiment["preprocessing_info"]["noise_injection"] = {
					"injection_policy": batch_config["preprocessing"]["noise_injection"]["injection_policy"],
					"noise_rate": noise_rate
				}

				if clustering_config["algorithm"] == "bkmeans":
					number_of_clusters = clustering_config["parameters"]["n_clusters"]
					current_config = {
						"algorithm": "bkmeans",
						"parameters": {
							"n_clusters": number_of_clusters,
							"m": 5
						}
					}
					clusterer = BKMeans(**current_config["parameters"], random_state=0)
					clusterer.fit(X_clean_decomposed)

					discrete_clean_sequence = clusterer.predict(X_clean_grouped)
					discrete_noisy_sequence = clusterer.predict(X_noisy_grouped)

					current_experiment["sequence"]["discrete_clean"] = [str(i) for i in discrete_clean_sequence]
					current_experiment["sequence"]["discrete_noisy"] = [str(i) for i in discrete_noisy_sequence]

					hamming_distance = np.sum(discrete_clean_sequence != discrete_noisy_sequence).tolist()
					current_experiment["sequence"]["hamming_distance"] = hamming_distance

					effective_length = len(discrete_clean_sequence)
					current_experiment["sequence"]["effective_length"] = effective_length

					effective_noise_rate = hamming_distance/effective_length
					current_experiment["sequence"]["effective_noise_rate"] = effective_noise_rate

					current_experiment["preprocessing_info"]["clustering_config"] = clustering_config
					current_experiment["preprocessing_info"]["clustering_result"] = {
						"number_of_clusters": number_of_clusters,
						"cluster_reps": np.array(clusterer.cluster_centers_).tolist(),
					}

					# take portions of signal
					augmented_regime_length = int(regime_length / piece_length)
					for index, signal_portion in enumerate(grouper(discrete_clean_sequence, augmented_regime_length)):
						discrete_clean_sequence_pd = pd.Series(signal_portion)
						distribution_parameters = discrete_clean_sequence_pd.value_counts(normalize=True, sort=False).to_dict()
						for token in range(number_of_clusters):
							if token not in distribution_parameters:
								distribution_parameters[token] = 0
						distribution_parameters = collections.OrderedDict(sorted(distribution_parameters.items(), key=lambda t: t[0]))
						current_experiment["sequence"]["parameters"][index] = distribution_parameters

					# algs. and parameters
					current_experiment["parameters"] = {
						"slwe_e": {
							"lambda": batch_config["parameters"]["slwe_e"]["lambda"]
						},
						"slwe_t": {
							"lambda": batch_config["parameters"]["slwe_t"]["lambda"]
						},
						"slwe_p_a": {
							"eta_min": batch_config["parameters"]["slwe_p_a"]["eta_min"],
							"eta_max": batch_config["parameters"]["slwe_p_a"]["eta_max"],
							"rho_eta_mapping": batch_config["parameters"]["slwe_p_a"]["rho_eta_mapping"],
							"k": batch_config["parameters"]["slwe_p_a"]["k"],
							"L": batch_config["parameters"]["slwe_p_a"]["L"]
						},
						"slwe_p_e": {
							"eta_min": batch_config["parameters"]["slwe_p_e"]["eta_min"],
							"eta_max": batch_config["parameters"]["slwe_p_e"]["eta_max"],
							"rho_eta_mapping": batch_config["parameters"]["slwe_p_e"]["rho_eta_mapping"],
							"k": batch_config["parameters"]["slwe_p_e"]["k"],
							"L": batch_config["parameters"]["slwe_p_e"]["L"]
						}
					}

					experiment_name = "exp_{}".format(experiment_file_count)
					current_experiment["experiment_info"]["name"] = experiment_name
					experiment_file_path = "{}/{}.json".format(experiment_files_folder_path, experiment_name)
					dump_json(current_experiment, experiment_file_path)
					experiment_file_count += 1
					pbar.update()
