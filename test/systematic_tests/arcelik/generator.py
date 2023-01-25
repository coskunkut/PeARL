# batch experiment generator

import collections
import glob
import json
import math
import os
import sys

import numpy as np
import pandas as pd
from bkmeans import BKMeans
from more_itertools.recipes import grouper
from tqdm import tqdm

from pearl.generator import Multinomial
from pearl.tools.signal import decompose_signal


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
exp_type = batch_config["exp_type"]
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

# noise
injection_type = batch_config["preprocessing"]["noise"]["injection_type"]
if exp_type == "fixed_omega":
	start = batch_config["preprocessing"]["noise"]["noise_rates"]["start"]
	stop = batch_config["preprocessing"]["noise"]["noise_rates"]["stop"]
	step = batch_config["preprocessing"]["noise"]["noise_rates"]["step"]
	num = int(((stop - start) / step) + 1)
	noise_rate_list = np.linspace(start, stop, num)
elif exp_type == "changing_omega":
	noise_rate_list = [0] # dummy list to reuse noise injection code
	n_drift_type = list(batch_config["preprocessing"]["noise"]["noise_drift"].keys())[0]
	if n_drift_type == "sudden":
		n_drift_frequency = batch_config["preprocessing"]["noise"]["noise_drift"]["sudden"]["frequency"]
		n_drift_severity = batch_config["preprocessing"]["noise"]["noise_drift"]["sudden"]["severity"]
		n_drift_d = n_drift_frequency["d"]
		n_drift_min_diff, n_drift_max_diff = n_drift_severity["min"], n_drift_severity["max"]

total_number_of_runs = np.prod([
		number_of_runs,
		len(batch_config["data"]["motor_ids"]),
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
		current_rng = np.random.default_rng(current_seed)

		for raw_signal_ids, raw_signal_file_paths in raw_signal_file_paths_dict.items():

			for noise_rate in noise_rate_list:

				# prepare current experiment file
				current_experiment = {
					"experiment_info": {
						"type": exp_type,
						"seed": current_seed,
						"sequence_id": raw_signal_ids,
						"experiment_id": "",
					},
					"algorithms_to_run": algorithms_to_run,
					"parameters": {
						"slwe_e": {
							"lambda": batch_config["parameters"]["slwe_e"]["lambda"]
						},
						"slwe_t": {
							"lambda": batch_config["parameters"]["slwe_t"]["lambda"]
						},
						"pearl_a": {
							"eta_min": batch_config["parameters"]["pearl_a"]["eta_min"],
							"eta_max": batch_config["parameters"]["pearl_a"]["eta_max"],
							"rho_eta_mapping": batch_config["parameters"]["pearl_a"]["rho_eta_mapping"],
							"k": batch_config["parameters"]["pearl_a"]["k"],
							"L": batch_config["parameters"]["pearl_a"]["L"]
						},
						"pearl_e": {
							"eta_min": batch_config["parameters"]["pearl_e"]["eta_min"],
							"eta_max": batch_config["parameters"]["pearl_e"]["eta_max"],
							"rho_eta_mapping": batch_config["parameters"]["pearl_e"]["rho_eta_mapping"],
							"k": batch_config["parameters"]["pearl_e"]["k"],
							"L": batch_config["parameters"]["pearl_e"]["L"]
						}
					},
					"preprocessing": {
						"decomposition": batch_config["preprocessing"]["decomposition"],
						"noise": {}
					},
					"sequence": {
						"parameters": {},
						# "raw_clean": [],
						"discrete_clean": [],
						"discrete_noisy": [],
						"change_points": [],
						"hamming_distance": -1,
						"effective_length": -1,
						"effective_noise_rate": -1
					},
					"visualization": batch_config["visualization"]
				}

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
				number_of_regimes = index + 1
				regime_length = ssi_end - ssi_start
				# current_experiment["sequence"]["raw_clean"] = X_clean
				# X_clean = np.array(X_clean).reshape(-1, 1)
				X_clean = np.array(X_clean)
				# print(X_clean.shape)

				# inject white noise
				observation_count = X_clean.shape[0]
				bounds = [np.min(X_clean), np.max(X_clean)]
				X_noisy = np.array(X_clean, copy=True)
				noise_flags = np.zeros([observation_count], dtype=int)
				if exp_type == "fixed_omega":
					noisy_instance_count = int(np.ceil(observation_count * noise_rate))
					noisy_instance_indices = current_rng.choice(observation_count, noisy_instance_count, replace=False)
					white_noise = np.zeros(noisy_instance_count)
					white_noise += current_rng.uniform(bounds[0], bounds[1], noisy_instance_count)
					if batch_config["preprocessing"]["noise"]["injection_type"] == "add":
						X_noisy[noisy_instance_indices] += white_noise
					elif batch_config["preprocessing"]["noise"]["injection_type"] == "replace":
						X_noisy[noisy_instance_indices] = white_noise
					noise_flags[noisy_instance_indices] += 1
					current_experiment["preprocessing"]["noise"]["noise_rate"] = noise_rate
				elif exp_type == "changing_omega":
					# generate noise regimes
					alp_car = batch_config["preprocessing"]["clustering"]["parameters"]["n_clusters"]
					multinomial_ins = Multinomial(alp_car, current_seed)
					noise_regimes = multinomial_ins.generate_noise_regimes(
						sequence_length = observation_count,
						regime_length = n_drift_d,
						delta_min = n_drift_min_diff,
						delta_max = n_drift_max_diff,
						omega_min = 0.2,
						omega_max = 0.75
					)
					for noise_regime_bounds, noise_regime_omega in noise_regimes.items():
						start, end = [int(bound) for bound in noise_regime_bounds.split("_")]
						noise_regime_length = end - start
						noisy_instance_count = int(np.ceil(noise_regime_length * noise_regime_omega))
						noisy_instance_indices = current_rng.choice(
							noise_regime_length,
							noisy_instance_count,
							replace = False
						) + start
						noise_flags[noisy_instance_indices] += 1
						white_noise = np.zeros(noisy_instance_count)
						white_noise += current_rng.uniform(bounds[0], bounds[1], noisy_instance_count)
						# X_noisy[noisy_instance_indices, :] += white_noise
						if batch_config["preprocessing"]["noise"]["injection_type"] == "add":
							X_noisy[noisy_instance_indices] += white_noise
						elif batch_config["preprocessing"]["noise"]["injection_type"] == "replace":
							X_noisy[noisy_instance_indices] = white_noise
						# print(start, end, noise_regime_omega, noise_regime_length, noisy_instance_count, noisy_instance_indices)
					# print(X_clean[0:300]-X_noisy[0:300])
					current_experiment["preprocessing"]["noise"] = {
						"injection_type": injection_type,
						"drift_type": n_drift_type,
						"drift_params": {
							"frequency": n_drift_frequency,
							"severity": n_drift_severity
						},
						"noise_regimes": noise_regimes
					}

				fragment_length = batch_config["preprocessing"]["decomposition"]["fragment_length"]
				X_clean_decomposed = decompose_signal(X_clean, fragment_length, True)
				
				# print(X_clean_decomposed[0:10])

				X_clean_grouped = decompose_signal(X_clean, fragment_length, False)
				X_noisy_grouped = decompose_signal(X_noisy, fragment_length, False)

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

					current_experiment["preprocessing"]["clustering_config"] = clustering_config
					current_experiment["preprocessing"]["clustering_result"] = {
						"number_of_clusters": number_of_clusters,
						"cluster_reps": np.array(clusterer.cluster_centers_).tolist(),
					}

					# take portions of signal
					augmented_regime_length = int(regime_length / fragment_length)
					for index, signal_portion in enumerate(grouper(discrete_clean_sequence, augmented_regime_length)):
						discrete_clean_sequence_pd = pd.Series(signal_portion)
						distribution_parameters = discrete_clean_sequence_pd.value_counts(normalize=True, sort=False).to_dict()
						for token in range(number_of_clusters):
							if token not in distribution_parameters:
								distribution_parameters[token] = 0
						distribution_parameters = collections.OrderedDict(sorted(distribution_parameters.items(), key=lambda t: t[0]))
						current_experiment["sequence"]["parameters"][index] = distribution_parameters

					change_points = augmented_regime_length * np.arange(1, number_of_regimes, 1, dtype=int)
					current_experiment["sequence"]["change_points"] = change_points.tolist()

					experiment_name = "{}_lrc{}_orc{}".format(raw_signal_ids, run_count, experiment_file_count)
					current_experiment["experiment_info"]["name"] = experiment_name
					experiment_file_path = "{}/{}.json".format(experiment_files_folder_path, experiment_name)
					dump_json(current_experiment, experiment_file_path)
					experiment_file_count += 1
					pbar.update()
