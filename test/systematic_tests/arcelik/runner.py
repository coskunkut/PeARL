# batch experiment runner

import json
import multiprocessing as mp
import os
import sys

import numpy as np
from natsort import natsorted
from pbnm.estimator import SLWE, SLWE_P
from tqdm import tqdm


# each process calls this function to run a given experiment
def run_experiment(experiment_file):

	# read experiment file as JSON
	experiment_file_path = "{}/{}".format(experiment_files_folder_path, experiment_file)
	with open(experiment_file_path) as current_experiment_file:
		current_experiment = json.load(current_experiment_file)

	# skip if already processed
	# if "results" in current_experiment:
		# return

	current_experiment["results"] = {}

	# read parameters
	sequence = current_experiment["sequence"]["discrete_noisy"]
	alp_car = current_experiment["preprocessing_info"]["clustering_result"]["number_of_clusters"]
	algorithms_to_run = current_experiment["algorithms_to_run"]

	# slwe_e
	if "slwe_e" in algorithms_to_run:
		slwe_e_lambda = current_experiment["parameters"]["slwe_e"]["lambda"]
		slwe_e_ins = SLWE(alp_car, slwe_e_lambda)
		slwe_e_ins.process_sequence(sequence, progress=False)
		current_experiment["results"]["slwe_e"] = {}
		current_experiment["results"]["slwe_e"]["estimates"] = slwe_e_ins.estimates
		current_experiment["results"]["slwe_e"]["token_index_map"] = slwe_e_ins.token_index_map

	# slwe_t
	if "slwe_t" in algorithms_to_run:
		slwe_t_lambda = current_experiment["parameters"]["slwe_t"]["lambda"]
		slwe_t_ins = SLWE(alp_car, slwe_t_lambda)
		slwe_t_ins.process_sequence(sequence, progress=False)
		current_experiment["results"]["slwe_t"] = {}
		current_experiment["results"]["slwe_t"]["estimates"] = slwe_t_ins.estimates
		current_experiment["results"]["slwe_t"]["token_index_map"] = slwe_t_ins.token_index_map

	# slwe_p
	if "slwe_p" in algorithms_to_run:
		slwe_p_eta_min = current_experiment["parameters"]["slwe_p"]["eta_min"]
		slwe_p_eta_max = current_experiment["parameters"]["slwe_p"]["eta_max"]
		slwe_p_k = current_experiment["parameters"]["slwe_p"]["k"]
		slwe_p_L = current_experiment["parameters"]["slwe_p"]["L"]
		slwe_p_ins = SLWE_P(alp_car, "event_linear")
		slwe_p_ins.eta_min = slwe_p_eta_min
		slwe_p_ins.eta_max = slwe_p_eta_max
		slwe_p_ins.k = slwe_p_k
		slwe_p_ins.L = slwe_p_L
		slwe_p_ins.process_sequence(sequence, progress=False)
		current_experiment["results"]["slwe_p"] = {}
		current_experiment["results"]["slwe_p"]["estimates"] = slwe_p_ins.estimates
		current_experiment["results"]["slwe_p"]["token_index_map"] = slwe_p_ins.token_index_map

	# append results to experiment file
	with open(experiment_file_path, "w") as current_experiment_file:
		json.dump(current_experiment, current_experiment_file, indent=4)

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
figures_folder_path = os.path.join(batch_output_folder_path, "figures")

# map each experimet file to a process
pool = mp.Pool()
# pool.map(run_experiment, experiment_files)
r = list(tqdm(
	pool.imap(run_experiment, experiment_files),
	total = len(experiment_files),
	desc = "run".ljust(10),
	bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}',
	ascii = True
))
pool.close()
