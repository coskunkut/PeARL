# batch experiment runner

import json
import multiprocessing as mp
import os
import sys

from natsort import natsorted
from tqdm import tqdm

from pearl.estimator import SLWE, PeARL
from pearl.tools.encoder import NumpyEncoder


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
	sequence = current_experiment["sequence_info"]["sequence"]
	alp_car = current_experiment["sequence_info"]["alphabet_cardinality"]
	algorithms_to_run = current_experiment["algorithms_to_run"]

	# slwe_e
	if "slwe_e" in algorithms_to_run:
		slwe_e_lambda = current_experiment["parameters"]["slwe_e"]["lambda"]
		slwe_e_ins = SLWE(alp_car, slwe_e_lambda)
		slwe_e_ins.process_sequence(sequence, progress=False)
		current_experiment["results"]["slwe_e"] = {}
		current_experiment["results"]["slwe_e"]["estimates"] = slwe_e_ins.estimates
		current_experiment["results"]["slwe_e"]["symbol_index_map"] = slwe_e_ins.symbol_index_map

	# slwe_t
	if "slwe_t" in algorithms_to_run:
		slwe_t_lambda = current_experiment["parameters"]["slwe_t"]["lambda"]
		slwe_t_ins = SLWE(alp_car, slwe_t_lambda)
		slwe_t_ins.process_sequence(sequence, progress=False)
		current_experiment["results"]["slwe_t"] = {}
		current_experiment["results"]["slwe_t"]["estimates"] = slwe_t_ins.estimates
		current_experiment["results"]["slwe_t"]["symbol_index_map"] = slwe_t_ins.symbol_index_map

	# pearl_a
	if "pearl_a" in algorithms_to_run:
		pearl_a_ins = PeARL(
			alphabet_cardinality=alp_car,
			eta_max=current_experiment["parameters"]["pearl_a"]["eta_max"],
			eta_min=current_experiment["parameters"]["pearl_a"]["eta_min"],
			k=current_experiment["parameters"]["pearl_a"]["k"],
			L=current_experiment["parameters"]["pearl_a"]["L"],
			rho_eta_mapping=current_experiment["parameters"]["pearl_a"]["rho_eta_mapping"]
		)
		pearl_a_ins.process_sequence(sequence, progress=False)
		current_experiment["results"]["pearl_a"] = {}
		current_experiment["results"]["pearl_a"]["estimates"] = pearl_a_ins.estimates
		current_experiment["results"]["pearl_a"]["symbol_index_map"] = pearl_a_ins.symbol_index_map

	# pearl_e
	if "pearl_e" in algorithms_to_run:
		pearl_e_ins = PeARL(
			alphabet_cardinality=alp_car,
			eta_max=current_experiment["parameters"]["pearl_e"]["eta_max"],
			eta_min=current_experiment["parameters"]["pearl_e"]["eta_min"],
			k=current_experiment["parameters"]["pearl_e"]["k"],
			L=current_experiment["parameters"]["pearl_e"]["L"],
			rho_eta_mapping=current_experiment["parameters"]["pearl_e"]["rho_eta_mapping"]
		)
		pearl_e_ins.process_sequence(sequence, progress=False)
		current_experiment["results"]["pearl_e"] = {}
		current_experiment["results"]["pearl_e"]["estimates"] = pearl_e_ins.estimates
		current_experiment["results"]["pearl_e"]["rho"] = pearl_e_ins.rho
		current_experiment["results"]["pearl_e"]["symbol_index_map"] = pearl_e_ins.symbol_index_map

	# append results to experiment file
	with open(experiment_file_path, "w") as current_experiment_file:
		json.dump(current_experiment, current_experiment_file, indent=4, cls=NumpyEncoder)

# get experiment id
if len(sys.argv) == 2:
	batch_id = sys.argv[1]
else:
	sys.exit()

# get a list of files in experiment folder
root_folder_path = os.path.abspath(os.curdir)
test_folder_path = os.path.join(root_folder_path, "test", "systematic_tests", "multinomial")
batch_output_folder_path = os.path.join(test_folder_path, "batch_output", batch_id)
if not os.path.exists(batch_output_folder_path):
	print("Batch output folder not found.")
	sys.exit()
experiment_files_folder_path = os.path.join(batch_output_folder_path, "experiment_files")
experiment_files = natsorted(os.listdir(experiment_files_folder_path), reverse=False)
figures_folder_path = os.path.join(batch_output_folder_path, "figures")

# map each experiment file to a process
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
