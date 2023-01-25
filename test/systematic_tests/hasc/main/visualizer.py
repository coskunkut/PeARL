# batch or single experiment result visualizer

import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from natsort import natsorted

from pearl.tools.visualizer import construct_figure_folder_path, get_matplotlib_config, set_size


def nr_mae_fig(
		noise_rates,
		slwe_e_mae_list,
		slwe_e_lambda,
		slwe_t_mae_list,
		slwe_t_lambda,
		pearl_a_mae_list,
		pearl_e_mae_list,
		target_doc_props,
		fig_path
	):

	# generate figure for current experiment (estimates)
	fig, ax = plt.subplots(
		nrows = 1,
		ncols = 1,
		figsize = set_size(
			width = target_doc_props["textwidth"],
			aspect_ratio = target_doc_props["ar"],
			fraction = target_doc_props["fr"]
		)
	)
	fig.set_tight_layout(True)

	ax.plot(
		noise_rates,
		pearl_a_mae_list,
		label = "PeARL with AP2LR",
		color = "steelblue",
		alpha = 0.9,
		linewidth = 1,
		linestyle = "solid",
		zorder = 10
	)
	ax.plot(
		noise_rates,
		pearl_e_mae_list,
		label = "PeARL with EP2LR",
		color = "seagreen",
		alpha = 0.9,
		linewidth = 1,
		linestyle = "solid",
		zorder = 10
	)
	ax.plot(
		noise_rates,
		slwe_e_mae_list,
		label = "SLWE ($\eta={:.2f}$)".format(1-slwe_e_lambda),
		color = "black",
		alpha = 0.9,
		linewidth = 1,
		linestyle = "dashed",
		zorder = 10
	)
	ax.plot(
		noise_rates,
		slwe_t_mae_list,
		label = "SLWE ($\eta={:.2f}$)".format(1-slwe_t_lambda),
		color = "black",
		alpha = 0.9,
		linewidth = 1,
		linestyle = "dotted",
		zorder = 10
	)

	ax.set_xlabel("Noise Rate ($\omega$)")
	# ax.set_xlim(0, 1)
	ax.set_xticks(np.linspace(0, 1, num=11))
	# ax.set_ylim(0, 0.4)
	# ax.set_ylim(0)
	ax.set_ylabel("MAE")
	ax.grid(linestyle="dashed", alpha=0.5)
	ax.legend(loc="upper left")

	fig.savefig(fig_path, bbox_inches="tight", pad_inches=0.05)
	plt.close()

# get batch id
sequence_ids = []
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

# get a sample experiment file to obtain visualization config
sample_experiment_file_path = "{}/{}".format(experiment_files_folder_path, experiment_files[0])
with open(sample_experiment_file_path) as sample_experiment_file:
	sample_experiment = json.load(sample_experiment_file)
	plots = sample_experiment["visualization"]["plots"]

run_logs_path = os.path.join(batch_output_folder_path, "run_logs.csv")
result_table_path = os.path.join(batch_output_folder_path, "result_table_hasc.csv")

# construct figure folder paths
figures_folder_path = construct_figure_folder_path(batch_output_folder_path, "figures", remove_existing=True)

for plot_type in plots:

	# initialize directory
	current_figures_folder_path = construct_figure_folder_path(
		figures_folder_path,
		plot_type,
		remove_existing=True
	)

	slwe_e_lambda = 0.9
	slwe_t_lambda = 0.99

	if plot_type == "mae_omega":
		target_doc_props = {
			"vanilla_paper": {
				"textwidth": "pat_reg",
				"fr": 0.5,
				"ar": 1.25
			}
		}
		target_doc = "vanilla_paper"
		plt.rcParams.update(get_matplotlib_config("generic_latex"))

		raw_table = pd.read_csv(result_table_path, sep=",", index_col=False)

		# these matrices are for overall nr_mae fig
		slwe_e_mae_lists = np.zeros((18, 21))
		slwe_t_mae_lists = np.zeros((18, 21))
		pearl_a_mae_lists = np.zeros((18, 21))
		pearl_e_mae_lists = np.zeros((18, 21))

		sequence_id_list = raw_table["sequence_id"].unique()
		for index, sequence_id in enumerate(sequence_id_list):
			
			filtered_table = raw_table.loc[(raw_table["sequence_id"] == sequence_id)]
			noise_rates = filtered_table["noise_rate"].values
			slwe_e_mae_list = filtered_table["SLWE (E)"].values
			slwe_t_mae_list = filtered_table["SLWE (T)"].values
			pearl_a_mae_list = filtered_table["PeARL-A"].values
			pearl_e_mae_list = filtered_table["PeARL-E"].values

			slwe_e_mae_lists[index] = slwe_e_mae_list
			slwe_t_mae_lists[index] = slwe_t_mae_list
			pearl_a_mae_lists[index] = pearl_a_mae_list
			pearl_e_mae_lists[index] = pearl_e_mae_list

			fig_name = "{}".format(sequence_id)
			fig_path = "{}/{}.{}".format(current_figures_folder_path, fig_name, "pdf")
			nr_mae_fig(
				noise_rates,
				slwe_e_mae_list,
				slwe_e_lambda,
				slwe_t_mae_list,
				slwe_t_lambda,
				pearl_a_mae_list,
				pearl_e_mae_list,
				target_doc_props[target_doc],
				fig_path
			)

		# overall nr_mae
		fig_path = "{}/{}.{}".format(current_figures_folder_path, "HASC_ALL", "pdf")
		nr_mae_fig(
			noise_rates,
			slwe_e_mae_lists.mean(0),
			slwe_e_lambda,
			slwe_t_mae_lists.mean(0),
			slwe_t_lambda,
			pearl_a_mae_lists.mean(0),
			pearl_e_mae_lists.mean(0),
			target_doc_props[target_doc],
			fig_path
		)

	elif plot_type == "ae_time_omega":
		target_doc_props = {
			"vanilla_paper": {
				"textwidth": "pat_reg",
				"fr": 0.5,
				"ar": 1.25
			}
		}
		target_doc = "vanilla_paper"
		plt.rcParams.update(get_matplotlib_config("generic_latex"))

		for experiment_file in experiment_files:
			# read experiment file as JSON
			experiment_file_path = "{}/{}".format(experiment_files_folder_path, experiment_file)
			with open(experiment_file_path) as current_experiment_file:
				current_experiment = json.load(current_experiment_file)

			# generate figure for current experiment (estimates)
			fig, ax1 = plt.subplots(
				nrows = 1,
				ncols = 1,
				figsize = set_size(
					width = target_doc_props[target_doc]["textwidth"],
					aspect_ratio = target_doc_props[target_doc]["ar"],
					fraction = target_doc_props[target_doc]["fr"]
				)
			)
			fig.set_tight_layout(True)

			ax2 = ax1.twinx()

			omega_all = []
			for start_end_str, omega in current_experiment["preprocessing"]["noise"]["noise_regimes"].items():
				start, end = [int(t) for t in start_end_str.split("_")]
				omega_all.extend(np.repeat(omega, end-start).tolist())

			ax2.plot(
				omega_all,
				label = "$\omega$",
				color = "maroon",
				alpha = 0.9,
				linewidth = 1,
				linestyle = "solid",
				zorder = 10
			)

			ax1.plot(
				current_experiment["results"]["pearl_e"]["ae"],
				label = "PeARL with EP2LR",
				color = "seagreen",
				alpha = 0.9,
				linewidth = 0.5,
				linestyle = "solid",
				zorder = 7
			)
			ax1.plot(
				current_experiment["results"]["slwe_t"]["ae"],
				label = "SLWE ($\eta={:.2f}$)".format(1-slwe_t_lambda),
				color = "black",
				alpha = 0.9,
				linewidth = 0.5,
				linestyle = "solid",
				zorder = 6
			)
			ax1.plot(
				current_experiment["results"]["slwe_e"]["ae"],
				label = "SLWE ($\eta={:.2f}$)".format(1-slwe_e_lambda),
				color = "darkgray",
				alpha = 0.9,
				linewidth = 0.5,
				linestyle = "solid",
				zorder = 5
			)

			p_change_points = current_experiment["sequence"]["change_points"]
			ax1.set_xticks(p_change_points)

			ax1.set_xlabel("Observation Count")
			ax1.set_ylabel("Absolute Error")
			ax2.set_ylabel("Noise Rate ($\omega$)")

			ax1.set_xlim(0, len(omega_all))
			ax2.set_ylim(0, 1)

			ax1.grid(linestyle="dashed", alpha=0.5)
			ax1.legend(loc="upper left").set_zorder(100)
			ax2.legend(loc="upper right").set_zorder(100)

			fig_name = "{}".format(current_experiment["experiment_info"]["name"])
			fig.savefig(
				"{}/{}.{}".format(current_figures_folder_path, fig_name, "pdf"),
				bbox_inches = "tight",
				pad_inches = 0.05
			)
			plt.close()

	elif plot_type == "cae_time_omega":
		target_doc_props = {
			"vanilla_paper": {
				"textwidth": "pat_reg",
				"fr": 0.5,
				"ar": 1.25
			}
		}
		target_doc = "vanilla_paper"
		plt.rcParams.update(get_matplotlib_config("generic_latex"))

		for experiment_file in experiment_files:

			# read experiment file as JSON
			experiment_file_path = "{}/{}".format(experiment_files_folder_path, experiment_file)
			with open(experiment_file_path) as current_experiment_file:
				current_experiment = json.load(current_experiment_file)

			# generate figure for current experiment (estimates)
			fig, ax1 = plt.subplots(
				nrows = 1,
				ncols = 1,
				figsize = set_size(
					width = target_doc_props[target_doc]["textwidth"],
					aspect_ratio = target_doc_props[target_doc]["ar"],
					fraction = target_doc_props[target_doc]["fr"]
				)
			)
			fig.set_tight_layout(True)

			ax2 = ax1.twinx()

			omega_all = []
			for start_end_str, omega in current_experiment["preprocessing"]["noise"]["noise_regimes"].items():
				start, end = [int(t) for t in start_end_str.split("_")]
				omega_all.extend(np.repeat(omega, end-start).tolist())

			ax2.plot(
				omega_all,
				label = "$\omega$",
				color = "maroon",
				alpha = 0.9,
				linewidth = 1,
				linestyle = "solid",
				zorder = 10
			)
			
			ax1.plot(
				np.cumsum(current_experiment["results"]["pearl_e"]["ae"]),
				# label = "PeARL with EP2LR",
				label = "PeARL",
				color = "seagreen",
				alpha = 0.9,
				linewidth = 1.0,
				linestyle = "solid",
				zorder = 10
			)
			ax1.plot(
				np.cumsum(current_experiment["results"]["slwe_t"]["ae"]),
				label = "SLWE ($\eta={:.2f}$)".format(1-slwe_t_lambda),
				color = "black",
				alpha = 0.9,
				linewidth = 1.0,
				linestyle = "dotted",
				zorder = 10
			)
			ax1.plot(
				np.cumsum(current_experiment["results"]["slwe_e"]["ae"]),
				label = "SLWE ($\eta={:.2f}$)".format(1-slwe_e_lambda),
				color = "black",
				alpha = 0.9,
				linewidth = 1.0,
				linestyle = "dashed",
				zorder = 10
			)
			
			p_change_points = current_experiment["sequence"]["change_points"]
			ax1.set_xticks(p_change_points)
			ax1.xaxis.set_tick_params(rotation=90)

			ax1.set_xlabel("Observation Count")
			ax1.set_ylabel("CAE")
			ax2.set_ylabel("Noise Rate ($\omega$)")

			ax1.set_xlim(0, len(omega_all))
			ax2.set_ylim(0, 1)

			ax1.grid(linestyle="dashed", alpha=0.5)
			ax1.legend(loc="upper left").set_zorder(100)
			ax2.legend(loc="lower right").set_zorder(100)

			ax1.set_zorder(2)
			ax2.set_zorder(1)
			ax1.patch.set_visible(False)

			fig_name = "{}".format(current_experiment["experiment_info"]["name"])
			fig.savefig(
				"{}/{}.{}".format(current_figures_folder_path, fig_name, "pdf"),
				bbox_inches = "tight",
				pad_inches = 0.05
			)
			plt.close()
