# batch or single experiment result visualizer

import csv
import glob
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cycler import cycler
from natsort import natsorted
from pbnm.tools import plot_tools
from tqdm import tqdm

def nr_mae_fig(
		noise_rate_list,
		slwe_e_mae_list,
		slwe_e_lambda,
		slwe_t_mae_list,
		slwe_t_lambda,
		slwe_p_mae_list,
		target_doc_props,
		fig_path
	):

	# generate figure for current experiment (estimates)
	fig, ax = plt.subplots(
		nrows = 1,
		ncols = 1,
		figsize = plot_tools.set_size(
			width = target_doc_props["textwidth"],
			aspect_ratio = target_doc_props["ar"],
			fraction = target_doc_props["fr"]
		)
	)
	fig.set_tight_layout(True)

	ax.plot(
		noise_rate_list,
		slwe_p_mae_list,
		label = "SLWE-P",
		color = "black",
		alpha = 0.9,
		linewidth = 1,
		zorder = 10
	)
	ax.plot(
		noise_rate_list,
		slwe_e_mae_list,
		label = "SLWE ($\eta={:.2f}$)".format(1-slwe_e_lambda),
		color = "black",
		alpha = 0.9,
		linewidth = 1,
		zorder = 10
	)
	ax.plot(
		noise_rate_list,
		slwe_t_mae_list,
		label = "SLWE ($\eta={:.2f}$)".format(1-slwe_t_lambda),
		color = "black",
		alpha = 0.9,
		linewidth = 1,
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
run_logs_path = os.path.join(batch_output_folder_path, "run_logs.csv")

# get a sample experiment file to obtain visualization config
sample_experiment_file_path = "{}/{}".format(experiment_files_folder_path, experiment_files[0])
with open(sample_experiment_file_path) as sample_experiment_file:
	sample_experiment = json.load(sample_experiment_file)
	plots = sample_experiment["visualization"]["plots"]

run_logs_path = os.path.join(batch_output_folder_path, "run_logs.csv")
result_table_path = os.path.join(batch_output_folder_path, "result_table.csv")

# construct figure folder paths
figures_folder_path = plot_tools.construct_figure_folder_path(batch_output_folder_path, "figures", remove_existing=True)

for plot_type in plots:

	# initialize directory
	current_figures_folder_path = plot_tools.construct_figure_folder_path(
		figures_folder_path,
		plot_type,
		remove_existing=True
	)

	slwe_e_lambda = 0.9
	slwe_t_lambda = 0.99

	if plot_type == "mae_omega":
		target_doc_props = {
			"vanilla_paper": {
				"textwidth": "vanilla_paper",
				"fr": 0.5,
				"ar": 1.25,
				"tex_config": "latex"
			}
		}
		target_doc = "vanilla_paper"
		plt.rcParams.update(plot_tools.config[target_doc_props[target_doc]["tex_config"]])

		raw_table = pd.read_csv(result_table_path, sep=",", index_col=False)

		sequence_id_list = raw_table["sequence_id"].unique()
		for sequence_id in sequence_id_list:
			filtered_table = raw_table.loc[(raw_table["sequence_id"] == sequence_id)]
			# print(filtered_table)
			noise_rates = filtered_table["noise_rate"].values
			slwe_e_mae_list = filtered_table["SLWE (E)"].values
			slwe_t_mae_list = filtered_table["SLWE (T)"].values
			slwe_p_a_mae_list = filtered_table["SLWE-PA"].values
			slwe_p_e_mae_list = filtered_table["SLWE-PE"].values

			# generate figure for current experiment (estimates)
			fig, ax = plt.subplots(
				nrows = 1,
				ncols = 1,
				figsize = plot_tools.set_size(
					width = target_doc_props[target_doc]["textwidth"],
					aspect_ratio = target_doc_props[target_doc]["ar"],
					fraction = target_doc_props[target_doc]["fr"]
				)
			)
			fig.set_tight_layout(True)

			ax.plot(
				noise_rates,
				slwe_p_a_mae_list,
				label = "SLWE-PA",
				color = "darkgreen",
				alpha = 0.9,
				linewidth = 1,
				linestyle = "solid",
				zorder = 10
			)
			ax.plot(
				noise_rates,
				slwe_p_e_mae_list,
				label = "SLWE-PE",
				color = "maroon",
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

			# fig_name = "k{}_L{}".format(slwe_p_a_k, slwe_p_a_L)
			fig_name = "{}".format(sequence_id)
			fig.savefig(
				"{}/{}.{}".format(current_figures_folder_path, fig_name, "pdf"),
				bbox_inches = "tight",
				pad_inches = 0.05
			)
			plt.close()

	elif plot_type == "ae_time_omega":
		target_doc_props = {
			"vanilla_paper": {
				"textwidth": "vanilla_paper",
				"fr": 0.5,
				"ar": 1.25,
				"tex_config": "latex"
			}
		}
		target_doc = "vanilla_paper"
		plt.rcParams.update(plot_tools.config[target_doc_props[target_doc]["tex_config"]])

		for experiment_file in experiment_files:
			# read experiment file as JSON
			experiment_file_path = "{}/{}".format(experiment_files_folder_path, experiment_file)
			with open(experiment_file_path) as current_experiment_file:
				current_experiment = json.load(current_experiment_file)

			# generate figure for current experiment (estimates)
			fig, ax1 = plt.subplots(
				nrows = 1,
				ncols = 1,
				figsize = plot_tools.set_size(
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
				current_experiment["results"]["slwe_e"]["ae"],
				label = "SLWE ($\eta={:.2f}$)".format(1-slwe_e_lambda),
				color = "darkgray",
				alpha = 0.9,
				linewidth = 0.5,
				linestyle = "solid",
				zorder = 10
			)
			ax1.plot(
				current_experiment["results"]["slwe_t"]["ae"],
				label = "SLWE ($\eta={:.2f}$)".format(1-slwe_t_lambda),
				color = "darkgreen",
				alpha = 0.9,
				linewidth = 0.5,
				linestyle = "solid",
				zorder = 10
			)
			ax1.plot(
				current_experiment["results"]["slwe_p_e"]["ae"],
				label = "SLWE-PE",
				color = "black",
				alpha = 0.9,
				linewidth = 0.5,
				linestyle = "solid",
				zorder = 10
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
				"textwidth": "vanilla_paper",
				"fr": 0.5,
				"ar": 1.25,
				"tex_config": "latex"
			}
		}
		target_doc = "vanilla_paper"
		plt.rcParams.update(plot_tools.config[target_doc_props[target_doc]["tex_config"]])

		for experiment_file in experiment_files:

			# read experiment file as JSON
			experiment_file_path = "{}/{}".format(experiment_files_folder_path, experiment_file)
			with open(experiment_file_path) as current_experiment_file:
				current_experiment = json.load(current_experiment_file)

			# generate figure for current experiment (estimates)
			fig, ax1 = plt.subplots(
				nrows = 1,
				ncols = 1,
				figsize = plot_tools.set_size(
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
				np.cumsum(current_experiment["results"]["slwe_e"]["ae"]),
				label = "SLWE ($\eta={:.2f}$)".format(1-slwe_e_lambda),
				color = "black",
				alpha = 0.9,
				linewidth = 1.0,
				linestyle = "dashed",
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
				np.cumsum(current_experiment["results"]["slwe_p_e"]["ae"]),
				label = "SLWE-PE",
				color = "black",
				alpha = 0.9,
				linewidth = 1.0,
				linestyle = "solid",
				zorder = 10
			)

			p_change_points = current_experiment["sequence"]["change_points"]
			ax1.set_xticks(p_change_points)

			ax1.set_xlabel("Observation Count")
			ax1.set_ylabel("CAE")
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

exit()
# build propery dict for target documents
target_doc_props = {
	"paper_double": {
		"textwidth": "vanilla_paper",
		"fr": 0.5,
		"ar": 1.25,
		"tex_config": "latex"
	},
	"paper_single": {
		"textwidth": "vanilla_paper",
		"fr": 1,
		"ar": 2.25,
		"tex_config": "latex"
	}
}

# construct figure folder paths
figures_folder_path = plot_tools.construct_figure_folder_path(batch_output_folder_path, "figures", remove_existing=True)
nr_mae_figures_folder_path = plot_tools.construct_figure_folder_path(figures_folder_path, "nr_mae", remove_existing=True)
for target_doc in target_doc_props.keys():
	plot_tools.construct_figure_folder_path(nr_mae_figures_folder_path, target_doc, remove_existing=True)

raw_table =  pd.read_csv(run_logs_path, sep=",", index_col=False)
group_by_filter = [
	"sequence_id",
	"noise_rate"
]
grouped_table = raw_table.groupby(by=group_by_filter).mean().reset_index()

slwe_p_k_list = raw_table["slwe_p_k"].unique()
slwe_p_L_list = raw_table["slwe_p_L"].unique()

alp_car = raw_table["alp_car"].unique()[0]

# these matrices are for overall nr_mae fig
slwe_e_mae_lists = np.zeros((18, 21))
slwe_t_mae_lists = np.zeros((18, 21))
slwe_p_mae_lists = np.zeros((18, 21))

total_number_of_figs = len(target_doc_props.keys()) * len(sequence_ids) * len(slwe_p_k_list) * len(slwe_p_L_list)

with tqdm(
	total = total_number_of_figs,
	position = 0,
	leave = True,
	desc = "visualize".ljust(10),
	bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}',
	ascii = True
	) as pbar:

	for target_doc in target_doc_props.keys():

		# matplotlib config
		plt.rcParams.update(plot_tools.config[target_doc_props[target_doc]["tex_config"]])

		# different line styles for each curve
		linestyle_cycler = cycler("linestyle", ["solid", "dashed", "dotted", "dashdot"])
		plt.rc("axes", prop_cycle=linestyle_cycler)

		for index, sequence_id in enumerate(sequence_ids):

			for slwe_p_k in slwe_p_k_list:
				for slwe_p_L in slwe_p_L_list:

					filtered_table = grouped_table.loc[
						(grouped_table['sequence_id'] == sequence_id) &
						(grouped_table['slwe_p_k'] == slwe_p_k) &
						(grouped_table['slwe_p_L'] == slwe_p_L)
					]

					noise_rate_list = filtered_table["noise_rate"].unique()

					slwe_e_lambda = filtered_table["slwe_e_lambda"].unique()[0]
					slwe_e_mae_list = filtered_table["slwe_e_mae"].unique()

					slwe_t_lambda = filtered_table["slwe_t_lambda"].unique()[0]
					slwe_t_mae_list = filtered_table["slwe_t_mae"].unique()

					slwe_p_mae_list = filtered_table["slwe_p_mae"].unique()

					slwe_e_mae_lists[index] = slwe_e_mae_list
					slwe_t_mae_lists[index] = slwe_t_mae_list
					slwe_p_mae_lists[index] = slwe_p_mae_list

					fig_name = "{}".format(sequence_id)
					fig_path = "{}/{}/{}.{}".format(nr_mae_figures_folder_path, target_doc, fig_name, "pdf")

					nr_mae_fig(
						noise_rate_list,
						slwe_e_mae_list,
						slwe_e_lambda,
						slwe_t_mae_list,
						slwe_t_lambda,
						slwe_p_mae_list,
						target_doc_props[target_doc],
						fig_path
					)

					pbar.update()

# overall nr_mae
fig_path = "{}/{}/{}.{}".format(nr_mae_figures_folder_path, "paper_single", "HASC_ALL", "pdf")
nr_mae_fig(
	np.linspace(0,1,21),
	slwe_e_mae_lists.mean(0),
	0.9,
	slwe_t_mae_lists.mean(0),
	0.99,
	slwe_p_mae_lists.mean(0),
	target_doc_props["paper_single"],
	fig_path
)
