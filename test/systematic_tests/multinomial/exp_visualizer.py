# batch or single experiment result visualizer

import os
import sys
import glob
import json
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from natsort import natsorted
from tqdm import tqdm
from cycler import cycler

from pbnm.tools import plot_tools

# get experiment id
exp_list = []
if len(sys.argv) == 2:
	batch_id = sys.argv[1]
	exp_list = "all"
elif len(sys.argv) >= 3:
	batch_id = sys.argv[1]
	for exp_id in sys.argv[2:]:
		exp_list.append("exp_{}.json".format(exp_id))
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
run_logs_path = os.path.join(batch_output_folder_path, "run_logs.csv")

# construct figure folder paths
figures_folder_path = plot_tools.construct_figure_folder_path(batch_output_folder_path, "figures", remove_existing=True)
nr_mae_figures_folder_path = plot_tools.construct_figure_folder_path(figures_folder_path, "nr_mae", remove_existing=True)
# estimate_figures_folder_path = plot_tools.construct_figure_folder_path(figures_folder_path, "estimates", remove_existing=True)
# absolute_error_figures_folder_path = plot_tools.construct_figure_folder_path(figures_folder_path, "absolute_errors", remove_existing=True)
# cumulative_error_figures_folder_path = plot_tools.construct_figure_folder_path(figures_folder_path, "cumulative_errors", remove_existing=True)
# change_score_figures_folder_path = plot_tools.construct_figure_folder_path(figures_folder_path, "change_scores", remove_existing=True)

# build propery dict for target documents
target_doc_props = {}
target_doc_props["paper"] = {}
target_doc_props["paper"]["textwidth"] = "default"
target_doc_props["paper"]["fr"] = 1
target_doc_props["paper"]["ar"] = 2.25
target_doc_props["paper"]["tex_config"] = "full"

raw_table =  pd.read_csv(run_logs_path, sep=",", index_col=False)
group_by_filter = [
	"drift_type",
	"drift_params",
	"alp_car",
	"noise_rate",
	"slwe_p_k",
	"slwe_p_L"
]
grouped_table = raw_table.groupby(by=group_by_filter).mean().reset_index()

slwe_p_k_list = raw_table["slwe_p_k"].unique()
slwe_p_L_list = raw_table["slwe_p_L"].unique()
# print(slwe_p_k_list)

for target_doc in target_doc_props.keys():

	# matplotlib config
	plt.rcParams.update(plot_tools.tex_config[target_doc_props[target_doc]["tex_config"]])

	# different line styles for each curve
	linestyle_cycler = cycler("linestyle", ["solid", "dashed", "dotted", "dashdot"])
	plt.rc("axes", prop_cycle=linestyle_cycler)

	for slwe_p_k in slwe_p_k_list:
		for slwe_p_L in slwe_p_L_list:

			filtered_table = grouped_table.query("slwe_p_k=={} and slwe_p_L=={}".format(slwe_p_k, slwe_p_L)).reset_index()
			# print(filtered_table)
			noise_rate_list = filtered_table["noise_rate"].unique()

			slwe_e_lambda = filtered_table["slwe_e_lambda"].unique()[0]
			slwe_e_mae_list = filtered_table["slwe_e_mae"].unique()

			slwe_t_lambda = filtered_table["slwe_t_lambda"].unique()[0]
			slwe_t_mae_list = filtered_table["slwe_t_mae"].unique()

			slwe_p_mae_list = filtered_table["slwe_p_mae"].unique()

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
			ax.set_ylim(0, 0.4)
			ax.set_ylabel("MAE")
			ax.grid(linestyle="dashed", alpha=0.5)
			ax.legend(loc="upper left")

			fig_name = "k{}_L{}".format(slwe_p_k, slwe_p_L)
			fig.savefig("{}/{}.{}".format(nr_mae_figures_folder_path, fig_name, "pdf"), bbox_inches="tight", pad_inches=0.05)
			plt.close()

			# plt.show()

exit()

total_number_of_figs = len(experiment_files) if exp_list == "all" else (len(exp_list)*len(target_doc_props.keys()))

with tqdm(
	total = total_number_of_figs,
	position = 0,
	leave = True,
	desc = "figures".ljust(10),
	bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}',
	ascii = True
	) as pbar:

	for target_doc in target_doc_props.keys():

		# matplotlib config
		plt.rcParams.update(plot_tools.tex_config[target_doc_props[target_doc]["tex_config"]])

		# iterate experiment file list
		for experiment_file in experiment_files:

			# check if experiment file is included in given list
			if exp_list != "all" and experiment_file not in exp_list:
				continue

			# read experiment file as JSON
			experiment_file_path = "{}/{}".format(experiment_files_folder_path, experiment_file)
			with open(experiment_file_path) as current_experiment_file:
				current_experiment = json.load(current_experiment_file)

			# read necessary info for the figure
			experiment_name = current_experiment["experiment_info"]["name"]
			drift_type = current_experiment["sequence_info"]["drift_type"]

			if drift_type == "sudden" or drift_type == "incremental_sin" or drift_type == "incremental_lin":
				n = current_experiment["sequence_info"]["drift_params"]["n"]
				d = current_experiment["sequence_info"]["drift_params"]["d"]
				sequence_length = n * d
				regime_parameters = current_experiment["sequence_info"]["regime_parameters"]
				parameter_sequence = np.repeat(np.array(regime_parameters), d)
			# elif drift_type == "mixed":
			# 	small_sudden = current_experiment["sequence_info"]["drift_params"]["small_sudden"]
			# 	incremental_sin = current_experiment["sequence_info"]["drift_params"]["incremental_sin"]
			# 	large_sudden = current_experiment["sequence_info"]["drift_params"]["large_sudden"]
			# 	incremental_lin = current_experiment["sequence_info"]["drift_params"]["incremental_lin"]
			# 	sequence_length = (
			# 		small_sudden["n"] * small_sudden["d"] + 
			# 		incremental_sin["n"] * incremental_sin["d"] + 
			# 		large_sudden["n"] * large_sudden["d"] + 
			# 		incremental_lin["n"] * incremental_lin["d"]
			# 	)
				# regime_parameters = current_experiment["sequence_info"]["regime_parameters"]

			parameter_sequence = current_experiment["sequence_info"]["p_sequence"]
			sequence_length = len(parameter_sequence)

			slwe_e_estimates = current_experiment["results"]["slwe_e"]["estimates"]
			slwe_t_estimates = current_experiment["results"]["slwe_t"]["estimates"]
			slwe_alm_estimates = current_experiment["results"]["slwe_alm"]["estimates"]
			slwe_alm_learning_rates = current_experiment["results"]["slwe_alm"]["learning_rates"]
			hy2_estimates = current_experiment["results"]["hy2"]["estimates"]
			# hddm_w_estimates = current_experiment["results"]["hddm_w"]["estimates"]

			# slwe_alm_change_scores = current_experiment["results"]["slwe_alm"]["change_scores"]
			# slwe_absolute_error = np.absolute(slwe_estimates - parameter_sequence)
			# slwe_alm_absolute_error = np.absolute(slwe_alm_estimates - parameter_sequence)
			# slwe_cumulative_absolute_error = np.cumsum(slwe_absolute_error, axis=0)
			# slwe_alm_cumulative_absolute_error = np.cumsum(slwe_alm_absolute_error, axis=0)

			# generate figure for current experiment (estimates)
			fig, ax = plt.subplots(
				nrows = 2,
				ncols = 2,
				figsize = plot_tools.set_size(
					width = target_doc_props[target_doc]["textwidth"],
					aspect_ratio = target_doc_props[target_doc]["ar"],
					fraction = target_doc_props[target_doc]["fr"]
				)
			)
			fig.set_tight_layout(True)

			for i in range(2):
				for j in range(2):
					k = 2*i+j
					ax[i][j].plot(
						[p[k] for p in parameter_sequence],
						label = "p {}".format(k),
						color = "black",
						alpha = 0.9,
						linewidth = 1,
						zorder = 10
					)
					# ax[i][j].plot(
					# 	[p[k] for p in slwe_e_estimates],
					# 	label = "SLWE (E)",
					# 	alpha = 0.9,
					# 	linewidth = 1,
					# 	zorder = 2
					# )
					ax[i][j].plot(
						[p[k] for p in slwe_alm_estimates],
						label = "SLWE-ALM",
						alpha = 0.5,
						linewidth = 1,
						zorder = 4
					)
					ax[i][j].plot(
						[p[k] for p in hy2_estimates],
						label = "HY2",
						alpha = 0.5,
						linewidth = 1,
						zorder = 4
					)
					# ax[i][j].plot(
					# 	[p[k] for p in slwe_alm_ins.reg_mean],
					# 	label = "SLWE-ALM reg_mean",
					# 	alpha = 0.9,
					# 	linewidth = 1,
					# 	zorder = 4
					# )
					# ax[i][j].plot(
					# 	[p[k] for p in slwe_alm_ins.reg_mean_mstd],
					# 	label = "SLWE-ALM mstd",
					# 	alpha = 0.9,
					# 	linewidth = 1,
					# 	zorder = 4
					# )
					ax[i][j].plot(
						slwe_alm_learning_rates,
						label = "SLWE-ALM eta",
						alpha = 0.9,
						linewidth = 1,
						zorder = 4
					)
					# ax[i][j].set_ylim([0,1])
					# ax[i][j].legend()
					# ax[i][j].grid(
					# 	linestyle = "dashed",
					# 	color = "lightgray",
					# 	alpha = 1.00,
					# 	zorder = 0
					# )

					# # ax.set_xlabel("Observation Count")
					ax[i][j].set_xlim([0, sequence_length])
					if drift_type == "sudden":
						ax[i][j].set_xticks(np.linspace(0, sequence_length, n + 1))
					# 	# plt.xticks(rotation=90)
					# elif drift_type == "incremental_sin":
					# 	ax.set_xticks(np.linspace(0, sequence_length, 5))
					# elif drift_type == "incremental_lin":
					# 	ax.set_xticks(np.linspace(0, sequence_length, 5))

					# # ax.set_ylabel("Probability")
					ax[i][j].set_ylim([-0.05,1.25])
					ax[i][j].set_yticks(np.linspace(0, 1, 6))
					ax[i][j].grid(linestyle="dashed", alpha=0.5)
					ax[i][j].legend(
						loc="upper right", 
						ncol=2,
						handlelength = 1,
						mode = "expand",
						framealpha = 1
					)

			fig.savefig("{}/{}_{}.{}".format(estimate_figures_folder_path, experiment_name, "est", "pdf"), bbox_inches="tight", pad_inches=0.05)
			plt.close()

			# # generate figure for current experiment (absolute error)
			# fig, ax = plt.subplots(1, 1, figsize=plot_tools.set_size("thesis", aspect_ratio=2.25, fraction=1))
			# fig.set_tight_layout(True)
			# ax.plot(slwe_absolute_error, label="SLWE", color="silver", linewidth=1, zorder=1)
			# ax.plot(slwe_alm_absolute_error, label="SLWE-ALM", color="navy", linewidth=1, zorder=2)
			# ax.set_xlabel("Observation Count")
			# ax.set_xlim([0,np.sum(regime_lengths)])
			# ax.set_xticks(np.cumsum(regime_lengths))
			# ax.set_ylabel("Absolute Error")
			# # ax.set_ylim([0,1])
			# ax.grid(linestyle="dashed", alpha=0.5)
			# ax.legend(loc="upper left", ncol=2)
			# fig.savefig("{}/{}_{}.{}".format(absolute_error_figures_folder_path, experiment_name, "abs_err", "pdf"), bbox_inches="tight", pad_inches=0.05)
			# plt.close()

			# # generate figure for current experiment (cumulative absolute error)
			# fig, ax = plt.subplots(1, 1, figsize=plot_tools.set_size("thesis", aspect_ratio=2.25, fraction=1))
			# fig.set_tight_layout(True)
			# ax.plot(slwe_cumulative_absolute_error, label="SLWE", color="black", linestyle="solid", linewidth=1, zorder=1)
			# ax.plot(slwe_alm_cumulative_absolute_error, label="SLWE-ALM", color="black", linestyle="dashed", linewidth=1, zorder=2)
			# ax.set_xlabel("Observation Count")
			# ax.set_xlim([0,np.sum(regime_lengths)])
			# ax.set_xticks(np.cumsum(regime_lengths))
			# ax.set_ylabel("Cumulative Absolute Error")
			# # ax.set_ylim([0,1])
			# ax.grid(linestyle="dashed", alpha=0.5)
			# ax.legend(loc="upper left", ncol=2)
			# fig.savefig("{}/{}_{}.{}".format(cumulative_error_figures_folder_path, experiment_name, "cum_abs_err", "pdf"), bbox_inches="tight", pad_inches=0.05)
			# plt.close()

			# # generate figure for current experiment (change_scores)
			# fig, ax = plt.subplots(1, 1, figsize=plot_tools.set_size("thesis", aspect_ratio=2.25, fraction=1))
			# fig.set_tight_layout(True)
			# ax.plot(slwe_alm_change_scores, label="SLWE-ALM", color="black", linestyle="solid", linewidth=1, zorder=1)
			# ax.set_xlabel("Observation Count")
			# ax.set_xlim([0,np.sum(regime_lengths)])
			# ax.set_xticks(np.cumsum(regime_lengths))
			# ax.set_ylabel("Change Score")
			# # ax.set_ylim([0,1])
			# ax.grid(linestyle="dashed", alpha=0.5)
			# # ax.legend(loc="upper left", ncol=2)
			# fig.savefig("{}/{}_{}.{}".format(change_score_figures_folder_path, experiment_name, "ch_scr", "pdf"), bbox_inches="tight", pad_inches=0.05)
			# plt.close()

			pbar.update()
