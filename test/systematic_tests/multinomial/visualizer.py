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
target_doc_props["paper"]["fr"] = 0.5
target_doc_props["paper"]["ar"] = 1.25
target_doc_props["paper"]["tex_config"] = "half"

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

drift_type = raw_table["drift_type"].unique()[0]
drift_params = raw_table["drift_params"].unique()[0].replace("/", "_")
alp_car = raw_table["alp_car"].unique()[0]

total_number_of_figs = len(target_doc_props.keys()) * len(slwe_p_k_list) * len(slwe_p_L_list)

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
				ax.set_xticks(np.linspace(0, 1, num=11))
				# ax.set_ylim(0, 0.4)
				# ax.set_ylim(0)
				ax.set_ylabel("MAE")
				ax.grid(linestyle="dashed", alpha=0.5)
				ax.legend(loc="upper left")

				# fig_name = "k{}_L{}".format(slwe_p_k, slwe_p_L)
				fig_name = "{}_{}_c{}".format(drift_type, drift_params, alp_car)
				fig.savefig("{}/{}.{}".format(nr_mae_figures_folder_path, fig_name, "pdf"), bbox_inches="tight", pad_inches=0.05)
				plt.close()

				# plt.show()

				pbar.update()
