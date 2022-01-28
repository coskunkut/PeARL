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
		slwe_p_a_mae_list,
		slwe_p_e_mae_list,
		target_doc_props,
		fig_path
	):

	# generate figure for current experiment (estimates)
	fig, ax = plt.subplots(
		nrows = 1,
		ncols = 1,
		figsize = plot_tools.set_size(
			width = target_doc_props["textwidth"],
			aspect_ratio = target_doc_props["aspect_ratio"],
			fraction = target_doc_props["fraction"]
		)
	)
	fig.set_tight_layout(True)

	if slwe_p_a_mae_list is not None:
		ax.plot(
			noise_rate_list,
			slwe_p_a_mae_list,
			label = "SLWE-P (A)",
			color = "maroon",
			alpha = 0.9,
			linewidth = 1,
			linestyle = "solid",
			zorder = 10
		)
	ax.plot(
		noise_rate_list,
		slwe_p_e_mae_list,
		label = "SLWE-P (E)",
		color = "black",
		alpha = 0.9,
		linewidth = 1,
		linestyle = "solid",
		zorder = 10
	)
	ax.plot(
		noise_rate_list,
		slwe_e_mae_list,
		label = "SLWE ($\eta={:.2f}$)".format(1-slwe_e_lambda),
		color = "black",
		alpha = 0.9,
		linewidth = 1,
		linestyle = "dashed",
		zorder = 10
	)
	ax.plot(
		noise_rate_list,
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
	ax.set_ylabel("Mean Absolute Error (MAE)")
	ax.grid(linestyle="dashed", alpha=0.5)
	ax.legend(loc="upper left")

	fig.savefig(fig_path, bbox_inches="tight", pad_inches=0.05)
	plt.close()

# get batch id
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
run_logs_path = os.path.join(batch_output_folder_path, "run_logs.csv")

# build propery dict for target documents
target_doc_props = {
	"vanilla_paper_1c": {
		"textwidth": "vanilla_paper",
		"config": "latex",
		"fraction": 1,
		"aspect_ratio": 2.5
	}
}

# construct figure folder paths
figures_folder_path = plot_tools.construct_figure_folder_path(batch_output_folder_path, "figures", remove_existing=True)
nr_mae_figures_folder_path = plot_tools.construct_figure_folder_path(figures_folder_path, "nr_mae", remove_existing=True)
for target_doc in target_doc_props.keys():
	plot_tools.construct_figure_folder_path(nr_mae_figures_folder_path, target_doc, remove_existing=True)

raw_table =  pd.read_csv(run_logs_path, sep=",", index_col=False)
sequence_ids = raw_table["sequence_id"].unique()
group_by_filter = [
	"sequence_id",
	"noise_rate"
]
grouped_table = raw_table.groupby(by=group_by_filter).mean().reset_index()

alp_car = raw_table["alp_car"].unique()[0]

# these matrices are for overall nr_mae fig
slwe_e_mae_lists = np.zeros((18, 21))
slwe_t_mae_lists = np.zeros((18, 21))
slwe_p_a_mae_lists = np.zeros((18, 21))
slwe_p_e_mae_lists = np.zeros((18, 21))

total_number_of_figs = len(target_doc_props.keys()) * len(sequence_ids)

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
		plt.rcParams.update(plot_tools.config[target_doc_props[target_doc]["config"]])

		# different line styles for each curve
		# linestyle_cycler = cycler("linestyle", ["solid", "dashed", "dotted", "dashdot"])
		# plt.rc("axes", prop_cycle=linestyle_cycler)

		for index, sequence_id in enumerate(sequence_ids):

			filtered_table = grouped_table.loc[(grouped_table['sequence_id'] == sequence_id)]

			noise_rate_list = filtered_table["noise_rate"].unique()

			slwe_e_lambda = filtered_table["slwe_e_lambda"].unique()[0]
			slwe_e_mae_list = filtered_table["slwe_e_mae"].unique()

			slwe_t_lambda = filtered_table["slwe_t_lambda"].unique()[0]
			slwe_t_mae_list = filtered_table["slwe_t_mae"].unique()

			slwe_p_a_mae_list = filtered_table["slwe_p_a_mae"].unique()
			slwe_p_e_mae_list = filtered_table["slwe_p_e_mae"].unique()

			slwe_e_mae_lists[index] = slwe_e_mae_list
			slwe_t_mae_lists[index] = slwe_t_mae_list
			slwe_p_a_mae_lists[index] = slwe_p_a_mae_list
			slwe_p_e_mae_lists[index] = slwe_p_e_mae_list

			fig_name = "{}".format(sequence_id)
			fig_path = "{}/{}/{}.{}".format(nr_mae_figures_folder_path, target_doc, fig_name, "pdf")

			nr_mae_fig(
				noise_rate_list,
				slwe_e_mae_list,
				slwe_e_lambda,
				slwe_t_mae_list,
				slwe_t_lambda,
				slwe_p_a_mae_list,
				slwe_p_e_mae_list,
				target_doc_props[target_doc],
				fig_path
			)

			pbar.update()

# overall nr_mae
fig_path = "{}/{}/{}.{}".format(nr_mae_figures_folder_path, "vanilla_paper_1c", "arcelik_all", "pdf")
nr_mae_fig(
	np.linspace(0,1,21),
	slwe_e_mae_lists.mean(0),
	0.9,
	slwe_t_mae_lists.mean(0),
	0.99,
	None,
	# slwe_p_a_mae_lists.mean(0),
	slwe_p_e_mae_lists.mean(0),
	target_doc_props["vanilla_paper_1c"],
	fig_path
)
