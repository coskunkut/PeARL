import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pbnm.estimator import SLWE, SLWE_P
from pbnm.generator import Multinomial
from pbnm.tools import align_matrix, plot_tools
from sklearn import metrics
from tabulate import tabulate

# build paths
root_folder = os.path.abspath(os.curdir)
figure_folder = os.path.join(root_folder, "test", "figures")

# generate non-stationary sequence
multinomial_ins = Multinomial(25)
multinomial_ins.sudden_drift_diff(
	alp_car = 4,
	n = 4,
	d = 500,
	min_diff = 0.3,
	max_diff = 0.6
)

# inject white noise
# multinomial_ins.inject_white_noise(4, 0.1, start_index=500, stop_index=1000)
multinomial_ins.inject_white_noise(4, 0.3, start_index=None, stop_index=None)

sequence = [token for token in multinomial_ins.o_sequence]
# sequence = [int(token) for token in multinomial_ins.o_sequence]

# run estimators
print("running estimators...")

# exploratory slwe
slwe_explore_lambda = 1 - (0.05)
slwe_explore = SLWE(4, slwe_explore_lambda)
slwe_explore.process_sequence(sequence, progress=True, progress_desc="SLWE (E)")
aligned_p_sequence = align_matrix(multinomial_ins.p_sequence, slwe_explore.token_index_map)
slwe_explore_mae = metrics.mean_absolute_error(aligned_p_sequence, slwe_explore.estimates)

# exploitative slwe
slwe_exploit_lambda = 1 - (0.01)
slwe_exploit = SLWE(4, slwe_exploit_lambda)
slwe_exploit.process_sequence(sequence, progress=True, progress_desc="SLWE (T)")
aligned_p_sequence = align_matrix(multinomial_ins.p_sequence, slwe_exploit.token_index_map)
slwe_exploit_mae = metrics.mean_absolute_error(aligned_p_sequence, slwe_exploit.estimates)

slwe_p = SLWE_P(4)
slwe_p.process_sequence(sequence, progress=True, progress_desc="SLWE-P")
aligned_p_sequence = align_matrix(multinomial_ins.p_sequence, slwe_p.token_index_map)
slwe_p_mae = metrics.mean_absolute_error(aligned_p_sequence, slwe_p.estimates)

slwe_comp_lambda = 1 - (np.mean(slwe_p.eta_history))
slwe_comp = SLWE(4, slwe_comp_lambda)
slwe_comp.process_sequence(sequence, progress=True, progress_desc="SLWE (C)")
aligned_p_sequence = align_matrix(multinomial_ins.p_sequence, slwe_comp.token_index_map)
slwe_comp_mae = metrics.mean_absolute_error(aligned_p_sequence, slwe_comp.estimates)

print("\nSLWE-P Stats:")
print(tabulate([
		["Avg. Eta", "{:.4f}".format(np.mean(slwe_p.eta_history))],
		["Avg. Rho", "{:.4f}".format(np.mean(slwe_p.rho))],
	],
	headers=["Key", "Value"])
)

print("")
print(tabulate([
		["SLWE (E)", "{:.4f}".format(1-slwe_explore_lambda), "{:.4f}".format(slwe_explore_mae)],
		["SLWE (T)", "{:.4f}".format(1-slwe_exploit_lambda), "{:.4f}".format(slwe_exploit_mae)],
		["SLWE (C)", "{:.4f}".format(1-slwe_comp_lambda), "{:.4f}".format(slwe_comp_mae)],
		["SLWE-P", "{:.4f}".format(np.mean(slwe_p.eta_history)), "{:.4f}".format(slwe_p_mae)],
	],
	headers=["Algorithm", "Eta", "MAE"])
)

# visualization
# plt.rcParams.update(plot_tools.tex_config_full)

# separate figs
for i in range(4):
	fig, ax = plt.subplots(1, 1)
	fig.set_tight_layout(True)
	ax.plot(
		[p[i] for p in multinomial_ins.p_sequence],
		label = "p {}".format(i),
		color = "black",
		alpha = 0.9,
		linewidth = 0.5,
		zorder = 10
	)
	ax.plot(
		[p[slwe_p.token_index_map[str(i)]] for p in slwe_p.estimates],
		label = "SLWE-P",
		alpha = 0.9,
		linewidth = 0.5,
		zorder = 4
	)
	ax.plot(
		slwe_p.eta_history,
		label = "SLWE-P eta",
		alpha = 0.9,
		linewidth = 0.5,
		zorder = 4
	)
	ax.plot(
		[p[slwe_explore.token_index_map[str(i)]] for p in slwe_explore.estimates],
		label = "SLWE ($\eta={:.2f}$)".format(1-slwe_explore_lambda),
		alpha = 0.9,
		linewidth = 0.5,
		zorder = 4
	)
	ax.plot(
		[p[slwe_exploit.token_index_map[str(i)]] for p in slwe_exploit.estimates],
		label = "SLWE ($\eta={:.2f}$)".format(1-slwe_exploit_lambda),
		alpha = 0.9,
		linewidth = 0.5,
		zorder = 4
	)
	ax.plot(
		[p[slwe_comp.token_index_map[str(i)]] for p in slwe_comp.estimates],
		label = "SLWE ($\eta={:.2f}$)".format(1-slwe_comp_lambda),
		alpha = 0.9,
		linewidth = 0.5,
		zorder = 4
	)
	# ax.fill_between(
	# 	x = np.arange(0, len(sequence)),
	# 	y1 = 0,
	# 	y2 = 1,
	# 	where = np.array(multinomial_ins.noise_flags) > 0,
    #     color = "lightcoral",
	# 	alpha = 1,
	# 	# interpolate = True,
	# 	# step = "pre",
	# 	# linewidth = 0
	# )
	ax.set_xlim([0, len(sequence)])
	ax.set_ylim([0, 1])
	ax.set_yticks(np.linspace(0, 1, 6))
	ax.grid(linestyle="dashed", alpha=0.5)
	ax.legend(loc="best")
	figure_address = "{0}/lr_control_noise_{1}.{2}".format(figure_folder, i, "pdf")
	fig.savefig(figure_address, bbox_inches="tight", pad_inches=0.05)
	plt.cla()

# grid
fig, ax = plt.subplots(2, 2)
fig.set_tight_layout(True)
for i in range(2):
	for j in range(2):
		k = 2*i+j
		ax[i][j].plot(
			[p[k] for p in multinomial_ins.p_sequence],
			label = "p {}".format(k),
			color = "black",
			alpha = 0.9,
			linewidth = 0.5,
			zorder = 10
		)
		ax[i][j].plot(
			[p[slwe_explore.token_index_map[str(k)]] for p in slwe_explore.estimates],
			label = "SLWE (E)",
			alpha = 0.9,
			linewidth = 1,
			zorder = 2
		)
		ax[i][j].plot(
			[p[slwe_p.token_index_map[str(k)]] for p in slwe_p.estimates],
			label = "SLWE-P",
			alpha = 0.5,
			linewidth = 0.5,
			zorder = 4
		)
		ax[i][j].plot(
			slwe_p.eta_history,
			label = "SLWE-P eta",
			alpha = 0.9,
			linewidth = 0.5,
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
		# ax[i][j].set_xlim([0, sequence_length])

		# # ax.set_ylabel("Probability")
		ax[i][j].set_ylim([-0.05,1])
		ax[i][j].set_yticks(np.linspace(0, 1, 6))
		ax[i][j].grid(linestyle="dashed", alpha=0.5)
		ax[i][j].legend(
			loc="best"
		)

figure_address = "{0}/fvsp_grid.{1}".format(figure_folder, "pdf")
fig.savefig(figure_address, bbox_inches="tight", pad_inches=0.05)
plt.close()
