import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler

from pearl.tools.visualizer import get_matplotlib_config, set_size

# build paths
root_folder = os.path.abspath(os.curdir)
figure_folder = os.path.join(root_folder, "test", "figure_generator")
Path(figure_folder).mkdir(parents=True, exist_ok=True)

def persistence(l, L, k):
	return (np.power(k, np.divide(l, L)) - 1) / (k - 1)

k_list = [5, 15, 50, 100]
L = 10
kL1 = zip(k_list, np.repeat(L, len(k_list)))
k = 15
L_list = [15, 50, 100]
kL2 = zip(np.repeat(k, len(L_list)), L_list)
kL = [list(kL1), list(kL2)]
for index, sublist in enumerate(kL):

	# matplotlib config
	plt.rcParams.update(get_matplotlib_config("generic_latex"))

	# different line styles for each curve
	linestyle_cycler = cycler("linestyle", ["solid", "dashed", "dotted", "dashdot"])
	plt.rc("axes", prop_cycle=linestyle_cycler)

	# visualization
	fig, ax = plt.subplots(
		nrows = 1,
		ncols = 1,
		figsize = set_size(
			width = "pat_reg_dc",
			aspect_ratio = 1.5,
			fraction = 1
		)
	)
	fig.set_tight_layout(True)

	for k, L in sublist:
		# print("k:{}, L:{}".format(k, L))
		l = np.arange(0, L+1)
		ax.plot(
			l/L,
			persistence(l, L, k),
			label = "${0[0]}={0[1]}$".format(["k", k] if index == 0 else ["L", L]),
			color = "black",
			linewidth = 0.75,
			zorder = 5
		)

	ax.set_xlabel("Normalized Occurrence Frequency")
	ax.set_xlim([0,1])
	ax.set_ylabel("Persistence")
	ax.set_ylim([0,1])
	ax.grid(linestyle="dashed", color="lightgray", alpha=1.00, zorder=0)
	ax.legend(loc="upper left")

	figure_address = "{0}/rho_{1}.{2}".format(figure_folder, ["k", "L"][index], "pdf")
	print(figure_address)
	fig.savefig(figure_address, bbox_inches="tight", pad_inches=0.05)
	# plt.show()
