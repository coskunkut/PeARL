import os
import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler

from pbnm.tools import plot_tools

# build paths
root_folder = os.path.abspath(os.curdir)
figure_folder = os.path.join(root_folder, "test", "figures")

# build propery dict for target documents
target_doc_props = {}
target_doc_props["pat_reg"] = {}
target_doc_props["pat_reg"]["textwidth"] = "pat_reg_dc"
target_doc_props["pat_reg"]["fr"] = 1
target_doc_props["pat_reg"]["ar"] = 1.5
target_doc_props["pat_reg"]["config"] = "latex"

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

	for target_doc in target_doc_props.keys():

		# matplotlib config
		plt.rcParams.update(plot_tools.config[target_doc_props[target_doc]["config"]])

		# different line styles for each curve
		linestyle_cycler = cycler("linestyle", ["solid", "dashed", "dotted", "dashdot"])
		plt.rc("axes", prop_cycle=linestyle_cycler)

		# visualization
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
