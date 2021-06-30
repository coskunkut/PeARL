import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from sklearn import metrics

from persistence.generator import Multinomial
from persistence.estimator import SLWE, SLWE_P
from persistence.tools import plot_tools


# build paths
root_folder = os.path.abspath(os.curdir)
figure_folder = os.path.join(root_folder, "test", "figures")

# build propery dict for target documents
target_doc_props = {}
target_doc_props["default"] = {}
target_doc_props["default"]["textwidth"] = "default"
target_doc_props["default"]["fr"] = 1
target_doc_props["default"]["ar"] = 2.25
target_doc_props["default"]["tex_config"] = "full"

slwe_explore_mae_list = []
slwe_exploit_mae_list = []
slwe_alr_mae_list = []
slwe_p_mae_list = []
slwe_explore_eta = 0.1
slwe_exploit_eta = 0.01

noise_bounds = [0, 1.0]
noise_rate_list = np.linspace(noise_bounds[0], noise_bounds[1], num=15)
for noise_rate in tqdm(noise_rate_list):
	multinomial_ins = Multinomial(25)
	multinomial_ins.sudden_drift_diff(4, 10, 500, 0.3, 0.6)
	multinomial_ins.inject_white_noise(4, noise_rate)

	slwe_explore = SLWE(4, 1-slwe_explore_eta)
	slwe_explore.process_sequence(multinomial_ins.sequence, progress=False, progress_desc="SLWE (E)")
	slwe_explore_mae = metrics.mean_absolute_error(multinomial_ins.p_sequence, slwe_explore.estimates)
	slwe_explore_mae_list.append(slwe_explore_mae)

	slwe_exploit = SLWE(4, 1-slwe_exploit_eta)
	slwe_exploit.process_sequence(multinomial_ins.sequence, progress=False, progress_desc="SLWE (T)")
	slwe_exploit_mae = metrics.mean_absolute_error(multinomial_ins.p_sequence, slwe_exploit.estimates)
	slwe_exploit_mae_list.append(slwe_exploit_mae)

	slwe_p = SLWE_P(4)
	slwe_p.process_sequence(multinomial_ins.sequence, progress=False, progress_desc="SLWE_P")
	slwe_p_mae = metrics.mean_absolute_error(multinomial_ins.p_sequence, slwe_p.estimates)
	slwe_p_mae_list.append(slwe_p_mae)

# get turning points
slwe_mae_best = np.minimum(slwe_explore_mae_list, slwe_exploit_mae_list)
nr, mae_1 = plot_tools.upsample_coords([noise_rate_list, slwe_mae_best])
nr, mae_2 = plot_tools.upsample_coords([noise_rate_list, slwe_p_mae_list])
diff = np.abs(np.subtract(mae_1, mae_2))
turning_points = nr[np.argwhere(diff < 0.0001)].flatten().tolist()
print("\nturning points:\n{}".format(turning_points))

# visualization
target_doc = "default"
plt.rcParams.update(plot_tools.tex_config[target_doc_props[target_doc]["tex_config"]])
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
	slwe_explore_mae_list, 
	label="SLWE ($\eta={:.2f}$)".format(slwe_explore_eta), 
	linestyle="solid", 
	linewidth=0.75, 
	zorder=5
)
ax.plot(
	noise_rate_list, 
	slwe_exploit_mae_list, 
	label="SLWE ($\eta={:.2f}$)".format(slwe_exploit_eta), 
	linestyle="solid", 
	linewidth=0.75, 
	zorder=5
)
ax.plot(
	noise_rate_list,
	slwe_p_mae_list,
	label = "SLWE-P",
	linestyle = "solid",
	linewidth = 0.75,
	zorder = 5
)
for turning_point in turning_points:
	ax.axvline(
		x = turning_point,
		linestyle = "dashed",
		linewidth = 0.75,
		color = "black",
		zorder = 5
	)
	# ax.text(
	# 	turning_point, 0.08,
	# 	"{:.3f}".format(turning_point),
	# 	va="center", ha="center",
	# 	zorder=6,
	#     bbox={'facecolor': 'lightgray', 'alpha': 1, 'pad': 5}
	# )
# ax.fill_between(
# 	nr,
# 	mae_1,
# 	mae_2,
# 	where = mae_2 < mae_1,
# 	alpha = 0.2,
# 	linewidth = 0
# )

ax.set_xlabel("Noise Rate")
ax.set_xlim(noise_bounds)
# ax.set_xticks(x_ticks)
plt.xticks(list(plt.xticks()[0]) + turning_points)
# plt.xticks(rotation=90)
ax.set_ylabel("MAE")
# ax.set_ylim([0,1])
ax.grid(linestyle="dashed", color="lightgray", alpha=1.00, zorder=0)
ax.legend(loc="upper left")

figure_address = "{0}/turning_points.{1}".format(figure_folder, "pdf")
fig.savefig(figure_address, bbox_inches="tight", pad_inches=0.05)
# plt.show()
