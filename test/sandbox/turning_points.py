import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from tqdm import tqdm

from pearl.estimator import SLWE, PeARL
from pearl.generator import Multinomial
from pearl.tools.signal import align_matrix
from pearl.tools.visualizer import get_matplotlib_config, set_size, upsample_coords

# build paths
root_folder = os.path.abspath(os.curdir)
figure_folder = os.path.join(root_folder, "test", "sandbox")

slwe_explore_mae_list = []
slwe_exploit_mae_list = []
slwe_alr_mae_list = []
pearl_mae_list = []
slwe_explore_eta = 0.1
slwe_exploit_eta = 0.01

noise_bounds = [0, 1.0]
noise_rate_list = np.linspace(noise_bounds[0], noise_bounds[1], num=15)
for noise_rate in tqdm(noise_rate_list):

	# generate sequence
	multinomial_ins = Multinomial(4, 25)
	multinomial_ins.sudden_drift_diff(
		n = 10,
		d = 500,
		min_diff = 0.3,
		max_diff = 0.6
	)
	multinomial_ins.inject_white_noise(
		noise_rate = noise_rate
	)
	sequence = [symbol for symbol in multinomial_ins.o_sequence]

	slwe_explore = SLWE(4, 1-slwe_explore_eta)
	slwe_explore.process_sequence(sequence, progress=False, progress_desc="SLWE (E)")
	aligned_p_sequence = align_matrix(multinomial_ins.p_sequence, slwe_explore.symbol_index_map)
	slwe_explore_mae = metrics.mean_absolute_error(aligned_p_sequence, slwe_explore.estimates[1:])
	slwe_explore_mae_list.append(slwe_explore_mae)

	slwe_exploit = SLWE(4, 1-slwe_exploit_eta)
	slwe_exploit.process_sequence(sequence, progress=False, progress_desc="SLWE (T)")
	aligned_p_sequence = align_matrix(multinomial_ins.p_sequence, slwe_exploit.symbol_index_map)
	slwe_exploit_mae = metrics.mean_absolute_error(aligned_p_sequence, slwe_exploit.estimates[1:])
	slwe_exploit_mae_list.append(slwe_exploit_mae)

	pearl = PeARL(4)
	pearl.process_sequence(sequence, progress=False, progress_desc="pearl")
	aligned_p_sequence = align_matrix(multinomial_ins.p_sequence, pearl.symbol_index_map)
	pearl_mae = metrics.mean_absolute_error(aligned_p_sequence, pearl.estimates[1:])
	pearl_mae_list.append(pearl_mae)

# get turning points
slwe_mae_best = np.minimum(slwe_explore_mae_list, slwe_exploit_mae_list)
nr, mae_1 = upsample_coords([noise_rate_list, slwe_mae_best])
nr, mae_2 = upsample_coords([noise_rate_list, pearl_mae_list])
diff = np.abs(np.subtract(mae_1, mae_2))
turning_points = nr[np.argwhere(diff < 0.0001)].flatten().tolist()
print("\nturning points:\n{}".format(turning_points))

# visualization
plt.rcParams.update(get_matplotlib_config("generic_latex"))
fig, ax = plt.subplots(
	nrows = 1,
	ncols = 1,
	figsize = set_size(
		width = "generic_paper",
		aspect_ratio = 2.25,
		fraction = 1
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
	pearl_mae_list,
	label = "PeARL",
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
