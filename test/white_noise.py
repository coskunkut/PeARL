# persistence of white noise

import os
import sys
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from pbnm.tools import MovingStatistics, plot_tools
from tqdm import tqdm

# persistence calculation
def calculate_rho (k, l, L):
	return (np.power(k, np.divide(l, L)) - 1) / (k - 1)

# build paths
root_folder = os.path.abspath(os.curdir)
figure_folder = os.path.join(root_folder, "test", "figures")

# generate random sequence
rng_seed = 42
rng = np.random.default_rng(rng_seed)
alp_car = 4
n = 5000
sequence = rng.choice(
	a = alp_car,
	size = n
)
print("sequence stats: {}".format(Counter(sequence)))

# persistence parameters
k = 5
L = 50

# calculate expected rho (avg)
avg_expected_rho = (np.power(k, 1/alp_car) - 1) / (k - 1)
print("expected avg. rho: {}".format(avg_expected_rho))

# calculate the following
# - raw rho history
# - rho sample mean history
# - final rho vector
rho_history = []
rho_sample_mean = []
rho_ms = MovingStatistics()
memory = []
l = np.zeros(alp_car, dtype=int)
for token in sequence:
	memory.append(token)
	l[token] += 1
	if len(memory) > L:
		oldest_token = memory.pop(0)
		l[oldest_token] -= 1
	rho = calculate_rho(k, l, L)
	rho_history.append(rho.tolist())
	# rho_ms.add(rho)
	rho_ms.add(np.mean(rho))
	rho_sample_mean.append(rho_ms.mean)
print("final rho vec: {}".format(rho_ms.mean))
print("final avg. rho: {}".format(np.mean(rho_ms.mean)))
print("theoretical - empirical: {}".format(avg_expected_rho - np.mean(rho_ms.mean)))


# build propery dict for target documents
target_doc_props = {
	"paper": {
		"textwidth": "default",
		"fr": 1,
		"ar": 2.25,
		"tex_config": "full"
	}
}

# visualization
target_doc = "paper"
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
	[np.mean(rho) for rho in rho_history],
	label = "average persistence",
	color = "black",
	alpha = 0.9,
	linewidth = 0.5,
	linestyle = "solid",
	zorder = 3
)
ax.plot(
	rho_sample_mean,
	label = "average persistence (sample mean)",
	color = "black",
	alpha = 0.9,
	linewidth = 1,
	linestyle = "dashed",
	zorder = 4
)
ax.set_xlabel("Observation Count")
ax.set_xlim([0, n])
ax.set_ylabel("Persistence")
# ax.set_ylim([0, 0.08])
ax.grid(linestyle="dashed", alpha=0.5)
ax.legend(loc="lower right")
figure_address = "{0}/white_noise_rho.{1}".format(figure_folder, "pdf")
fig.savefig(figure_address, bbox_inches="tight", pad_inches=0.05)
# plt.show()
