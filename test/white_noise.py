# persistence of white noise

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm

from pbnm.tools import MovingStatistics

# build paths
root_folder = os.path.abspath(os.curdir)
figure_folder = os.path.join(root_folder, "test", "figures")

rng_seed = 42
rng = np.random.default_rng(rng_seed)

alp_car = 4
n = 10000

sequence = rng.choice(
	a = alp_car,
	size = n
)

# print(Counter(sequence))

# persistence parameters
k = 15
L = 100

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
	rho = (np.power(k, np.divide(l, L)) - 1) / (k - 1)
	rho_history.append(rho.tolist())
	# rho_ms.add(np.mean(rho))
	rho_ms.add(rho)
	rho_sample_mean.append(rho_ms.mean)
print("final rho vec: {}".format(rho_ms.mean))
# print("freqs: {}".format(l))
rho_wn = (np.power(k, 1/alp_car) - 1) / (k - 1)
print("expected: {}".format(rho_wn))

# visualization
fig, ax = plt.subplots(1, 1)
fig.set_tight_layout(True)
ax.plot(
	# [np.mean(rho) for rho in rho_history],
	[np.mean(rho) for rho in rho_history],
	label = "rho {}".format(0),
	# color = "black",
	alpha = 0.9,
	linewidth = 0.5,
	zorder = 3
)
ax.plot(
	rho_sample_mean,
	label = "rho sm",
	# color = "black",
	alpha = 0.9,
	linewidth = 0.5,
	zorder = 4
)
ax.set_xlim([0, n])
# ax.set_ylim([0, 1])
ax.grid(linestyle="dashed", alpha=0.5)
ax.legend(loc="best")
figure_address = "{0}/white_noise_rho.{1}".format(figure_folder, "pdf")
fig.savefig(figure_address, bbox_inches="tight", pad_inches=0.05)
# plt.show()
