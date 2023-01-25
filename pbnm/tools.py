import os
import glob
import numpy as np
from scipy import interpolate
from more_itertools import windowed, grouper

# orders columns of a matrix based on an index mapping
def align_matrix(matrix, index_mapping):
	matrix = np.array(matrix)
	aligned_matrix = np.empty(matrix.shape)
	for token, index in index_mapping.items():
		aligned_matrix[:, index] = matrix[:, int(token)]
	return aligned_matrix

# decomposes given 1d signal into overlapping parts
def decompose_signal(data, piece_length, overlap):
	if overlap:
		time_series_pieces = list(windowed(data, piece_length))
	else:
		time_series_pieces = list(grouper(data, piece_length, 0))
	return time_series_pieces

# calculates the difference between clean and noisy sequence, returns estimated noise rate
def get_effective_noise_rate(clean_sequence, noisy_sequence):
	sequence_diff = np.subtract(clean_sequence, noisy_sequence)
	diff_count = np.count_nonzero(sequence_diff == 0)
	return 1 - (diff_count / len(clean_sequence))

# efficient calculation of moving statistics
class MovingStatistics(object):

	def __init__(self):
		self.reset()

	# Welford’s method
	# for unbiased var use:
	# self.var = ((self.var * (self.N - 2)) + (new - new_mean) * (new - old_mean)) / (self.N - 1)
	def add(self, new):
		self.N += 1
		if self.N == 1:
			self.mean = new
			self.var = 0
			self.std = 0
		else:
			old_mean = self.mean
			new_mean = old_mean + (new - old_mean) / self.N
			self.mean = new_mean
			self.var = ((self.var * (self.N - 1)) + (new - new_mean) * (new - old_mean)) / (self.N)
			self.std = np.sqrt(self.var)

	def replace(self, old, new):
		old_mean = self.mean
		new_mean = old_mean + (new - old) / self.N
		self.mean = new_mean
		self.var += (new - old) * (new - new_mean + old - old_mean) / (self.N)
		self.std = np.sqrt(self.var)

	def reset(self):
		self.N = 0
		self.mean = 0
		self.var = 0
		self.std = 0

	def show(self):
		# print("N:{}, mean:{:.4f}, var:{:.4f}, std:{:.4f}".format(self.N, self.mean, self.var, self.std))
		print("ms: mean:{:.4f}, var:{:.4f}, std:{:.4f}".format(self.mean, self.var, self.std))


class plot_tools:

	config = {
		"latex": {
			"text.usetex": True,
			"font.family": "serif",
			"font.size": 10,
			"axes.titlesize": 10,
			"axes.labelsize": 10,
			"legend.fontsize": 8,
			"xtick.labelsize": 8,
			"ytick.labelsize": 8,
			"figure.titlesize": 8,
			"figure.facecolor": "#FFFFFF"
		}
	}

	def __init__(self):
		pass

	def set_size(width, aspect_ratio=1.25, fraction=1):
		"""Set figure dimensions to avoid scaling in LaTeX.

		Parameters
		----------
		width: float
		        Document textwidth or columnwidth in pts
		fraction: float, optional
		        Fraction of the width which you wish the figure to occupy

		Returns
		-------
		fig_dim: tuple
		        Dimensions of figure in inches
		"""

		if width == "vanilla_paper":
			width_pt = 472.03123
		elif width == "thesis":
			# width_pt = 443.57848
			width_pt = 426.79135
		elif width == "pat_reg":
			width_pt = 468.3324
		elif width == "pat_reg_dc":
			width_pt = (494.50888/2)-18
		elif width == "beamer":
			width_pt = 307.28987
		elif width == "beamer_169":
			width_pt = 398.3386
		else:
			width_pt = width

		if aspect_ratio == "golden":
			aspect_ratio = 1.61803398875

		# Width of figure (in pts)
		fig_width_pt = width_pt * fraction

		# Convert from pt to inches
		inches_per_pt = 1 / 72.27

		# Golden ratio to set aesthetic figure height
		# https://disq.us/p/2940ij3
		# golden_ratio = (5**.5 - 1) / 2
		# golden_ratio = 1.61803398875
		# golden_ratio = 1.25

		# Figure width in inches
		fig_width_in = fig_width_pt * inches_per_pt
		# Figure height in inches
		fig_height_in = fig_width_in / aspect_ratio

		fig_dim = (fig_width_in, fig_height_in)

		return fig_dim

	# constructs a path, creates the folder if does not exists, clears existing figures if wanted
	def construct_figure_folder_path (parent_path, folder_name, remove_existing):
		new_path = os.path.join(parent_path, folder_name)
		if not os.path.exists(new_path):
			os.mkdir(new_path)
		elif remove_existing:
			existing_figures = glob.glob("{}/*.pdf".format(new_path))
			for figure in existing_figures:
			    os.remove(figure)
		return new_path

	def upsample_coords(coord_list):
		# s is smoothness, set to zero
		# k is degree of the spline. setting to 1 for linear spline
		tck, u = interpolate.splprep(coord_list, k=1, s=0.0)
		upsampled_coords = interpolate.splev(np.linspace(0, 1, 1000), tck)
		return upsampled_coords