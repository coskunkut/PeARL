import glob
import os

import numpy as np
from scipy import interpolate


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

def get_matplotlib_config(config_name):
	if config_name == "generic_latex":
		matplotlib_config = {
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
	return matplotlib_config

def set_size(width, aspect_ratio=1.25, fraction=1):
	"""Set figure dimensions to avoid scaling in LaTeX.

	Parameters
	----------
	width: float, string
			Document textwidth or columnwidth in pts
	fraction: float, optional
			Fraction of the width which you wish the figure to occupy

	Returns
	-------
	fig_dim: tuple
			Dimensions of figure in inches
	"""

	if width == "generic_paper":
		width_pt = 472.03123
	elif width == "thesis":
		# width_pt = 443.57848
		width_pt = 426.79135
	elif width == "pat_reg":
		width_pt = 468.3324
	elif width == "pat_reg_dc":
		width_pt = (494.50888/2)-18
	elif width == "lncp":
		width_pt = 347.12354
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
