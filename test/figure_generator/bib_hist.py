import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from pearl.tools.visualizer import get_matplotlib_config, set_size
from matplotlib import cm

years = [2008, 2019, 2015, 2006, 1967, 2010, 2013, 1991, 2017, 2019, 2019, 2020, 2003, 2012, 2005, 2007, 2011, 2020,
         2022, 2022, 2015, 2011, 1995, 2013, 2005, 2015, 2018, 2022, 2022, 2023, 2011, 2011, 2020, 1979]

types = {
	"Journal Article": 16,
	"Conference Paper": 14,
	"Book": 2,
	"Miscellaneous": 2
}

def absolute_value(val):
    return int(np.round(val/100.*np.sum(list(types.values())), 0))

# build paths
root_folder = os.path.abspath(os.curdir)
figure_folder = os.path.join(root_folder, "test", "figure_generator")
Path(figure_folder).mkdir(parents=True, exist_ok=True)

d = len(set(years)),
counts, bins = np.histogram(years, bins=np.arange(1965, 2030, 5))
# print(counts, bins)

# matplotlib config
plt.rcParams.update(get_matplotlib_config("generic_latex"))

# visualization
fig, ax = plt.subplots(
	nrows=1,
	ncols=1,
	figsize=set_size(
		width="pat_reg_dc",
		aspect_ratio=1.5,
		fraction=1
	)
)
fig.set_tight_layout(True)

plt.stairs(
	values=counts,
	edges=bins,
	fill=True,
	color="steelblue",
	zorder=2
)

ax.set_xticks(bins)
plt.xticks(rotation=45)
ax.set_xlabel("Years")
ax.set_ylabel("Number of Citations")
ax.grid(linestyle="dashed", color="lightgray", alpha=1.00, zorder=0)

figure_address = "{0}/bib_hist.{1}".format(figure_folder, "pdf")
fig.savefig(figure_address, bbox_inches="tight", pad_inches=0.05)
plt.cla()

plt.pie(
	x=types.values(),
	labels=types.keys(),
	colors=cm.Pastel1([0,1,2,3]),
	# explode=[0.1, 0, 0, 0],
	explode=np.repeat(0.02, len(types)),
	autopct=absolute_value
)
ax.axis("equal")
figure_address = "{0}/bib_pie.{1}".format(figure_folder, "pdf")
fig.savefig(figure_address, bbox_inches="tight", pad_inches=0.05)
