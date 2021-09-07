# breathing k-means and davies-bouldin score

import glob
import os

import matplotlib.pyplot as plt
import numpy as np
from clue.evaluate import Evaluator
from natsort import natsorted
from sklearn import preprocessing
from tqdm import tqdm

# prepare paths
root_folder_path = os.path.abspath(os.curdir)
test_folder_path = os.path.join(root_folder_path, "test", "systematic_tests", "hasc", "preliminary", "clustering")
figure_folder_path = os.path.join(test_folder_path, "figures")
data_folder_path = os.path.join(root_folder_path, "data", "hasc")
sequence_parent_folder_path = os.path.join(data_folder_path, "0_sequence")

config = {
	"algorithm": "bkmeans",
	"parameters": {
		"n_clusters": np.arange(4,21).tolist(),
		"m": 5,
		"theta": 1.1
	}
}

# sequence paths
sequence_data_paths = natsorted(glob.glob("{}/*/HASC*.csv".format(sequence_parent_folder_path)), reverse=False)

with tqdm(
	total = len(sequence_data_paths) * len(config["parameters"]["n_clusters"]),
	position = 0,
	leave = True,
	desc = "DBS".ljust(10),
	bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}',
	ascii = True
	) as pbar:

	for sequence_data_path in sequence_data_paths:

		# get person and experiment id
		person_id = sequence_data_path.rsplit("/", 2)[1]
		sequence_data_id = sequence_data_path.rsplit("/", 1)[1].split(".")[0]

		# print(person_id, sequence_data_id)

		# read raw data (including time stamps)
		raw_data = np.genfromtxt(sequence_data_path, usecols=(0, 1, 2, 3), delimiter=",")
		# print(raw_data.shape)

		# downsample
		# downsample_rate = 3
		# raw_data = raw_data[::downsample_rate,:]

		# downsampling alternative, applies a low-pass filter first
		# raw_data = signal.decimate(raw_data, q=3, n=60, ftype="fir", axis=0)

		time_stamps = raw_data[:, 0]
		X = raw_data[:, 1:]

		# round time stamps to make it easier to compare with label ranges
		sequence_times = np.round(time_stamps, 3)

		# apply normalization
		# X = preprocessing.normalize(X)

		clue_ins = Evaluator(
			metrics_list = [
				"dbs",
				# "chs",
				# "sil",
				# "rerr"
			]
		)

		if config["algorithm"] == "bkmeans":

			from bkmeans import BKMeans

			for k in config["parameters"]["n_clusters"]:

				current_config = {
					"algorithm": "bkmeans",
					"parameters": {
						"n_clusters": k,
						"m": 4,
						"theta": 1.1
					}
				}
				clusterer = BKMeans(**current_config["parameters"], random_state=0)
				clusterer.fit(X)

				clue_ins.evaluate(
					data = X, 
					labels = clusterer.labels_, 
					clustering_config = current_config
				)

				pbar.update()

			fig_path = "{}/{}_{}.pdf".format(
				figure_folder_path,
				person_id,
				sequence_data_id
			)
			
			# clue_ins.report("n_clusters")
			clue_ins.plot_values("n_clusters", "dbs", fig_path)
