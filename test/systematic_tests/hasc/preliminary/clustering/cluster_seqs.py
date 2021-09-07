import glob
import os

import numpy as np
from bkmeans import BKMeans
from natsort import natsorted
from tqdm import tqdm

# prepare paths
root_folder_path = os.path.abspath(os.curdir)
test_folder_path = os.path.join(root_folder_path, "test", "systematic_tests", "hasc", "preliminary", "clustering")
figure_folder_path = os.path.join(test_folder_path, "figures")
data_folder_path = os.path.join(root_folder_path, "data", "hasc")

# clean
clean_sequence_parent_folder_path = os.path.join(data_folder_path, "0_sequence")
discrete_clean_sequence_parent_folder_path = os.path.join(data_folder_path, "0_sequence_discrete")

# noisy
noisy_sequence_parent_folder_path = os.path.join(data_folder_path, "0_sequence_noisy")
discrete_noisy_sequence_parent_folder_path = os.path.join(data_folder_path, "0_sequence_discrete_noisy")

config = {
	"HASC1001": 4,
	"HASC1002": 5,
	"HASC1003": 4,
	"HASC1004": 6,
	"HASC1005": 6,
	"HASC1006": 8,
	"HASC1007": 7,
	"HASC1008": 5,
	"HASC1009": 4,
	"HASC1010": 4,
	"HASC1011": 10,
	"HASC1012": 8,
	"HASC1013": 4,
	"HASC1014": 5,
	"HASC1015": 5,
	"HASC1016": 4,
	"HASC1017": 5,
	"HASC1018": 4,
}

# sequence paths
clean_sequence_data_paths = natsorted(glob.glob("{}/*/HASC*.csv".format(clean_sequence_parent_folder_path)), reverse=False)
noisy_sequence_data_paths = natsorted(glob.glob("{}/*/HASC*.csv".format(noisy_sequence_parent_folder_path)), reverse=False)
path_couples = list(zip(clean_sequence_data_paths, noisy_sequence_data_paths))

# print(path_couples)

for path_couple in tqdm(path_couples):

	clean_path, noisy_path = path_couple
	clean_data = np.genfromtxt(clean_path, usecols=(0, 1, 2, 3), delimiter=",")
	noisy_data = np.genfromtxt(noisy_path, usecols=(0, 1, 2, 3, 4), delimiter=",")

	time_stamps = clean_data[:, 0]
	X_clean = clean_data[:, 1:4]
	X_noisy = noisy_data[:, 1:4]
	noise_flags = noisy_data[:, 4]

	# get person and experiment id
	person_id = clean_path.rsplit("/", 2)[1]
	sequence_data_id = clean_path.rsplit("/", 1)[1].split(".")[0]
	observation_count = len(time_stamps)

	# print(person_id, sequence_data_id)

	current_config = {
		"algorithm": "bkmeans",
		"parameters": {
			"n_clusters": config[str(sequence_data_id)],
			"m": 4,
			"theta": 1.1
		}
	}
	clusterer = BKMeans(**current_config["parameters"], random_state=0)
	clusterer.fit(X_clean)

	# discrete_clean_data = np.zeros([observation_count], dtype=int)
	# discrete_clean_data[:, 0] = clusterer.labels_
	# discrete_clean_data[:, 1] = noise_flags
	discrete_clean_sequence = clusterer.predict(X_clean)
	discrete_clean_data_path = "{}/{}_{}.csv".format(discrete_clean_sequence_parent_folder_path, person_id, sequence_data_id)
	np.savetxt(discrete_clean_data_path, discrete_clean_sequence, delimiter=",", fmt="%d")

	discrete_noisy_data = np.zeros([observation_count, 2], dtype=int)
	discrete_noisy_data[:, 0] = clusterer.predict(X_noisy)
	discrete_noisy_data[:, 1] = noise_flags
	discrete_noisy_data_path = "{}/{}_{}.csv".format(discrete_noisy_sequence_parent_folder_path, person_id, sequence_data_id)
	np.savetxt(discrete_noisy_data_path, discrete_noisy_data, delimiter=",", fmt="%d")
