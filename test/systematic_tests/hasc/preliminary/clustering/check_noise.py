import glob
import os
from collections import Counter

import numpy as np
from natsort import natsorted

# prepare paths
root_folder_path = os.path.abspath(os.curdir)
test_folder_path = os.path.join(root_folder_path, "test", "systematic_tests", "hasc", "preliminary", "clustering")
figure_folder_path = os.path.join(test_folder_path, "figures")
data_folder_path = os.path.join(root_folder_path, "data", "hasc")

# clean and noisy sequence parent folder paths
discrete_clean_sequence_parent_folder_path = os.path.join(data_folder_path, "0_sequence_discrete")
discrete_noisy_sequence_parent_folder_path = os.path.join(data_folder_path, "0_sequence_discrete_noisy")


clean_sequence_data_paths = natsorted(glob.glob("{}/*HASC*.csv".format(discrete_clean_sequence_parent_folder_path)), reverse=False)
noisy_sequence_data_paths = natsorted(glob.glob("{}/*HASC*.csv".format(discrete_noisy_sequence_parent_folder_path)), reverse=False)

path_couples = list(zip(clean_sequence_data_paths, noisy_sequence_data_paths))

for path_couple in path_couples:

	clean_path, noisy_path = path_couple
	clean_sequence = np.genfromtxt(clean_path, delimiter=",", dtype=int)
	noisy_data = np.genfromtxt(noisy_path, delimiter=",", dtype=int)
	noisy_sequence = noisy_data[:, 0]
	noise_flags = noisy_data[:, 1]

	noisy_observation_indices = np.nonzero(clean_sequence - noisy_sequence)[0]

	clean_freqs = Counter(clean_sequence[noisy_observation_indices])
	noise_freqs = Counter(noisy_sequence[noisy_observation_indices])

	print(noisy_observation_indices.shape)
	print("clean: {}".format(clean_freqs))
	print("noisy: {}".format(noise_freqs))
