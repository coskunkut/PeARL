import glob
import os
from pathlib import Path

import numpy as np
from natsort import natsorted

rng = np.random.default_rng(1123)
noise_rate = 0.2

# prepare paths
root_folder_path = os.path.abspath(os.curdir)
test_folder_path = os.path.join(root_folder_path, "test", "systematic_tests", "hasc", "preliminary", "clustering")
figure_folder_path = os.path.join(test_folder_path, "figures")
data_folder_path = os.path.join(root_folder_path, "data", "hasc")
sequence_parent_folder_path = os.path.join(data_folder_path, "0_sequence")
noisy_sequence_parent_folder_path = os.path.join(data_folder_path, "0_sequence_noisy")

# sequence paths
sequence_data_paths = natsorted(glob.glob("{}/*/HASC*.csv".format(sequence_parent_folder_path)), reverse=False)

for sequence_data_path in sequence_data_paths:

		# get person and experiment id
		person_id = sequence_data_path.rsplit("/", 2)[1]
		sequence_data_id = sequence_data_path.rsplit("/", 1)[1].split(".")[0]

		# read raw data (including time stamps)
		raw_data = np.genfromtxt(sequence_data_path, usecols=(0, 1, 2, 3), delimiter=",")

		print(person_id, sequence_data_id, raw_data.shape)

		time_stamps = raw_data[:, 0]
		X = raw_data[:, 1:]

		observation_count = X.shape[0]

		mins = np.min(X, axis=0)
		maxs = np.max(X, axis=0)
		bounds = list(zip(mins, maxs))

		noisy_instance_count = int(np.ceil(observation_count * noise_rate))
		noisy_instance_indices = np.random.choice(observation_count, noisy_instance_count, replace=False)
		print(noisy_instance_count)

		white_noise = np.zeros([noisy_instance_count, 3])
		for dim in range(3):
			white_noise[:, dim] += rng.uniform(bounds[dim][0], bounds[dim][1], noisy_instance_count)
		# print(white_noise.shape)

		X[noisy_instance_indices, :] += white_noise

		noise_flags = np.zeros([observation_count], dtype=int)
		noise_flags[noisy_instance_indices] += 1

		X_new = np.zeros([observation_count, 5])
		X_new[:, 0] = time_stamps
		X_new[:, 1:4] = X
		X_new[:, 4] = noise_flags

		folder_path = os.path.join(noisy_sequence_parent_folder_path, person_id)
		Path(folder_path).mkdir(parents=True, exist_ok=True)
		noisy_sequence_file_path = "{}/{}.csv".format(folder_path, sequence_data_id)
		np.savetxt(noisy_sequence_file_path, X_new, delimiter=",", fmt="%.6f")
