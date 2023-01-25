import numpy as np
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
		time_series_pieces = list(grouper(data, piece_length))
	return time_series_pieces

# calculates the difference between clean and noisy sequence, returns estimated noise rate
def get_effective_noise_rate(clean_sequence, noisy_sequence):
	sequence_diff = np.subtract(clean_sequence, noisy_sequence)
	diff_count = np.count_nonzero(sequence_diff == 0)
	return 1 - (diff_count / len(clean_sequence))