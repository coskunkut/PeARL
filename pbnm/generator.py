import os
import numpy as np
import more_itertools as mit

class Multinomial:

	# p: parameter
	# d: duration
	# n: number of regimes

	def __init__(self, alp_car, rng_seed):
		self.set_rng(rng_seed)
		self.alp_car = alp_car
		self.reset()

	def set_rng(self, rng_seed):
		self.rng = np.random.default_rng(rng_seed)

	def reset(self):
		self.o_sequence = []
		self.p_sequence = []
		self.p = []
		self.noise_flags = []
		self.omega_list = []
		self.noise_regimes = {}

	# generates P matrix with difference of each consecutive rows controlled
	# - n: number of rows
	# - min_diff: minimum value for P[i]-P[i-1]
	# - max_diff: maximum value for P[i]-P[i-1]
	def multinomial_p_with_diff(self, n, min_diff, max_diff):
		ps = np.array([]).reshape((0, self.alp_car))
		while ps.shape[0] < n:
			ok = False
			not_ok_count = 0
			while not ok:

				# generate next p
				p_next = self.rng.dirichlet(np.repeat(1, self.alp_car))

				# first p is always accepted
				if ps.shape[0] == 0:
					ps = np.vstack((ps, p_next))
					ok = True
					continue

				# measure the difference between most recent ps
				current_difference = np.abs(p_next - ps[-1])
				if np.mean(current_difference) >= min_diff and np.mean(current_difference) <= max_diff:
				# if np.min(current_difference) >= min_diff and np.max(current_difference) <= max_diff:
				# if np.max(current_difference) > min_diff:
					ps = np.vstack((ps, p_next))
					ok = True
				else:
					not_ok_count += 1
					if not_ok_count > 1000:
						# ps = np.array([]).reshape((0, self.alp_car))
						ps = np.delete(ps, -1, 0)
						# print("NOTOK limit reached.")
						break

		return ps.tolist()

	# generates d[i]-long regimes with parameter p[i]
	def sudden_drift_pd(self, p, d, append=False):
		if not append:
			self.reset()
		self.p.extend(p)
		self.p_sequence.extend(np.repeat(p, d, axis=0).tolist())
		n = len(p)
		if not isinstance(d, list):
			d = np.repeat(d, n)
		for regime_count in range(n):
			subsequence = self.rng.choice(
				a = len(p[0]),
				size = d[regime_count],
				p = p[regime_count]
			)
			self.o_sequence.extend([str(o) for o in subsequence])
		self.noise_flags.extend(np.zeros(np.sum(d), dtype=int))

	def sudden_drift_diff(self, n, d, min_diff, max_diff, append=False):
		if not append:
			self.reset()
		p = self.multinomial_p_with_diff(n, min_diff, max_diff)
		self.sudden_drift_pd(p, d, append)

	def noise_sudden_drift(self, n, d, delta_min, delta_max):
		omega_initial = self.rng.random()
		omega_list = [omega_initial]
		for regime_count in range(n):
			omega_next = self.rng.random()
			omega_diff = np.abs(omega_next - omega_list[-1])
			while omega_diff < delta_min or omega_diff > delta_max:
				omega_next = self.rng.random()
				omega_diff = np.abs(omega_next - omega_list[-1])
			omega_list.append(omega_next)
		for regime_count in range(n):
			start_index = regime_count * d
			stop_index = start_index + d
			# print(start_index, stop_index, omega_list[regime_count])
			self.noise_regimes["{}_{}".format(start_index, stop_index)] = omega_list[regime_count]
			self.inject_white_noise(omega_list[regime_count], start_index, stop_index)

	# injects white noise to the sequence with a predetermined rate
	def inject_white_noise(self, noise_rate, start_index=None, stop_index=None):
		noisy_subsequence = []
		subsequence_noise_flags = []
		if start_index is None:
			start_index = 0
		if stop_index is None:
			stop_index = len(self.o_sequence)
		for index, token in enumerate(self.o_sequence[start_index:stop_index]):
			if self.rng.random() <= noise_rate:
				prime_list = [str(i) for i in range(self.alp_car) if str(i) != token]
				noisy_subsequence.append(self.rng.choice(prime_list))
				subsequence_noise_flags.append(1)
			else:
				noisy_subsequence.append(token)
				subsequence_noise_flags.append(0)
		self.o_sequence[start_index:stop_index] = noisy_subsequence.copy()
		if len(self.noise_flags) == 0:
			self.noise_flags = np.zeros(len(self.o_sequence), dtype=int).tolist()
		self.noise_flags[start_index:stop_index] = subsequence_noise_flags.copy()

	def estimate_noise_rate(self):
		return np.sum(self.noise_flags) / len(self.noise_flags)

	# calculates the difference between clean and noisy sequence, returns estimated noise rate
	def estimate_noise_rate_diff (self):
		sequence_diff = np.subtract(self.sequence, self.clean_sequence)
		diff_count = np.count_nonzero(sequence_diff == 0)
		return 1 - (diff_count / self.sequence_length)

	# # generates a non-stationarity sequence using the regime information
	# def generate_clean_sequence (self, p, d):
	# 	self.p = p
	# 	self.d = d
	# 	self.alphabet_cardinality = len(p[0])
	# 	self.n = len(p)
	# 	clean_sequence = []
	# 	for regime_count in range(self.n):
	# 		clean_subsequence = self.rng.choice(
	# 			a = self.alphabet_cardinality,
	# 			size = d[regime_count],
	# 			p = p[regime_count]
	# 		)
	# 		clean_sequence.extend(clean_subsequence)
	# 	self.sequence = clean_sequence.copy()
	# 	# self.noise_flags = [0 for token_count in range(self.sequence_length)]
