import numpy as np
from tqdm import tqdm

class SLWE:

	def __init__(self, alphabet_cardinality, learning_coefficient):

		# alphabet cardinality must be known to initialize estimate vectors
		self.alphabet_cardinality = alphabet_cardinality

		# AKA: lambda
		self.learning_coefficient = learning_coefficient

		# estimate history
		self.initial_estimate = np.ones(self.alphabet_cardinality) / self.alphabet_cardinality
		self.estimates = []

	# SLWE update
	def LRI_update (self, token):

		# get the most recent estimate
		if len(self.estimates) == 0:
			most_recent_estimate = self.initial_estimate
		else:
			most_recent_estimate = self.estimates[-1]

		# apply rewarding to relevant probability
		new_estimate = np.copy(most_recent_estimate)
		new_estimate[token] = most_recent_estimate[token] + ((1 - self.learning_coefficient) * (1 - most_recent_estimate[token]))

		# apply penalty to irrelevant probabilities
		irrelevant_indices = [i for i in np.arange(self.alphabet_cardinality) if i != token]
		new_estimate[irrelevant_indices] *= self.learning_coefficient

		# append the current estimate to estimate history
		self.estimates.append(new_estimate.tolist())

	def process_token(self, token):
		self.LRI_update(token)

	# processes each token in a given sequence, shows progress information
	def process_sequence(self, sequence, progress=False, progress_desc="", progress_desc_width=8):
		sequence_length = len(sequence)
		if progress:
			for token_count in tqdm(
					range(sequence_length), 
					desc = progress_desc.ljust(progress_desc_width),
					bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}',
					ascii = True
				):
				self.LRI_update(sequence[token_count])
		else:
			for token_count in range(sequence_length):
				self.LRI_update(sequence[token_count])

class SLWE_P:

	def __init__(self, alphabet_cardinality):

		# alphabet cardinality must be known to initialize estimate vectors
		self.alphabet_cardinality = alphabet_cardinality

		# estimate history
		self.initial_estimate = np.ones(self.alphabet_cardinality) / self.alphabet_cardinality
		self.estimates = []

		# persistence related
		self.eta_min = 0.01
		self.eta_max = 0.10
		self.k = 15
		self.L = 15
		self.l = np.zeros(self.alphabet_cardinality, dtype=int)
		self.token_memory = []
		self.rho = []

		# AKA: lambda
		self.eta = self.eta_max
		self.eta_history = []

	# SLWE update
	def LRI_update (self, token):

		# get the most recent estimate
		if len(self.estimates) == 0:
			most_recent_estimate = self.initial_estimate
		else:
			most_recent_estimate = self.estimates[-1]

		# apply rewarding to relevant probability
		new_estimate = np.copy(most_recent_estimate)
		new_estimate[token] = most_recent_estimate[token] + (self.eta * (1 - most_recent_estimate[token]))

		# apply penalty to irrelevant probabilities
		irrelevant_indices = [i for i in np.arange(self.alphabet_cardinality) if i != token]
		new_estimate[irrelevant_indices] *= (1 - self.eta)

		# append the current estimate to estimate history
		self.estimates.append(new_estimate.tolist())

	# update token memory, occurrence frequencies and persistence
	def calculate_rho(self, token):
		self.token_memory.append(token)
		self.l[token] += 1
		if len(self.token_memory) > self.L:
			oldest_token = self.token_memory.pop(0)
			self.l[oldest_token] -= 1
		rho = (np.power(self.k, np.divide(self.l, self.L)) - 1) / (self.k - 1)
		self.rho.append(rho.tolist())

	# calculate the new value of eta by a linear mapping to rho
	def rho_to_eta_linear(self, token):
		bound_diff = self.eta_max - self.eta_min
		self.eta = (self.rho[-1][token] * bound_diff) + self.eta_min
		self.eta_history.append(self.eta)

	# processes a given token
	def process_token(self, token):
		self.calculate_rho(token)
		self.rho_to_eta_linear(token)
		self.LRI_update(token)

	# processes each token in a given sequence, shows progress information
	def process_sequence(self, sequence, progress=False, progress_desc="", progress_desc_width=8):
		sequence_length = len(sequence)
		if progress:
			for token_count in tqdm(
					range(sequence_length), 
					desc = progress_desc.ljust(progress_desc_width),
					bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}',
					ascii = True
				):
				self.process_token(sequence[token_count])
		else:
			for token_count in range(sequence_length):
				self.process_token(sequence[token_count])