import numpy as np
from tqdm import tqdm

class SLWE:

	def __init__(self, alphabet_cardinality, learning_coefficient):

		# alphabet cardinality must be known to initialize estimate vectors
		self.alphabet_cardinality = alphabet_cardinality

		# the corresponding index for each token is stored in this dict.
		self.token_index_map = {}

		# AKA: lambda
		self.learning_coefficient = learning_coefficient

		# estimate history
		self.initial_estimate = np.ones(self.alphabet_cardinality) / self.alphabet_cardinality
		self.estimates = []

	# SLWE update
	def LRI_update (self, token_index):

		# get the most recent estimate
		if len(self.estimates) == 0:
			most_recent_estimate = self.initial_estimate
		else:
			most_recent_estimate = self.estimates[-1]

		# apply rewarding to relevant probability
		new_estimate = np.copy(most_recent_estimate)
		new_estimate[token_index] = most_recent_estimate[token_index] + ((1 - self.learning_coefficient) * (1 - most_recent_estimate[token_index]))

		# apply penalty to irrelevant probabilities
		irrelevant_indices = [i for i in np.arange(self.alphabet_cardinality) if i != token_index]
		new_estimate[irrelevant_indices] *= self.learning_coefficient

		# append the current estimate to estimate history
		self.estimates.append(new_estimate.tolist())

	# manages token->index mapping
	# - assigns a new index to given token if it has not been obverved before
	# - returns index
	def get_index(self, token):
		if len(self.token_index_map) == 0:
			self.token_index_map[token] = 0
		elif token not in self.token_index_map:
			if len(self.token_index_map) >= self.alphabet_cardinality:
				print("Unexpected token observed: {}".format(token))
				sys.exit()
			self.token_index_map[token] = max(self.token_index_map.values()) + 1
		index = self.token_index_map[token]
		return index

	def process_token(self, token):
		index = self.get_index(token)
		self.LRI_update(index)

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

class SLWE_P:

	def __init__(self, alphabet_cardinality, rho_eta_mapping="event_linear"):

		# alphabet cardinality must be known to initialize estimate vectors
		self.alphabet_cardinality = alphabet_cardinality

		# rho to eta mapping type
		self.rho_eta_mapping = rho_eta_mapping

		# the corresponding index for each token is stored in this dict.
		self.token_index_map = {}

		# estimate history
		self.initial_estimate = np.ones(self.alphabet_cardinality) / self.alphabet_cardinality
		self.estimates = []

		# persistence related
		self.eta_min = 0.01
		self.eta_max = 0.10
		self.k = 15
		self.L = 15
		self.l = np.zeros(self.alphabet_cardinality, dtype=int)
		self.memory = [] # this is index memory
		self.rho = []

		# AKA: lambda
		self.eta = self.eta_max
		self.eta_history = []

	# manages token->index mapping
	# - assigns a new index to given token if it has not been obverved before
	# - returns index
	def get_index(self, token):
		if len(self.token_index_map) == 0:
			self.token_index_map[token] = 0
		elif token not in self.token_index_map:
			if len(self.token_index_map) >= self.alphabet_cardinality:
				print("Unexpected token observed: {}".format(token))
				sys.exit()
			self.token_index_map[token] = max(self.token_index_map.values()) + 1
		index = self.token_index_map[token]
		return index

	# update memory, occurrence frequencies and persistence
	def calculate_rho(self, token_index):
		self.memory.append(token_index)
		self.l[token_index] += 1
		if len(self.memory) > self.L:
			oldest_index = self.memory.pop(0)
			self.l[oldest_index] -= 1
		rho = (np.power(self.k, np.divide(self.l, self.L)) - 1) / (self.k - 1)
		self.rho.append(rho.tolist())

	# calculate the new value of eta by a linear mapping to rho
	def rho_to_eta_event_linear(self, token_index):
		bound_diff = self.eta_max - self.eta_min
		self.eta = (self.rho[-1][token_index] * bound_diff) + self.eta_min
		self.eta_history.append(self.eta)

	# linear mapping from rho to eta, but min(rho) is E[rho] of white noise
	def rho_to_eta_avg_linear(self, token_index):
		avg_rho_min = (np.power(self.k, 1/self.alphabet_cardinality) - 1) / (self.k - 1)
		avg_rho_max = 1/self.alphabet_cardinality
		eta_bound_diff = self.eta_max - self.eta_min
		rho_bound_diff = avg_rho_max - avg_rho_min
		self.eta = self.eta_min + (((self.rho[-1][token_index] - avg_rho_min) * eta_bound_diff) / rho_bound_diff)
		self.eta_history.append(self.eta)

	# SLWE update
	def LRI_update (self, token_index):

		# get the most recent estimate
		if len(self.estimates) == 0:
			most_recent_estimate = self.initial_estimate
		else:
			most_recent_estimate = self.estimates[-1]

		# apply rewarding to relevant probability
		new_estimate = np.copy(most_recent_estimate)
		new_estimate[token_index] = most_recent_estimate[token_index] + (self.eta * (1 - most_recent_estimate[token_index]))

		# apply penalty to irrelevant probabilities
		irrelevant_indices = [i for i in np.arange(self.alphabet_cardinality) if i != token_index]
		new_estimate[irrelevant_indices] *= (1 - self.eta)

		# append the current estimate to estimate history
		self.estimates.append(new_estimate.tolist())

	# processes a given token
	def process_token(self, token):
		index = self.get_index(token)
		self.calculate_rho(index)
		if self.rho_eta_mapping == "event_linear":
			self.rho_to_eta_event_linear(index)
		elif self.rho_eta_mapping == "avg_linear":
			self.rho_to_eta_avg_linear(index)
		else:
			print("unknown mapping type")
			exit()
		self.LRI_update(index)

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