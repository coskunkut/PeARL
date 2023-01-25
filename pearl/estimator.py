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
		self.estimates = [self.initial_estimate]

		# symbol to index and index to symbol maps
		self.symbol_index_map = {}
		self.index_symbol_map = {}

	# SLWE update
	def update_estimate(self, symbol_index):

		# multiply the estimate vector with scalar lambda
		new_estimate = self.estimates[-1] * self.learning_coefficient

		# apply rewarding to relevant probability
		new_estimate[symbol_index] += (1 - self.learning_coefficient)

		# append the current estimate to estimate history
		self.estimates.append(new_estimate)

	# manages symbol->index and index->symbol mappings
	# - assigns a new index to given symbol if it has not been observed before
	# - returns index
	def get_index(self, symbol):
		if len(self.symbol_index_map) == 0:
			self.symbol_index_map[symbol] = 0
			self.index_symbol_map[0] = symbol
		elif symbol not in self.symbol_index_map:
			if len(self.symbol_index_map) >= self.alphabet_cardinality:
				print("Unexpected symbol observed: {}".format(symbol))
				exit()
			next_available_index = max(self.symbol_index_map.values()) + 1
			self.symbol_index_map[symbol] = next_available_index
			self.index_symbol_map[next_available_index] = symbol
		index = self.symbol_index_map[symbol]
		return index

	# processes a single symbol
	def process_symbol(self, symbol):
		index = self.get_index(symbol)
		self.update_estimate(index)

	# processes each symbol in a given sequence, shows progress information
	def process_sequence(self, sequence, progress=False, progress_desc="", progress_desc_width=8):
		if progress:
			pbar_conf = {
				"total": len(sequence),
				"desc": progress_desc.ljust(progress_desc_width),
				"bar_format": "{l_bar}{bar:10}{r_bar}{bar:-10b}",
				"ascii": True
			}
			with tqdm(**pbar_conf) as pbar:
				for symbol in sequence:
					self.process_symbol(symbol)
					pbar.update()
		else:
			for symbol in sequence:
				self.process_symbol(symbol)

class PeARL:

	def __init__(
			self,
			alphabet_cardinality,
			eta_max=0.1,
			eta_min=0.01,
			k=15,
			L=15,
			rho_eta_mapping="event_linear"
		):

		# alphabet cardinality must be known to initialize estimate vectors
		self.alphabet_cardinality = alphabet_cardinality

		# estimate history
		self.initial_estimate = np.ones(self.alphabet_cardinality) / self.alphabet_cardinality
		self.estimates = [self.initial_estimate]

		# symbol to index and index to symbol maps
		self.symbol_index_map = {}
		self.index_symbol_map = {}

		# persistence related
		self.rho_eta_mapping = rho_eta_mapping
		self.eta_max = eta_max
		self.eta_min = eta_min
		self.eta = self.eta_max
		self.eta_history = []
		self.k = k
		self.L = L
		self.l = np.zeros(self.alphabet_cardinality, dtype=int)
		self.memory = [] # this is index memory
		self.rho = []

	# manages symbol->index and index->symbol mappings
	# - assigns a new index to given symbol if it has not been observed before
	# - returns index
	def get_index(self, symbol):
		if len(self.symbol_index_map) == 0:
			self.symbol_index_map[symbol] = 0
			self.index_symbol_map[0] = symbol
		elif symbol not in self.symbol_index_map:
			if len(self.symbol_index_map) >= self.alphabet_cardinality:
				print("Unexpected symbol observed: {}".format(symbol))
				exit()
			next_available_index = max(self.symbol_index_map.values()) + 1
			self.symbol_index_map[symbol] = next_available_index
			self.index_symbol_map[next_available_index] = symbol
		index = self.symbol_index_map[symbol]
		return index

	# update memory, occurrence frequencies and persistence
	def calculate_rho(self, symbol_index):
		self.memory.append(symbol_index)
		self.l[symbol_index] += 1
		if len(self.memory) > self.L:
			oldest_index = self.memory.pop(0)
			self.l[oldest_index] -= 1
		rho = (np.power(self.k, np.divide(self.l, self.L)) - 1) / (self.k - 1)
		self.rho.append(rho.tolist())

	# calculate the new value of eta by a linear mapping to rho
	def rho_to_eta_event_linear(self, symbol_index):
		bound_diff = self.eta_max - self.eta_min
		self.eta = (self.rho[-1][symbol_index] * bound_diff) + self.eta_min
		self.eta_history.append(self.eta)

	# linear mapping from rho to eta, but min(rho) is E[rho] of white noise
	def rho_to_eta_avg_linear(self, symbol_index):
		avg_rho_min = (np.power(self.k, 1/self.alphabet_cardinality) - 1) / (self.k - 1)
		avg_rho_max = 1/self.alphabet_cardinality
		eta_bound_diff = self.eta_max - self.eta_min
		rho_bound_diff = avg_rho_max - avg_rho_min
		self.eta = self.eta_min + (((self.rho[-1][symbol_index] - avg_rho_min) * eta_bound_diff) / rho_bound_diff)
		self.eta_history.append(self.eta)

	# SLWE update
	def update_estimate(self, symbol_index):

		# multiply the estimate vector with scalar lambda
		new_estimate = self.estimates[-1] * (1 - self.eta)

		# apply rewarding to relevant probability
		new_estimate[symbol_index] += self.eta

		# append the current estimate to estimate history
		self.estimates.append(new_estimate)

	# processes a single symbol
	def process_symbol(self, symbol):
		index = self.get_index(symbol)
		self.calculate_rho(index)
		if self.rho_eta_mapping == "event_linear":
			self.rho_to_eta_event_linear(index)
		elif self.rho_eta_mapping == "avg_linear":
			self.rho_to_eta_avg_linear(index)
		else:
			print("unknown mapping type")
			exit()
		self.update_estimate(index)

	# processes each symbol in a given sequence, shows progress information
	def process_sequence(self, sequence, progress=False, progress_desc="", progress_desc_width=8):
		if progress:
			pbar_conf = {
				"total": len(sequence),
				"desc": progress_desc.ljust(progress_desc_width),
				"bar_format": "{l_bar}{bar:10}{r_bar}{bar:-10b}",
				"ascii": True
			}
			with tqdm(**pbar_conf) as pbar:
				for symbol in sequence:
					self.process_symbol(symbol)
					pbar.update()
		else:
			for symbol in sequence:
				self.process_symbol(symbol)
