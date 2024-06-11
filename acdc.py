import torch
from graph import ComputationalGraph, Edge
from jaxtyping import Float, Int, Bool
from typing import List, Optional, Callable, Tuple, Dict, Literal, Set, Union
from functools import partial
from ioi_dataset import IOIDataset
from torch import Tensor
from transformer_lens.hook_points import HookPoint


class ACDC:
	def __init__(self, config):
		'''
		Initialize the ACDC object.

		Args:
			config: A dictionary containing the configuration parameters.
				- model: The model to be used for the algorithm.
				- n_heads: The number of attention heads in the model.
				- n_layers: The number of layers in the model.
				- clean_cache: The cache of clean activations.
				- corrupted_cache: The cache of corrupted activations.
				- clean_dataset: The clean dataset.
				- corrupted_dataset: The corrupted dataset.
				- clean_logits: The clean logits.
				- n_epochs: The number of epochs to run the algorithm. Default is 1.
				- verbose: Whether to print verbose output. Default is True.
				- heads: A list of specific heads to consider. If None, all heads are used.
		'''
		self.model = config["model"]
		self.clean_cache = config["clean_cache"]
		self.corrupted_cache = config["corrupted_cache"]
		self.n_heads = config["n_heads"]
		self.n_layers = config["n_layers"]
		
		if "heads" in config:
			self.heads = sorted(config["heads"], key=lambda x: (x[0], x[1]))
		else:
			self.heads = [(i, j) for i in range(self.n_layers) for j in range(self.n_heads)]
		
		self.clean_dataset = config["clean_dataset"]
		self.corrupted_dataset: IOIDataset = config["corrupted_dataset"]
		self.clean_logits: Optional[Float[Tensor, "batch seq d_vocab"]] = config["clean_logits"]
		
		self.original_graph = ComputationalGraph(self.n_heads, self.n_layers, self.heads)
		self.pruned_graph = ComputationalGraph(self.n_heads, self.n_layers, self.heads)
		
		self.kl_threshold = 0.0575 # Threshold for KL divergence from ACDC paper (see Appendix D)
		self.dead_edges: List[Edge] = []
		self.num_epochs = config.get("n_epochs", 1)
		self.verbose = config.get("verbose", True)
		
	def run(self) -> List[Edge]:
		'''
		Performs the ACDC algorithm using the model and caches provided.
		The ACDC algorithm is a method for pruning unnecessary edges in a computational graph.
		Rather than actively pruning edges, we accumulate "dead edges" which are corrupted at runtime.
		The final graph is the original graph minus the dead edges.
		'''
		for _ in range(self.num_epochs):
			for receiver in self.original_graph.graph.keys():
				incoming_edges = self.original_graph.get_incoming_edges(receiver)
				if len(incoming_edges) == 0:
					continue
				if self.verbose:
					print('Receiver: {}'.format(receiver))
				for edge in incoming_edges:
					# Take the KL divergence of the graph without the edge removed
					test_dead_edges = self.dead_edges[:]
					kl_div_prev = self.ablate_paths(test_dead_edges)

					# Add the edge to the dead edges and take the KL divergence
					test_dead_edges.append(edge)
					kl_div_new = self.ablate_paths(test_dead_edges)

					# If the difference in KL divergence is less than the threshold, add the edge to the dead edges
					kl_div_diff = kl_div_new - kl_div_prev
					if kl_div_diff < self.kl_threshold:
						self.dead_edges.append(edge)
					# Otherwise, it is a valuable edge, and we keep it
					else:
						if self.verbose:
							print('Good edge!\n{}\nKL Divergence Diff: {} > {}\n'.format(edge, round(kl_div_diff, 4), self.kl_threshold))
				if self.verbose:
					print()		
					print('-'*50)
					print()
		if self.verbose:
			print('Edges at start: {}'.format(len(self.original_graph.get_all_edges())))
			print('Edges to remove: {}'.format(len(self.dead_edges)))
		for edge in self.dead_edges:
			self.pruned_graph.remove_edge(edge)
		good_edges = self.pruned_graph.get_all_edges()
		if self.verbose:
			print('Edges at end: {}'.format(len(self.pruned_graph.get_all_edges())))
			print('Final edges: {}'.format(len(good_edges)))
		self.pruned_graph.rename_nodes()
		self.pruned_graph.clean_dead_edges()
		self.pruned_graph.visualize()
		return good_edges		

	def patch_path(
		self,
		activation: Float[Tensor, "batch pos head_idx d_model"],
		hook: HookPoint,
		edges : List[Edge]
	) -> Float[Tensor, "batch pos head_idx d_model"]:
		'''
		Patches the path from sender to receiver heads in the model with the corrupted activations
		Edges are tuples (sender, receiver),
		where sender = (sender_act_name, sender_act_index_tuple)
		and receiver = (receiver_act_name, receiver_act_index_tuple)

		Args:
			activation: The activations of the model.
			hook: The hook point in the model.
			edges: The edges to patch.
		
		Returns:	
			The patched activations.
		'''

		# Check all edges in the list
		for edge in edges:
			sender, receiver = edge.sender, edge.receiver
			sender_hook_name, sender_hook_index = sender.name, sender.index
			receiver_act_name, receiver_act_index = receiver.name, receiver.index
			# If the receiver is the same as the hook, then we need to patch the path
			if hook.name == receiver_act_name:
				# Any information that was sent from the sender to the receiver is corrupted
				# We only care about direct paths, i.e., through the residual stream, so we can simply subtract the clean sender activations and add the corrupted sender activations
				activation[receiver_act_index] += (self.corrupted_cache[sender_hook_name][sender_hook_index] - self.clean_cache[sender_hook_name][sender_hook_index])
		return activation

	def get_batch_mean_kl_divergence(
			self,
			clean_logits_last_token: Float[Tensor, "batch seq d_vocab"],
			patched_logits_last_token: Float[Tensor, "batch seq d_vocab"]
		) -> float:
		'''
		The KL divergence is calculated as follows:
		1. Select the last logits from each sequence for comparison. These logits correspond to the final token
		in each sequence.
		2. Convert the clean and patched logits to log probabilities using log_softmax.
		3. Convert the clean log probabilities to probabilities using exp.
		4. Calculate the KL divergence using the formula: KL(P || Q) = sum(P * (log(P) - log(Q))), where P is the
		clean probability distribution and Q is the patched probability distribution.
		5. Take the average KL divergence across all sequences in the batch.

		Args:
			clean_logits: The logits obtained from the clean (original) model.
			patched_logits: The logits obtained from the patched model.

		Returns:
			The average KL divergence between the clean and patched probability distributions.
		'''

		# Select the last logits from each sequence for comparison
		last_token_indices = self.clean_dataset.word_idx['end'] # last token index is same for clean and patched logit pairs
		patched_logits_last_token = patched_logits_last_token[torch.arange(self.clean_dataset.N), last_token_indices]
		clean_logits_last_token = clean_logits_last_token[torch.arange(self.clean_dataset.N), last_token_indices]
		
		# Take logprobs for KL divergence calculation
		patched_logprobs = torch.nn.functional.log_softmax(patched_logits_last_token, dim=-1)
		clean_logprobs = torch.nn.functional.log_softmax(clean_logits_last_token, dim=-1)

		# Calculate probs from logprobs (instead of directly from logits) to avoid numerical instability
		clean_probs = torch.exp(clean_logprobs)

		# Calculate the KL divergence for each example in the batch
		kl_div = (clean_probs * (torch.log(clean_probs) - patched_logprobs)).sum(dim=-1)

		# Return the average KL divergence over all batches
		return kl_div.mean().item()

	def ablate_paths(
		self,
		edges : List[Edge]
	) -> float:
		'''
		Takes a list of "dead" edges, performs path patching by corrupting these edges,
		and calculates the average KL divergence between the clean and patched logits.

		This is akin to ablation of these paths in the computational graph.
		By corrupting them with a contrastive signal, we can measure the impact of these paths on the model's predictions.
		'''
		
		receivers = [edge.receiver for edge in edges]
		receiver_hook_names = [receiver.name for receiver in receivers]

		# Filter to only use the receiver hooks in the list of receiver hook names
		# We only need to use receiver hooks, because we corrupt activtions at the path destinations
		receiver_hook_names_filter = lambda name: name in receiver_hook_names

		# Initialize hook function
		hook_fn = partial(
			self.patch_path,
			edges=edges
		)
		
		# Run forward pass with path patching hooks
		self.model.add_hook(receiver_hook_names_filter, hook_fn, level=1)
		patched_logits = self.model.run_with_hooks(self.clean_dataset.toks)
		self.model.reset_hooks()

		# Calculate the average KL divergence between the patched logits and the original logits
		kl_div = self.get_batch_mean_kl_divergence(self.clean_logits, patched_logits)
		return kl_div
	
